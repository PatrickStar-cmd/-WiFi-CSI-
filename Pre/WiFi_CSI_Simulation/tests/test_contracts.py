"""Small software checks using hand-written labels/scores; no simulated CSI."""
from pathlib import Path
import unittest

import numpy as np

from csi_sim.config import changed, experiment_plan, load_config
from csi_sim.dataset import split_ids
from csi_sim.evaluate import classification, event_metrics
from csi_sim.fusion import confirm_states, fuse, resources, select_threshold

ROOT = Path(__file__).resolve().parents[1]


class SoftwareContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / "configs/default.json")

    def test_group_splits_are_disjoint(self):
        split = split_ids(self.cfg)
        a, b, c = (set(split[name]) for name in ("train", "validation", "test"))
        self.assertEqual((len(a), len(b), len(c)), (54, 18, 18))
        self.assertFalse(a & b or a & c or b & c)
        self.assertEqual(len(a | b | c), 90)

    def test_confirm_and_reset(self):
        scores = np.array([0.1, 0.8, 0.9, np.nan, 0.9, 0.9, 0.1, 0.1])
        self.assertEqual(confirm_states(scores, 0.5, 2).tolist(), [0, 0, 1, -1, -1, 1, 1, 0])

    def test_all_missing_fusion(self):
        score, count = fuse(np.full((1, 4), np.nan), np.zeros((1, 4), bool), np.ones((1, 4)), "four_quality")
        self.assertTrue(np.isnan(score[0]))
        self.assertEqual(count[0], 0)

    def test_quality_weight_cap_and_mask(self):
        scores = np.array([[0.0, 1.0, 0.9, 0.2]])
        good = np.array([[True, True, False, False]])
        quality = np.array([[1.0, 100.0, 1000.0, 1.0]])
        score, count = fuse(scores, good, quality, "four_quality", 10)
        self.assertAlmostEqual(score[0], 10 / 11)
        self.assertEqual(count[0], 2)

    def test_probability_fusion_float64_range(self):
        quality = np.array([[103.23, 91.46, 94.872, 102.643]], dtype=np.float32)
        score, _ = fuse(np.ones((1, 4)), np.ones((1, 4), bool), quality, "four_quality")
        self.assertEqual(score[0], 1.0)

    def test_log_complement_fusion_matches_probability_average(self):
        probabilities = np.array([[0.1, 0.4, 0.8, 0.99], [0.2, 0.3, 0.6, 0.9]])
        valid = np.array([[True, True, True, True], [True, False, True, False]])
        quality = np.array([[1, 2, 3, 4], [3, 1, 2, 4]], dtype=np.float32)
        expected, _ = fuse(probabilities, valid, quality, "four_quality")
        evidence, _ = fuse(-np.log1p(-probabilities), valid, quality, "four_quality", score_domain="negative_log_static_probability")
        np.testing.assert_allclose(-np.expm1(-evidence), expected, atol=1e-14, rtol=1e-14)

    def test_high_confidence_order_is_retained(self):
        evidence = np.logaddexp(0.0, np.array([[40., 41., 42., 43.], [60., 61., 62., 63.]]))
        fused, _ = fuse(evidence, np.ones((2, 4), bool), np.ones((2, 4)), "four_quality", score_domain="negative_log_static_probability")
        self.assertGreater(fused[1], fused[0])
        self.assertAlmostEqual(fused[1] - fused[0], 20.0)

    def test_post_confirmation_threshold_constraint(self):
        sequence = {"score": np.array([0.1, 0.2, 0.1, 0.9, 0.7, 0.8, 0.8, 0.9]),
                    "truth": np.array([0, 0, 0, 0, 1, 1, 1, 1])}
        result = select_threshold([sequence], self.cfg)
        self.assertLessEqual(result["validation_fpr"], 0.01)
        self.assertGreater(result["validation_f1"], 0)

    def test_unknown_is_a_separate_confusion_column(self):
        result = classification([0, 0, 1, 1], [0, -1, 1, -1], [0.1, np.nan, 0.8, np.nan])
        self.assertEqual(result["confusion"], [[1, 0, 1], [0, 1, 1]])
        self.assertEqual(result["coverage"], 0.5)

    def test_existing_alarm_is_separate_from_onset_detection(self):
        times = np.arange(0, 3, 0.2)
        truth = ((times >= 1) & (times < 2)).astype(int)
        event = [{"start": 1.0, "end": 2.0, "kind": "walk", "event_id": 0}]
        result, _ = event_metrics(times, truth, np.ones(len(times), dtype=int), event, 0.2, 3)
        self.assertEqual(result["detected_events"], 0)
        self.assertEqual(result["alarm_at_onset_events"], 1)

    def test_delay_and_stop_matching(self):
        times = np.arange(0, 3, 0.2)
        truth = ((times >= 1) & (times < 2)).astype(int)
        state = ((times >= 1.4 - 1e-9) & (times < 2.4 - 1e-9)).astype(int)
        event = [{"start": 1.0, "end": 2.0, "kind": "wave", "event_id": 0}]
        result, _ = event_metrics(times, truth, state, event, 0.2, 3)
        self.assertAlmostEqual(result["onset_median_s"], 0.4)
        self.assertAlmostEqual(result["stop_median_s"], 0.4)

    def test_equal_total_transmit_budget(self):
        one = resources("AC", 100, 60, self.cfg)
        dual = resources("dual_AC_AD", 100, 60, self.cfg)
        four = resources("four_quality", 50, 60, self.cfg)
        self.assertEqual(one["total_packets"], four["total_packets"])
        self.assertEqual(dual["transmit_pps"], four["transmit_pps"])
        self.assertEqual(four["receiver_count"], 2)

    def test_config_change_leaves_original_intact(self):
        other = changed(self.cfg, {"radio.packet_rate_hz": 50})
        self.assertEqual(self.cfg["radio"]["packet_rate_hz"], 100)
        self.assertEqual(other["radio"]["packet_rate_hz"], 50)
        self.assertIn("E5", experiment_plan(other)["stages"])

    def test_multiscale_noise_normalization(self):
        from csi_sim.optimized import multiscale_features
        a = np.random.default_rng(17).normal(1, .02, (2, 4, 56, 100))
        noise = np.full(4, .0004)
        base = multiscale_features(a, noise, 100, [.2,.4,1.0])
        scaled = multiscale_features(a * 3, noise * 9, 100, [.2,.4,1.0])
        np.testing.assert_allclose(base, scaled, atol=1e-6)

    def test_short_feature_uses_recent_suffix(self):
        from csi_sim.optimized import multiscale_features
        a = np.random.default_rng(18).normal(1, .02, (2, 4, 56, 100))
        before = multiscale_features(a, np.full(4,.0004),100,[.2,.4,1.0])
        a[...,:60] += np.linspace(0,10,60)
        after = multiscale_features(a, np.full(4,.0004),100,[.2,.4,1.0])
        np.testing.assert_allclose(before[...,:16],after[...,:16])

    def test_hysteresis_replays_original_confirmation(self):
        from csi_sim.optimized import hysteresis_states
        scores = np.array([0,2,2,0,0,np.nan,2,2,0,2,0,0],dtype=float)
        np.testing.assert_array_equal(hysteresis_states(scores,1.,1.,2,2),confirm_states(scores,1.,2))

    def test_optimized_operating_point_constraints(self):
        from csi_sim.optimized import select_operating_point
        truth = {'label':np.array([0,0,1,1,0,0]),'motion_fraction':np.array([0,0,1,1,.5,0]),
                 'kind':np.array(['still','still','wave','wave','still','still'])}
        protocol = {'threshold_quantiles':17,'overall_fpr_limit':.25,'steady_static_fpr_limit':0.,
                    'confirmation_candidates':[[1,1]],'off_threshold_ratios':[1.]}
        result = select_operating_point([{'score':np.array([0,.1,.8,.9,.6,0]),'truth':truth}],protocol)
        self.assertLessEqual(result['validation_fpr'],.25)
        self.assertEqual(result['validation_steady_fpr'],0.)
        self.assertEqual(result['validation_wave_recall'],1.)

    def test_cnn_architecture_contract(self):
        from csi_sim.cnn import TinyCNN
        network = TinyCNN()
        self.assertEqual(sum(p.numel() for p in network.parameters()), 7121)
        self.assertEqual(network.network[0].in_channels, 56)
        self.assertEqual(network.network[-1].out_features, 1)


if __name__ == "__main__":
    unittest.main()
