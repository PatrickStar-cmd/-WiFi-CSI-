from copy import deepcopy

import numpy as np
import pytest

from studio.engine import decode_people, evaluate_people
from studio.geometry import Channel3D, pose
from studio.project import duration, person_project, preset, validate, validate_route


def test_independent_tracks_share_time_and_mix_complex_echoes():
    project=validate(preset(3))
    validate_route(project)
    shorter=project['people'][0]
    shorter['actions'].pop()
    individual=[person_project(project,p) for p in project['people']]
    assert duration(project)==30 and duration(shorter)==16
    assert individual[0]['actions'][-1]=={'kind':'still','duration':14}
    times=np.array([2.,7.,13.,19.,25.])
    truth=np.column_stack([pose(p,times)['truth'] for p in individual])
    np.testing.assert_array_equal(truth,[[0,0],[1,0],[1,1],[0,1],[0,0]])
    channel=Channel3D(project)
    single=[Channel3D(p) for p in individual]
    for name in channel.links:
        background=channel.background[name]
        expected=background+sum(one.response(name,times)-one.background[name] for one in single)
        np.testing.assert_allclose(channel.response(name,times),expected,atol=1e-12)
    assert len(shorter['actions'])==3


def test_calibration_schema_ignores_scenario_tracks_and_labels():
    from studio.multi_detector import geometry_config

    project=preset(3)
    updated=deepcopy(project)
    updated['people'][0]['name']='改过的名字'
    updated['people'][0]['actions']=[{'kind':'still','duration':30}]
    updated['people'][1]['points'][0]['position'][0]+=.2
    updated['seed']+=7
    assert geometry_config(project)==geometry_config(updated)


def test_joint_scores_reject_ambiguous_windows_and_count_all_windows():
    posterior=np.array([[.8,.1,.05,.05],[.1,.7,.08,.12],[.05,.1,.7,.15],
                        [.1,.05,.10,.75],[.26,.25,.25,.24],[.1,.7,.1,.1]])
    scores,states=decode_people({'posterior':posterior,'valid':[1,1,1,1,1,0]},.50)
    expected=np.array([[0,0],[1,0],[0,1],[1,1],[-1,-1],[-1,-1]])
    np.testing.assert_array_equal(states,expected)
    assert scores[1,0]==pytest.approx(.82) and np.isnan(scores[-1]).all()
    truth=np.array([[0,0],[1,0],[0,1],[1,1],[0,0],[1,0]])
    report,joint=evaluate_people(truth,states,np.arange(0,6)*.2,[[],[]])
    assert joint['jointAccuracy']==pytest.approx(4/6)
    assert joint['coverage']==pytest.approx(4/6)
    assert joint['combinations'][1]['unknown']==1
    assert report[0]['recall']==pytest.approx(2/3) and report[1]['crossTalk']==0


def test_p2_confirmation_checks_weak_onset_only():
    from studio.multi_detector_v3 import _causal_joint_scores

    scores=np.array([[.95,.40],[.95,.40],[.95,.01],       # Two brief windows stay quiet.
                     [.95,.40],[.95,.40],[.95,.40],       # Three consecutive windows confirm.
                     [.95,.07],[.95,.01],[.95,.92],       # Sustain, stop, then immediate onset.
                     [.95,.40],[.95,.40],[.01,.40]])      # Invalid resets. P2 alone starts immediately.
    valid=np.ones(len(scores),dtype=bool); valid[9]=False
    posterior=_causal_joint_scores(scores,valid)
    _,states=decode_people({'posterior':posterior,'valid':valid},.44)
    np.testing.assert_array_equal(states[:,1],[0,0,0,0,0,1,1,0,1,-1,0,1])
    np.testing.assert_array_equal(states[:9,0],np.ones(9))

    changed=scores.copy(); changed[5:]=.01
    np.testing.assert_array_equal(_causal_joint_scores(changed,valid)[:5],posterior[:5])
