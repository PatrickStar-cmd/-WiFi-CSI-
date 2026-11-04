"""CLI. `plan` is read-only; data generation starts only under the `run` command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from csi_sim.config import changed, experiment_plan, load_config

ROOT = Path(__file__).resolve().parent
STAGES = ["E0", "E1", "E2", "E3", "E4", "E5", "demo", "coverage", "region"]


def parser():
    p = argparse.ArgumentParser(description="Wi-Fi CSI人体运动检测仿真与可视化")
    commands = p.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="只显示配置与实验计划")
    plan.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    run = commands.add_parser("run", help="执行明确选择的仿真阶段")
    run.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    run.add_argument("--stages", nargs="+", choices=STAGES + ["core", "all"], required=True)
    run.add_argument("--output", type=Path, required=True, help="新建结果目录")
    run.add_argument("--models", nargs="+", choices=["B0", "B0q", "B1", "B1a", "B2", "B3"])
    run.add_argument("--device", choices=["cpu", "cuda"])
    run.add_argument("--no-plots", action="store_true", help="保存实验结果，图表稍后通过plot生成")
    plot = commands.add_parser("plot", help="读取已有结果生成图表与动画")
    plot.add_argument("--run-dir", type=Path, required=True)
    plot.add_argument("--no-animation", action="store_true")
    return p


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    if args.command == "plan":
        print(json.dumps(experiment_plan(load_config(args.config)), ensure_ascii=False, indent=2))
    elif args.command == "plot":
        from csi_sim.plots import render_run
        render_run(args.run_dir.resolve(), animation=not args.no_animation)
    else:
        cfg = load_config(args.config)
        overrides = {}
        if args.models:
            overrides["training.models"] = args.models
        if args.device:
            overrides["training.device"] = args.device
        cfg = changed(cfg, overrides)
        stages = []
        for stage in args.stages:
            expanded = ["E0", "E1", "E2", "demo"] if stage == "core" else (STAGES[:-1] if stage == "all" else [stage])
            stages.extend(s for s in expanded if s not in stages)
        from csi_sim.runner import run
        result = run(cfg, args.output, stages, render=not args.no_plots)
        print(f"Results: {result}")


if __name__ == "__main__":
    main()
