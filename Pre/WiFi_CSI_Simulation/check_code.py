"""Syntax, imports and software contracts. Does not call simulation/train/plot entry points."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    files = sorted((ROOT / "csi_sim").glob("*.py")) + sorted(ROOT.glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = [p.stem for p in (ROOT / "csi_sim").glob("*.py") if p.stem != "__init__"]
    for module in modules:
        importlib.import_module(f"csi_sim.{module}")
    from csi_sim.config import load_config
    from csi_sim.utils import environment
    from matplotlib.font_manager import findfont
    cfg = load_config(ROOT / "configs/default.json")
    font = findfont(cfg["plots"]["font"], fallback_to_default=False)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_contracts.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"status": "passed" if result.wasSuccessful() else "failed", "python_files_parsed": len(files),
              "modules_imported": modules, "software_tests": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "font": font, "environment": environment(),
              "checks": ["AST syntax", "module imports", "configuration", "hand-written score/label contracts", "CNN parameter count"],
              "validation_scope": "software_only; experiment status is recorded under outputs",
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    output = ROOT / "validation/code_checks.v3.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validation report: {output}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
