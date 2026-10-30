#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверки CLI и совместимости ``solver.py``."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOLUTION = Path(__file__).resolve().parent
SOLVER = SOLUTION / "solver.py"
RESULTS = SOLUTION / "results.json"
TMP_ROOT = ROOT / ".tmp"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SOLVER), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


class TestCli(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(dir=str(TMP_ROOT))
        self.tmp = Path(self._tmp.name)
        self.results_before = RESULTS.read_bytes()

    def tearDown(self):
        self._tmp.cleanup()

    def test_core_text(self):
        proc = run_cli("--scope", "core", "--format", "text")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("EMV: A=70,00; B=63,00; выбор B", proc.stdout)
        self.assertIn("EOL: A=24,65; B=17,65; выбор B", proc.stdout)
        self.assertIn("mean–SD (λ=0,00)", proc.stdout)
        self.assertIn("CE (ρ=100,00)", proc.stdout)
        self.assertEqual(RESULTS.read_bytes(), self.results_before)

    def test_core_json_does_not_write_without_output(self):
        core_path = SOLUTION / "results_core.json"
        before = core_path.stat().st_mtime_ns if core_path.exists() else None
        proc = run_cli("--scope", "core", "--format", "json", "--runs", "1000")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(data["scope"], "core")
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["emv"]["expected_costs"], {"A": 70.0, "B": 63.0})
        self.assertEqual(data["simulation"]["runs"], 1000)
        after = core_path.stat().st_mtime_ns if core_path.exists() else None
        self.assertEqual(after, before)

    def test_core_output(self):
        target = self.tmp / "core.json"
        proc = run_cli(
            "--scope", "core", "--format", "json", "--runs", "1000",
            "--output", str(target),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data["scope"], "core")
        self.assertEqual(data["emv"]["expected_costs"], {"A": 70.0, "B": 63.0})

    def test_full_output_preserves_history(self):
        target = self.tmp / "full.json"
        proc = run_cli("--output", str(target))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(list(data), [
            "model", "emv", "tree", "criteria", "risk", "probability",
            "utility", "evpi", "evsi", "sensitivity", "split", "simulation",
        ])
        self.assertEqual(data["emv"]["expected_costs"], {"A": 70.0, "B": 63.0})
        self.assertEqual(target.read_bytes(), RESULTS.read_bytes())
        self.assertEqual(RESULTS.read_bytes(), self.results_before)

    def test_config_and_cli_priority(self):
        config = self.tmp / "model.json"
        config.write_text(
            json.dumps({"model": {"discount": 20}, "preferences": {"rho": 2}}),
            encoding="utf-8",
        )
        first = run_cli("--scope", "core", "--format", "json", "--runs", "1000",
                        "--config", str(config))
        self.assertEqual(first.returncode, 0, first.stderr)
        data = json.loads(first.stdout)
        self.assertEqual(data["emv"]["choice"], "A")
        self.assertEqual(data["utility"]["choice"], "B")

        second = run_cli(
            "--scope", "core", "--format", "json", "--runs", "1000",
            "--config", str(config), "--discount", "37",
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        data = json.loads(second.stdout)
        self.assertEqual(data["emv"]["choice"], "B")
        self.assertEqual(data["utility"]["choice"], "B")
        self.assertAlmostEqual(data["utility"]["B"]["ce"], 110.59, places=2)

    def test_core_rejects_full_only_flags(self):
        proc = run_cli("--lambda", "2")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--scope core", proc.stderr)
        self.assertEqual(RESULTS.read_bytes(), self.results_before)

    def test_invalid_input_exit_2(self):
        proc = run_cli("--scope", "core", "--discount", "-1")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("Ошибка", proc.stderr)
        self.assertEqual(RESULTS.read_bytes(), self.results_before)

    def test_unknown_config_key_exit_2(self):
        config = self.tmp / "bad.json"
        config.write_text('{"model": {"unknown": 1}}', encoding="utf-8")
        proc = run_cli("--scope", "core", "--config", str(config))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("неизвестные ключи", proc.stderr)

    def test_help(self):
        proc = run_cli("--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--scope", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
