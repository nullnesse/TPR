#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверки чистого ядра ``solver_core`` (стандартный unittest)."""

import json
import math
import unittest

import solver_core as core


class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = core.Model()
        cls.emv = core.emv(cls.model)
        cls.eol = core.eol(cls.model)

    def test_model_and_expected(self):
        self.assertEqual(self.model.defects, [100, 200, 300])
        self.assertEqual(self.model.repair_costs, [50, 100, 150])
        self.assertAlmostEqual(core.expected([1, 2, 3], [.2, .3, .5]), 2.3, places=12)
        self.assertAlmostEqual(core.variance([1, 2, 3], [.2, .3, .5]), .61, places=12)
        with self.assertRaises(ValueError):
            core.expected([1, 2], [.5, .4])

    def test_emv_base(self):
        self.assertEqual(self.emv["expected_costs"], {"A": 70.0, "B": 63.0})
        self.assertEqual(self.emv["expected_repair"], {"A": 70.0, "B": 100.0})
        self.assertEqual(self.emv["expected_defects"], {"A": 140.0, "B": 200.0})
        self.assertEqual(self.emv["choice"], "B")
        self.assertAlmostEqual(self.emv["saving_vs_A"], 7.0, places=12)
        self.assertAlmostEqual(self.emv["per_switch_saving_rub"], 0.7, places=12)
        self.assertEqual(self.emv["excess_over_best_mean"], {"A": 7.0, "B": 0.0})

    def test_eol_joint_states(self):
        self.assertEqual(len(self.eol["states"]), 9)
        self.assertAlmostEqual(sum(s["probability"] for s in self.eol["states"]), 1.0, places=12)
        self.assertAlmostEqual(self.eol["expected_minimum"], 45.35, places=12)
        self.assertAlmostEqual(self.eol["expected_regret"]["A"], 24.65, places=12)
        self.assertAlmostEqual(self.eol["expected_regret"]["B"], 17.65, places=12)
        self.assertEqual(self.eol["choice"], "B")
        for state in self.eol["states"]:
            self.assertAlmostEqual(
                state["regret_a"], state["cost_a"] - state["minimum"], places=12,
            )
            self.assertAlmostEqual(
                state["regret_b"], state["cost_b"] - state["minimum"], places=12,
            )

    def test_joint_marginals(self):
        states = self.eol["states"]
        for i, p in enumerate(self.model.p_a):
            self.assertAlmostEqual(sum(s["probability"] for s in states if s["i"] == i), p)
        for j, p in enumerate(self.model.p_b):
            self.assertAlmostEqual(sum(s["probability"] for s in states if s["j"] == j), p)

    def test_risk_and_difference(self):
        risk = core.risk(self.model)
        self.assertAlmostEqual(risk["sds"]["A"], math.sqrt(1100), places=12)
        self.assertAlmostEqual(risk["sds"]["B"], math.sqrt(1500), places=12)
        self.assertAlmostEqual(risk["p_a_cheaper"], 0.55, places=12)
        self.assertAlmostEqual(risk["p_b_cheaper"], 0.45, places=12)
        self.assertAlmostEqual(risk["expected_diff"], -7.0, places=12)
        self.assertAlmostEqual(risk["diff_variance"], 2600.0, places=9)
        self.assertAlmostEqual(risk["diff_sd"], math.sqrt(2600), places=9)

    def test_mean_sd(self):
        base = core.mean_sd(self.model, 0)
        self.assertEqual(base["choice"], "B")
        high = core.mean_sd(self.model, 2)
        self.assertEqual(high["choice"], "A")
        self.assertAlmostEqual(high["scores"]["A"], 136.332495807108, places=9)
        self.assertAlmostEqual(high["scores"]["B"], 140.45966692414834, places=9)
        self.assertAlmostEqual(
            high["lambda_star"], 7.0 / (math.sqrt(1500) - math.sqrt(1100)), places=12,
        )
        self.assertTrue(high["threshold"]["feasible"])

    def test_utility_presets(self):
        base = core.utility(self.model, 100)
        self.assertEqual(base["choice"], "B")
        self.assertAlmostEqual(base["A"]["ce"], 76.35731004430394, places=9)
        self.assertAlmostEqual(base["B"]["ce"], 70.37852435545759, places=9)
        strong = core.utility(core.Model(discount=20), 2)
        self.assertEqual(strong["choice"], "B")
        self.assertAlmostEqual(strong["A"]["ce"], 145.39482981406746, places=9)
        self.assertAlmostEqual(strong["B"]["ce"], 127.59205439138516, places=9)
        moderate = core.utility(core.Model(discount=20), 100)
        self.assertEqual(moderate["choice"], "A")
        self.assertAlmostEqual(moderate["A"]["ce"], 76.35731004430394, places=9)
        self.assertAlmostEqual(moderate["B"]["ce"], 87.37852435545759, places=9)

    def test_sensitivity_and_large_batch(self):
        s = core.sensitivity(self.model)
        self.assertAlmostEqual(s["delta_star"], 30.0, places=12)
        self.assertAlmostEqual(s["c_star"], 37.0 / 60.0, places=12)
        self.assertAlmostEqual(s["rate_b_star"], 0.0214, places=12)
        large = core.emv(core.Model(N=20_000))
        self.assertEqual(large["expected_costs"], {"A": 140.0, "B": 163.0})
        self.assertEqual(large["choice"], "A")

    def test_simulation_reproducible(self):
        first = core.simulation(self.model, runs=5_000, seed=7)
        second = core.simulation(self.model, runs=5_000, seed=7)
        self.assertEqual(first, second)
        self.assertLess(abs(first["mean_A"] - 70.0), 1.5)
        self.assertLess(abs(first["mean_B"] - 63.0), 1.5)
        self.assertLess(abs(first["p_a_cheaper"] - 0.55), 0.03)

    def test_build_core_results_is_json_safe(self):
        result = core.build_core_results(
            self.model, lambda_=2, rho=2, runs=1_000, seed=42, include_simulation=True,
        )
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["scope"], "core")
        self.assertEqual(result["emv"]["expected_costs"], {"A": 70.0, "B": 63.0})
        self.assertEqual(result["eol"]["expected_regret"], {"A": 24.65, "B": 17.65})
        self.assertEqual(result["mean_sd"]["choice"], "A")
        self.assertEqual(result["utility"]["choice"], "B")
        json.dumps(result, allow_nan=False)

    def test_price_shift_invariance(self):
        shifted = core.build_core_results(
            core.Model(price_a=100), lambda_=0, rho=100,
            runs=1_000, seed=42, include_simulation=False,
        )
        self.assertEqual(shifted["emv"]["expected_costs"], {"A": 170.0, "B": 163.0})
        self.assertAlmostEqual(shifted["eol"]["expected_regret"]["A"], 24.65, places=12)
        self.assertAlmostEqual(shifted["risk"]["sds"]["A"], math.sqrt(1100), places=12)
        self.assertAlmostEqual(shifted["utility"]["A"]["ce"], 176.35731004430394, places=9)
        self.assertEqual(shifted["emv"]["choice"], "B")

    def test_degenerate_and_zero_cost(self):
        degenerate = core.Model(p_a=(1, 0, 0), p_b=(0, 1, 0))
        self.assertEqual(core.emv(degenerate)["expected_costs"], {"A": 50.0, "B": 63.0})
        self.assertEqual(core.eol(degenerate)["expected_regret"], {"A": 0.0, "B": 13.0})
        self.assertEqual(core.risk(degenerate)["sds"], {"A": 0.0, "B": 0.0})

        free = core.Model(c=0)
        self.assertEqual(core.emv(free)["expected_costs"], {"A": 0.0, "B": -37.0})
        self.assertAlmostEqual(core.eol(free)["expected_regret"]["A"], 37.0, places=9)
        self.assertAlmostEqual(core.eol(free)["expected_regret"]["B"], 0.0, places=9)
        self.assertIsNone(core.sensitivity(free)["rate_b_star"])
        self.assertAlmostEqual(core.utility(free, 100)["A"]["ce"], 0.0, places=9)
        self.assertAlmostEqual(core.utility(free, 100)["B"]["ce"], -37.0, places=9)

    def test_choice_tie(self):
        self.assertEqual(core.choose({"A": 70.0, "B": 70.0}), "tie")
        self.assertEqual(core.choose({"A": 70.0, "B": 70.0000000001}), "tie")
        self.assertEqual(core.choose({"A": 70.0, "B": 71.0}), "A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
