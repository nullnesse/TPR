#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверки корректности solver.py (стандартный unittest)."""

import math
import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace

import solver


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = solver.Model()
        cls.res = solver.build_results(cls.model)

    def test_probabilities_sum_to_one(self):
        self.assertAlmostEqual(sum(self.model.p_a), 1.0, places=12)
        self.assertAlmostEqual(sum(self.model.p_b), 1.0, places=12)

    def test_defects_and_repair(self):
        self.assertEqual(self.model.defects, [100, 200, 300])
        self.assertEqual(self.model.repair_costs, [50, 100, 150])

    def test_expected_values(self):
        emv = self.res["emv"]
        self.assertAlmostEqual(emv["expected_defects"]["A"], 140.0, places=9)
        self.assertAlmostEqual(emv["expected_defects"]["B"], 200.0, places=9)
        self.assertAlmostEqual(emv["expected_repair"]["A"], 70.0, places=9)
        self.assertAlmostEqual(emv["expected_repair"]["B"], 100.0, places=9)

    def test_best_and_saving(self):
        emv = self.res["emv"]
        self.assertEqual(emv["best"], "B")
        self.assertAlmostEqual(emv["expected_costs"]["A"], 70.0, places=9)
        self.assertAlmostEqual(emv["expected_costs"]["B"], 63.0, places=9)
        self.assertAlmostEqual(emv["saving_vs_A"], 7.0, places=9)
        self.assertAlmostEqual(emv["per_switch_saving_rub"], 0.7, places=9)

    def test_eol_and_evpi_identity(self):
        """Сожаление сравнивает действия в каждом исходе, а не лучшие средние."""
        emv = self.res["emv"]
        self.assertAlmostEqual(emv["eol"]["A"], 24.65)
        self.assertAlmostEqual(emv["eol"]["B"], 17.65)
        self.assertEqual(emv["excess_over_best_mean"], {"A": 7, "B": 0})
        self.assertAlmostEqual(min(emv["eol"].values()), self.res["evpi"]["both_independent"])
        for name, own, other in (("A", "A", "B"), ("B", "B", "A")):
            regret = sum(pa * pb * max(a - b, 0)
                         for a, pa in zip(self.model.costs(own), self.model.probs(own))
                         for b, pb in zip(self.model.costs(other), self.model.probs(other)))
            self.assertAlmostEqual(emv["eol"][name], regret)

    def test_savage_joint_states(self):
        joint = self.res["criteria"]["savage_joint"]
        self.assertEqual(len(joint["states"]), 9)
        self.assertEqual(joint["scores"], {"A": 137, "B": 63})
        self.assertEqual(joint["choice"], "B")
        self.assertEqual(self.res["criteria"]["savage"]["scores"], {"A": 37, "B": 0})

    def test_tree(self):
        tree = self.res["tree"]
        self.assertAlmostEqual(tree["chance_nodes"]["A"], 70.0, places=9)
        self.assertAlmostEqual(tree["chance_nodes"]["B"], 63.0, places=9)
        self.assertAlmostEqual(tree["root"], 63.0, places=9)
        self.assertEqual(tree["chosen"], "B")

    def test_criteria_all_choose_b(self):
        crit = self.res["criteria"]
        self.assertEqual(crit["emv"]["choice"], "B")
        self.assertEqual(crit["wald"]["choice"], "B")
        self.assertEqual(crit["minimin"]["choice"], "B")
        self.assertEqual(crit["laplace"]["choice"], "B")
        self.assertEqual(crit["savage"]["choice"], "B")
        for a, block in crit["hurwicz"].items():
            self.assertEqual(block["choice"], "B", f"alpha={a}")

    def test_thresholds(self):
        s = self.res["sensitivity"]
        self.assertAlmostEqual(s["delta_star"], 30.0, places=9)
        self.assertAlmostEqual(s["c_star"], 37.0 / 60.0, places=9)
        self.assertAlmostEqual(s["rate_b_star"], 0.0214, places=9)

    def test_risk(self):
        risk = self.res["risk"]
        self.assertAlmostEqual(risk["sds"]["A"], math.sqrt(1100), places=9)
        self.assertAlmostEqual(risk["sds"]["B"], math.sqrt(1500), places=9)
        self.assertAlmostEqual(risk["p_a_cheaper"], 0.55, places=9)
        self.assertAlmostEqual(risk["p_b_cheaper"], 0.45, places=9)
        self.assertAlmostEqual(risk["expected_diff"], -7.0, places=9)

    def test_probability_binomial_model(self):
        binomial = self.res["probability"]["binomial"]
        self.assertAlmostEqual(binomial["A"]["within_var"], 137.6, places=9)
        self.assertAlmostEqual(binomial["A"]["variance"], 4537.6, places=9)
        self.assertAlmostEqual(binomial["B"]["within_var"], 195.4, places=9)
        self.assertAlmostEqual(binomial["B"]["variance"], 6195.4, places=9)
        self.assertAlmostEqual(binomial["A"]["cost_sd"], 33.6809, places=3)

    def test_probability_bayes_example(self):
        bayes = self.res["probability"]["bayes"]
        self.assertAlmostEqual(bayes["posterior_mean"], 0.0122474, places=6)
        self.assertAlmostEqual(sum(bayes["posterior"]), 1.0, places=12)
        self.assertAlmostEqual(bayes["cost_a"], 61.237, places=2)
        self.assertEqual(bayes["choice"], "A")

    def test_evpi(self):
        evpi = self.res["evpi"]
        self.assertAlmostEqual(evpi["after_choice"], 0.0, places=12)
        self.assertAlmostEqual(evpi["one_batch_A"], 9.1, places=9)
        self.assertAlmostEqual(evpi["one_batch_B"], 12.9, places=9)
        self.assertAlmostEqual(evpi["both_independent"], 17.65, places=9)
        self.assertAlmostEqual(evpi["both_comonotonic"], 7.8, places=9)
        self.assertAlmostEqual(evpi["e_perfect_comonotonic"], 55.2, places=9)

    def test_joint_distributions_preserve_marginals(self):
        """Зависимость меняет сочетания состояний, сохраняя каждое распределение."""
        for model in (self.model, replace(self.model, rates=(.03, .01, .02)),
                      replace(self.model, p_a=(1, 0, 0))):
            for coupling in ("independent", "comonotonic"):
                joint = solver.joint_probabilities(model, coupling)
                self.assertTrue(all(p >= 0 for row in joint for p in row))
                for i, p in enumerate(model.p_a):
                    self.assertAlmostEqual(sum(joint[i]), p)
                for j, p in enumerate(model.p_b):
                    self.assertAlmostEqual(sum(row[j] for row in joint), p)
                for name, weights in (("A", [sum(row) for row in joint]),
                                      ("B", [sum(row[j] for row in joint) for j in range(3)])):
                    self.assertAlmostEqual(sum(p * c for p, c in zip(weights, model.costs(name))),
                                           model.expected_cost(name))

    def test_bayes_price_a(self):
        result = solver.probability_block(solver.Model(price_a=100), 50, 1)["bayes"]
        self.assertAlmostEqual(result["cost_a"], 172.0175189968765)
        self.assertEqual(result["cost_b"], 163)
        self.assertEqual(result["choice"], "B")

    def test_finite_population_bayes(self):
        result = self.res["probability"]["bayes_finite"]
        weights = [p * math.comb(self.model.N - d, 50)
                   for p, d in zip(self.model.p_a, (100, 200, 300))]
        for actual, weight in zip(result["posterior"], weights):
            self.assertAlmostEqual(actual, weight / sum(weights))
        self.assertAlmostEqual(result["cost_a"], 61.22046557208348)
        self.assertEqual(result["choice"], "A")

    def test_price_shift_invariance(self):
        """Общая цена сдвигает затраты и CE, но не EOL, EVPI и EVSI."""
        shifted = solver.build_results(solver.Model(price_a=100))
        for name in ("A", "B"):
            self.assertAlmostEqual(shifted["emv"]["expected_costs"][name],
                                   self.res["emv"]["expected_costs"][name] + 100)
            self.assertAlmostEqual(shifted["emv"]["eol"][name], self.res["emv"]["eol"][name])
        for key in ("both_independent", "both_comonotonic", "one_batch_A", "one_batch_B"):
            self.assertAlmostEqual(shifted["evpi"][key], self.res["evpi"][key])
        for original, row in zip(self.res["evsi"]["table"], shifted["evsi"]["table"]):
            for name in ("A", "B", "both"):
                self.assertAlmostEqual(original[name], row[name])
        for original, row in zip(self.res["utility"]["exponential"], shifted["utility"]["exponential"]):
            for name in ("A", "B"):
                self.assertAlmostEqual(row[name]["ce"], original[name]["ce"] + 100)
                self.assertAlmostEqual(row[name]["risk_premium"], original[name]["risk_premium"])
        self.assertEqual(shifted["emv"]["best"], "B")
        output = io.StringIO()
        with redirect_stdout(output):
            solver.print_report(shifted)
        self.assertIn("Ожидаемая стоимость лучшего решения: 163,00 тыс. руб.", output.getvalue())
        self.assertNotIn("P_A + 163", output.getvalue())

    def test_relative_budget_and_cv(self):
        shifted = solver.build_results(solver.Model(price_a=100))
        row = next(row for row in shifted["utility"]["budget"] if row["budget"] == 60)
        self.assertEqual(row["absolute_budget"], 160)
        self.assertEqual((row["A"], row["B"]), (.7, .3))
        self.assertAlmostEqual(shifted["risk"]["cv"]["A"], math.sqrt(1100) / 170)
        self.assertAlmostEqual(shifted["risk"]["cv_relative_to_price_a"]["A"], math.sqrt(1100) / 70)

    def test_larger_batch_fixed_discount(self):
        block = solver.emv_block(solver.Model(N=20_000))
        self.assertEqual(block["expected_costs"], {"A": 140, "B": 163})
        self.assertEqual(block["best"], "A")

    def test_evsi(self):
        rows = {row["n"]: row for row in self.res["evsi"]["table"]}
        self.assertAlmostEqual(rows[10]["A"], 0.0, places=9)
        self.assertAlmostEqual(rows[10]["B"], 1.239966104991, places=9)
        self.assertAlmostEqual(rows[50]["both"], 4.541636865648, places=9)
        self.assertAlmostEqual(rows[100]["B"], 5.222, places=2)
        self.assertAlmostEqual(rows[200]["both"], 9.806, places=2)
        # EVSI не может превышать EVPI
        evpi = self.res["evpi"]["both_independent"]
        for row in rows.values():
            self.assertLessEqual(row["both"], evpi + 1e-9)

    def test_utility_exponential_b_always_better(self):
        for row in self.res["utility"]["exponential"]:
            self.assertEqual(row["choice"], "B", f"rho={row['rho']}")
            self.assertGreater(row["A"]["ce"], row["A"]["risk_premium"])

    def test_exponential_family_global_order(self):
        """Точное сравнение всей экспоненциальной семьи через Q(1+z), z>0."""
        a = [round(c - 13) for c in self.model.costs("A")]
        b = [round(c - 13) for c in self.model.costs("B")]
        pa = [round(10 * p) for p in self.model.p_a]
        pb = [round(10 * p) for p in self.model.p_b]
        def coefficient(k):
            return (sum(p * math.comb(c, k) for p, c in zip(pa, a) if c >= k)
                    - sum(p * math.comb(c, k) for p, c in zip(pb, b) if c >= k))
        self.assertEqual(coefficient(0), 0)
        self.assertTrue(all(coefficient(k) > 0 for k in range(1, max(a) + 1)))

    def test_mean_sd_threshold(self):
        lam = self.res["utility"]["lambda_star"]
        self.assertAlmostEqual(lam, 7.0 / (math.sqrt(1500) - math.sqrt(1100)), places=9)

    def test_budget(self):
        rows = {row["budget"]: row for row in self.res["utility"]["budget"]}
        self.assertAlmostEqual(rows[60]["A"], 0.7, places=9)
        self.assertAlmostEqual(rows[60]["B"], 0.3, places=9)
        self.assertEqual(rows[60]["choice"], "A")
        self.assertAlmostEqual(rows[113]["B"], 1.0, places=9)
        self.assertEqual(rows[113]["choice"], "B")

    def test_split(self):
        split = self.res["split"]
        self.assertAlmostEqual(split["x_star_min_variance"], 15.0 / 26.0, places=9)
        self.assertAlmostEqual(split["sd_at_x_star"], 25.191573682788945, places=6)
        self.assertAlmostEqual(split["mean_at_x_star"], 63.0 + 7.0 * 15.0 / 26.0, places=6)
        self.assertAlmostEqual(split["comonotonic_covariance"], 1000)
        middle = next(row for row in split["curve"] if row["x"] == .5)
        self.assertAlmostEqual(middle["sd_comonotonic"], math.sqrt(1150))

    def test_zero_prior_probability(self):
        result = solver.build_results(solver.Model(p_a=(1, 0, 0), p_b=(0, 1, 0)))
        self.assertEqual(result["probability"]["bayes"]["posterior"], [1, 0, 0])
        self.assertAlmostEqual(result["evpi"]["both_independent"], 0)
        self.assertAlmostEqual(result["split"]["sd_at_x_star"], 0)
        for row in result["evsi"]["table"]:
            for name in ("A", "B", "both"):
                self.assertAlmostEqual(row[name], 0)

    def test_zero_repair_cost(self):
        result = solver.build_results(solver.Model(c=0))
        self.assertEqual(result["emv"]["best"], "B")
        self.assertIsNone(result["sensitivity"]["rate_b_star"])
        self.assertIsNone(result["risk"]["cv"]["A"])
        self.assertEqual(result["split"]["sd_at_x_star"], 0)
        json.dumps(result, allow_nan=False)
        with redirect_stdout(io.StringIO()):
            solver.print_report(result)

    def test_sampling_boundary_rates_and_impossible_observation(self):
        weights = solver._posterior_weights((0, 1), (.7, .3), 50, 0)
        self.assertEqual(weights, [1, 0])
        table = solver.evsi_block(solver.Model(rates=(0, 1), p_a=(.7, .3), p_b=(.4, .6)), ns=(10,))
        self.assertTrue(all(math.isfinite(table["table"][0][name]) for name in ("A", "B", "both")))
        with self.assertRaisesRegex(ValueError, "Наблюдение невозможно"):
            solver._posterior_weights((0, 1), (.7, .3), 50, 1)
        self.assertEqual(solver.ce_cost((1, 10000), (1, 0), 2), 1)

    def test_invalid_rates_and_nonfinite_parameters(self):
        for rates in ((-.01, .02, .03), (.01, .02, 1.03), (.01, math.nan, .03)):
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                solver.Model(rates=rates).validate()
        for kwargs in ({"p_a": (math.nan, .2, .1)}, {"c": math.inf}, {"N": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                solver.Model(**kwargs).validate()

    def test_simulation(self):
        sim = solver.simulate_block(self.model, runs=20_000, seed=7)
        self.assertLess(abs(sim["mean_A"] - 70.0), 1.5)
        self.assertLess(abs(sim["mean_B"] - 63.0), 1.5)
        self.assertLess(abs(sim["p_a_cheaper"] - 0.55), 0.02)

    def test_discount_20_divergence(self):
        """При скидке 20 EMV выбирает A, а пессимистические критерии — B."""
        m = solver.Model(discount=20)
        self.assertEqual(solver.best(m), "A")
        crit = solver.criteria_block(m)
        self.assertEqual(crit["emv"]["choice"], "A")
        self.assertEqual(crit["wald"]["choice"], "B")
        self.assertEqual(crit["savage"]["choice"], "B")
        self.assertEqual(crit["laplace"]["choice"], "B")

    def test_repair_08_divergence(self):
        """При c=0,8 EMV выбирает A, а Вальд — B."""
        m = solver.Model(c=0.8)
        self.assertEqual(solver.best(m), "A")
        self.assertEqual(solver.criteria_block(m)["wald"]["choice"], "B")

    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            solver.Model(p_a=(0.5, 0.2, 0.1)).validate()

    def test_negative_cost_rejected(self):
        with self.assertRaises(ValueError):
            solver.Model(c=-1).validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
