#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты общей постановки: m поставщиков, разные наборы уровней брака.

Проверяются эталоны из плана 11 §5.1, инварианты EOL, классификация порогов,
нормировка при большой общей цене, валидация и совместимость с legacy API.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import solver_core as core

ROOT = Path(__file__).resolve().parent


def close(actual: float, expected: float, tol: float = 1e-9) -> bool:
    return abs(actual - expected) <= tol * max(1.0, abs(expected))


class BaseCaseTest(unittest.TestCase):
    """Общая модель повторяет исходную задачу и её числа."""

    def setUp(self) -> None:
        self.model = core.GeneralModel()
        self.data = core.build_general_results(self.model, include_simulation=False)

    def test_base_emv_matches_legacy(self) -> None:
        emv = self.data["emv"]
        self.assertEqual(emv["choice"], "B")
        self.assertEqual(emv["winners"], ["B"])
        self.assertAlmostEqual(emv["expected_costs_normalized"]["A"], 70.0, places=9)
        self.assertAlmostEqual(emv["expected_costs_normalized"]["B"], 63.0, places=9)
        self.assertAlmostEqual(emv["saving_vs_reference"]["B"], 7.0, places=9)
        self.assertAlmostEqual(emv["expected_defects"]["A"], 140.0, places=9)
        self.assertAlmostEqual(emv["expected_defects"]["B"], 200.0, places=9)

    def test_base_eol_matches_legacy(self) -> None:
        eol = self.data["eol"]
        self.assertEqual(eol["state_count"], 9)
        self.assertAlmostEqual(eol["expected_minimum"], 45.35, places=9)
        self.assertAlmostEqual(eol["expected_regret"]["A"], 24.65, places=9)
        self.assertAlmostEqual(eol["expected_regret"]["B"], 17.65, places=9)
        self.assertEqual(eol["choice"], "B")

    def test_base_variance_and_diff_probability(self) -> None:
        risk = self.data["risk"]
        self.assertAlmostEqual(risk["variances"]["A"], 1100.0, places=9)
        self.assertAlmostEqual(risk["variances"]["B"], 1500.0, places=9)
        self.assertAlmostEqual(risk["diff"]["p_first_cheaper"], 0.55, places=9)
        self.assertAlmostEqual(risk["diff"]["p_tie"], 0.0, places=9)

    def test_base_thresholds(self) -> None:
        thresholds = self.data["sensitivity"]["thresholds"]
        self.assertEqual(thresholds["discount"]["kind"], "single")
        self.assertTrue(thresholds["discount"]["feasible"])
        self.assertAlmostEqual(thresholds["discount"]["root"], 30.0, places=9)
        self.assertAlmostEqual(thresholds["repair_cost"]["root"], 37 / 60, places=9)
        self.assertAlmostEqual(thresholds["mean_rate_b"]["root"], 0.0214, places=9)
        self.assertFalse(thresholds["batch_size"]["feasible"])
        self.assertAlmostEqual(thresholds["batch_size"]["root"], 37 / 0.003, places=6)

    def test_base_mean_sd_and_utility(self) -> None:
        sd = core.general_mean_sd(self.model, 2.0)
        self.assertAlmostEqual(sd["scores"]["A"], 136.332495807108, places=9)
        self.assertAlmostEqual(sd["scores"]["B"], 140.45966692414834, places=9)
        self.assertEqual(sd["choice"], "A")
        utility = core.general_utility(self.model, 100.0)
        self.assertAlmostEqual(utility["A"]["ce"], 76.35731004430394, places=9)
        self.assertAlmostEqual(utility["B"]["ce"], 70.37852435545759, places=9)
        self.assertEqual(utility["choice"], "B")

    def test_base_risk_needs_pair_only_above_two(self) -> None:
        with self.assertRaises(ValueError):
            core.general_risk(core.GeneralModel(
                suppliers=core.GeneralModel().suppliers + (core.Supplier("C", (0.5,), (1.0,), 9.0),)
            ))


class SyntheticExamplesTest(unittest.TestCase):
    """Разные сетки: эталоны выведены точным перебором."""

    def test_unequal_2x3(self) -> None:
        data = core.build_general_results(
            core.GeneralModel(suppliers=(
                core.Supplier("A", (0.01, 0.03), (0.75, 0.25), 0.0),
                core.Supplier("B", (0.02, 0.03, 0.11), (0.5, 0.25, 0.25), -37.0),
            )),
            include_simulation=False,
        )
        self.assertEqual(data["state_count"], 6)
        self.assertAlmostEqual(data["emv"]["expected_costs_normalized"]["A"], 75.0, places=9)
        self.assertAlmostEqual(data["emv"]["expected_costs_normalized"]["B"], 188.0, places=9)
        self.assertAlmostEqual(data["eol"]["expected_minimum"], float(Fraction(989, 16)), places=9)
        self.assertAlmostEqual(data["eol"]["expected_regret"]["A"], float(Fraction(211, 16)), places=9)
        self.assertAlmostEqual(data["eol"]["expected_regret"]["B"], float(Fraction(2019, 16)), places=9)
        self.assertEqual(data["emv"]["choice"], "A")

    def test_teacher_grids(self) -> None:
        data = core.build_general_results(
            core.GeneralModel(suppliers=(
                core.Supplier("A", (0.01, 0.02, 0.03, 0.05, 0.07), (0.2,) * 5, 0.0),
                core.Supplier("B", (0.02, 0.03, 0.11), (1 / 3, 1 / 3, 1 / 3), -37.0),
            )),
            include_simulation=False,
        )
        self.assertEqual(data["state_count"], 15)
        self.assertAlmostEqual(data["emv"]["expected_costs_normalized"]["A"], 180.0, places=9)
        self.assertAlmostEqual(data["emv"]["expected_costs_normalized"]["B"], float(Fraction(689, 3)), places=9)
        self.assertAlmostEqual(data["eol"]["expected_minimum"], float(Fraction(1691, 15)), places=9)
        self.assertAlmostEqual(data["eol"]["expected_regret"]["A"], float(Fraction(1009, 15)), places=9)
        self.assertAlmostEqual(data["eol"]["expected_regret"]["B"], float(Fraction(1754, 15)), places=9)
        self.assertEqual(data["emv"]["choice"], "A")

    def test_three_suppliers_global_winner(self) -> None:
        model = core.GeneralModel(suppliers=(
            core.Supplier("A", (0.01, 0.03), (0.75, 0.25), 0.0),
            core.Supplier("B", (0.02, 0.03, 0.11), (0.5, 0.25, 0.25), -37.0),
            core.Supplier("C", (0.01,), (1.0,), 10.0),
        ))
        data = core.build_general_results(model, include_simulation=False)
        self.assertEqual(data["state_count"], 6)
        self.assertEqual(data["emv"]["winners"], ["C"])
        self.assertAlmostEqual(data["emv"]["expected_costs_normalized"]["C"], 60.0, places=9)
        self.assertAlmostEqual(data["eol"]["expected_regret"]["C"], 7.5, places=9)
        self.assertEqual(data["eol"]["choice"], "C")
        # Попарное сравнение A/B по-прежнему выбирает A, как в примере 2×3.
        pair = core.general_sensitivity(model, pair=("A", "B"))
        self.assertEqual(pair["pair"]["a"], "A")

    def test_single_supplier_has_zero_eol(self) -> None:
        model = core.GeneralModel(suppliers=(core.Supplier("Only", (0.01, 0.02), (0.4, 0.6), 0.0),))
        data = core.build_general_results(model, include_simulation=False)
        self.assertEqual(data["emv"]["winners"], ["Only"])
        self.assertEqual(data["eol"]["expected_regret"]["Only"], 0.0)
        self.assertEqual(data["state_count"], 2)
        self.assertIsNone(data["risk"]["diff"])
        self.assertIsNone(data["sensitivity"]["pair"])


class InvariantsTest(unittest.TestCase):
    """Инварианты EOL, вероятностей и устойчивости к общей цене."""

    def setUp(self) -> None:
        self.model = core.GeneralModel(suppliers=(
            core.Supplier("A", (0.01, 0.03), (0.75, 0.25), 0.0),
            core.Supplier("B", (0.02, 0.03, 0.11), (0.5, 0.25, 0.25), -37.0),
        ))

    def test_states_sum_and_marginals(self) -> None:
        data = core.general_eol(self.model, include_states=True)
        total = sum(state["probability"] for state in data["states"])
        self.assertAlmostEqual(total, 1.0, places=12)
        marginal_a = {}
        for state in data["states"]:
            marginal_a[state["rates"][0]] = marginal_a.get(state["rates"][0], 0.0) + state["probability"]
        self.assertAlmostEqual(marginal_a[0.01], 0.75, places=12)
        self.assertAlmostEqual(marginal_a[0.03], 0.25, places=12)

    def test_eol_identity_and_nonnegative(self) -> None:
        data = core.general_eol(self.model)
        emv = core.general_emv(self.model)
        expected_min = data["expected_minimum"] - self.model.reference_price
        for name in ("A", "B"):
            self.assertAlmostEqual(
                data["expected_regret"][name],
                emv["expected_costs_normalized"][name] - expected_min,
                places=9,
            )
            self.assertGreaterEqual(data["expected_regret"][name], -1e-12)
        self.assertEqual(set(data["winners"]), set(emv["winners"]))

    def test_direct_regret_sum_matches(self) -> None:
        data = core.general_eol(self.model, include_states=True)
        for name in ("A", "B"):
            direct = sum(state["probability"] * state["regrets"][name] for state in data["states"])
            self.assertAlmostEqual(direct, data["expected_regret"][name], places=12)

    def test_common_price_shift_changes_nothing(self) -> None:
        shifted = core.GeneralModel(
            N=self.model.N,
            c=self.model.c,
            reference_price=1e13,
            suppliers=self.model.suppliers,
        )
        base_emv = core.general_emv(self.model)
        shifted_emv = core.general_emv(shifted)
        self.assertEqual(shifted_emv["winners"], base_emv["winners"])
        self.assertAlmostEqual(shifted_emv["saving_vs_reference"]["B"], -113.0, places=6)
        base_eol = core.general_eol(self.model)
        shifted_eol = core.general_eol(shifted)
        for name in ("A", "B"):
            self.assertAlmostEqual(
                shifted_eol["expected_regret"][name], base_eol["expected_regret"][name], places=6
            )
        base_risk = core.general_risk(self.model)
        shifted_risk = core.general_risk(shifted)
        for name in ("A", "B"):
            self.assertAlmostEqual(
                shifted_risk["variances"][name], base_risk["variances"][name], places=6
            )

    def test_expensive_third_alternative_does_not_create_tie(self) -> None:
        model = core.GeneralModel(suppliers=core.GeneralModel().suppliers + (
            core.Supplier("C", (0.5,), (1.0,), 1e9),
        ))
        emv = core.general_emv(model)
        self.assertEqual(emv["choice"], "B")
        self.assertEqual(emv["winners"], ["B"])

    def test_zero_probability_keeps_state_count(self) -> None:
        model = core.GeneralModel(suppliers=(
            core.Supplier("A", (0.01, 0.02), (1.0, 0.0), 0.0),
            core.Supplier("B", (0.01, 0.02, 0.03), (0.3, 0.4, 0.3), -37.0),
        ))
        data = core.general_eol(model)
        self.assertEqual(data["state_count"], 6)
        self.assertEqual(data["positive_state_count"], 3)
        self.assertAlmostEqual(
            sum([data["expected_regret"][name] for name in ("A", "B")]),
            data["expected_regret"]["A"] + data["expected_regret"]["B"],
        )

    def test_state_limit_rejected_before_walk(self) -> None:
        with self.assertRaises(ValueError):
            core.general_eol(core.GeneralModel(), max_states=5)


class ValidationTest(unittest.TestCase):
    """Строгая граница нового формата не ослабляет legacy."""

    def _model(self, supplier: core.Supplier) -> core.GeneralModel:
        return core.GeneralModel(suppliers=(supplier,))

    def test_rejects_bad_inputs(self) -> None:
        cases = {
            "empty": core.Supplier("A", (), ()),
            "lengths": core.Supplier("A", (0.01, 0.02), (1.0,)),
            "duplicates": core.Supplier("A", (0.01, 0.01), (0.5, 0.5)),
            "descending": core.Supplier("A", (0.02, 0.01), (0.5, 0.5)),
            "prob sum": core.Supplier("A", (0.01,), (0.9,)),
            "negative prob": core.Supplier("A", (0.01, 0.02), (-0.1, 1.1)),
            "rate range": core.Supplier("A", (0.01, 1.5), (0.5, 0.5)),
            "reserved id": core.Supplier("tie", (0.01,), (1.0,)),
            "nan": core.Supplier("A", (float("nan"),), (1.0,)),
        }
        for name, supplier in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self._model(supplier).validate()

    def test_rejects_non_integer_defects_and_negative_price(self) -> None:
        with self.assertRaises(ValueError):
            self._model(core.Supplier("A", (0.01005,), (1.0,))).validate()
        with self.assertRaises(ValueError):
            core.GeneralModel(price_known=True, suppliers=(
                core.Supplier("A", (0.01,), (1.0,), -50.0),
            )).validate()
        with self.assertRaises(ValueError):
            core.GeneralModel(N=True).validate()
        with self.assertRaises(ValueError):
            core.GeneralModel(reference_price=-1.0).validate()

    def test_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            core.GeneralModel(suppliers=(
                core.Supplier("A", (0.01,), (1.0,)),
                core.Supplier("A", (0.02,), (1.0,), -1.0),
            )).validate()

    def test_legacy_model_keeps_unsorted_rates(self) -> None:
        legacy = core.Model(rates=(0.03, 0.01, 0.02))
        legacy.validate()
        self.assertEqual(len(core.joint_states(legacy)), 9)


class ThresholdTest(unittest.TestCase):
    """Классификация корня: single / none / all и границы области."""

    def test_identical_suppliers_give_all(self) -> None:
        supplier = core.Supplier("A", (0.01,), (1.0,), 0.0)
        model = core.GeneralModel(suppliers=(supplier, core.Supplier("B", (0.01,), (1.0,), 0.0)))
        thresholds = core.general_sensitivity(model)["thresholds"]
        self.assertEqual(thresholds["repair_cost"]["kind"], "all")
        self.assertEqual(thresholds["discount"]["kind"], "single")
        self.assertEqual(thresholds["discount"]["root"], 0.0)

    def test_equal_quality_different_price_has_no_repair_root(self) -> None:
        model = core.GeneralModel(suppliers=(
            core.Supplier("A", (0.01,), (1.0,), 0.0),
            core.Supplier("B", (0.01,), (1.0,), -37.0),
        ))
        thresholds = core.general_sensitivity(model)["thresholds"]
        self.assertEqual(thresholds["repair_cost"]["kind"], "none")
        self.assertEqual(thresholds["discount"]["root"], 0.0)

    def test_root_on_region_boundary(self) -> None:
        model = core.GeneralModel(suppliers=(
            core.Supplier("A", (0.01,), (1.0,), 0.0),
            core.Supplier("B", (0.02,), (1.0,), 0.0),
        ))
        thresholds = core.general_sensitivity(model)["thresholds"]
        self.assertEqual(thresholds["repair_cost"]["kind"], "single")
        self.assertEqual(thresholds["repair_cost"]["root"], 0.0)
        self.assertTrue(thresholds["repair_cost"]["feasible"])

    def test_rate_root_outside_achievable_range(self) -> None:
        model = core.GeneralModel(suppliers=(
            core.Supplier("A", (0.05,), (1.0,), 0.0),
            core.Supplier("B", (0.01, 0.02), (0.5, 0.5), -37.0),
        ))
        thresholds = core.general_sensitivity(model)["thresholds"]
        # Порог по средней доле B выше максимального уровня 2 %.
        self.assertGreater(thresholds["mean_rate_b"]["root"], 0.02)
        self.assertFalse(thresholds["mean_rate_b"]["feasible"])

    def test_known_price_limits_discount_region(self) -> None:
        model = core.GeneralModel(
            reference_price=100.0,
            price_known=True,
            suppliers=(
                core.Supplier("A", (0.01,), (1.0,), 0.0),
                core.Supplier("B", (0.02,), (1.0,), -37.0),
            ),
        )
        thresholds = core.general_sensitivity(model)["thresholds"]
        self.assertEqual(thresholds["discount"]["region"]["upper"], 100.0)
        self.assertTrue(thresholds["discount"]["feasible"])


class TieAndRenameTest(unittest.TestCase):
    """Ничьи и переименование не меняют чисел."""

    def test_full_tie_of_two(self) -> None:
        supplier = core.Supplier("A", (0.01, 0.02), (0.5, 0.5), 0.0)
        model = core.GeneralModel(suppliers=(supplier, core.Supplier("B", (0.01, 0.02), (0.5, 0.5), 0.0)))
        self.assertEqual(core.general_emv(model)["choice"], "tie")
        self.assertEqual(core.general_eol(model)["choice"], "tie")

    def test_two_of_three_tie(self) -> None:
        model = core.GeneralModel(suppliers=(
            core.Supplier("A", (0.01,), (1.0,), 0.0),
            core.Supplier("B", (0.01,), (1.0,), 0.0),
            core.Supplier("C", (0.03,), (1.0,), 0.0),
        ))
        emv = core.general_emv(model)
        self.assertEqual(emv["winners"], ["A", "B"])
        self.assertEqual(emv["choice"], "tie")

    def test_rename_changes_keys_not_numbers(self) -> None:
        base = core.general_emv(core.GeneralModel())
        renamed = core.general_emv(core.GeneralModel(suppliers=(
            core.Supplier("X", (0.01, 0.02, 0.03), (0.7, 0.2, 0.1), 0.0),
            core.Supplier("Y", (0.01, 0.02, 0.03), (0.3, 0.4, 0.3), -37.0),
        )))
        self.assertEqual(renamed["winners"], ["Y"])
        self.assertAlmostEqual(
            renamed["expected_costs_normalized"]["Y"],
            base["expected_costs_normalized"]["B"],
            places=12,
        )


class SimulationAndExportTest(unittest.TestCase):
    """Симуляция с одним seed и детерминированный экспорт."""

    def test_simulation_reproducible(self) -> None:
        model = core.GeneralModel()
        first = core.general_simulation(model, runs=5000, seed=7)
        second = core.general_simulation(model, runs=5000, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(first["p_best"].values()), 1.0, places=12)
        self.assertAlmostEqual(first["means"]["B"], 63.0, delta=1.5)

    def test_results_schema_and_serializable(self) -> None:
        data = core.build_general_results(core.GeneralModel(), include_simulation=True, runs=1000)
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual(data["scope"], "general")
        self.assertEqual(data["level_semantics"], "exact_count")
        text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertEqual(
            text,
            json.dumps(core.build_general_results(core.GeneralModel(), include_simulation=True, runs=1000),
                       ensure_ascii=False, indent=2, allow_nan=False),
        )

    def test_examples_have_four_cases(self) -> None:
        data = core.build_general_examples()
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual(set(data["cases"]), {"base", "unequal_2x3", "three_2x3x1", "teacher_grids"})
        self.assertEqual(data["cases"]["base"]["eol"]["state_count"], 9)
        self.assertEqual(data["cases"]["teacher_grids"]["eol"]["state_count"], 15)

    def test_export_guard_and_repeatability(self) -> None:
        import solver

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "generalized" / "results.json"
            solver.write_general_export(str(target))
            first = target.read_bytes()
            solver.write_general_export(str(target))
            self.assertEqual(first, target.read_bytes())
            self.assertTrue(first.endswith(b"\n"))
        with self.assertRaises(ValueError):
            solver.write_general_export(str(ROOT / "results.json"))

    def test_facade_reexports_general_api(self) -> None:
        import solver

        for name in ("GeneralModel", "Supplier", "general_emv", "general_eol",
                     "general_risk", "general_sensitivity", "build_general_results"):
            self.assertTrue(hasattr(solver, name), name)


if __name__ == "__main__":
    unittest.main()
