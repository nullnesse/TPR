#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Справочные расширения задачи «Луч».

Эти методы не входят в основной доклад, но сохраняются для истории,
тестов и полного ``results.json``: дерево, критерии неопределённости,
EVPI/EVSI, вероятностные модели и Байес, дробление заказа.
Модуль импортирует :mod:`solver_core`; обратного импорта нет.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import solver_core as core
from solver_core import Model


def _clamp_zero(x: float) -> float:
    """Убирает отрицательный численный шум у неотрицательных величин."""
    return 0.0 if abs(x) < 1e-12 else x


# ---------------------------------------------------------------------------
# Дерево и критерии неопределённости
# ---------------------------------------------------------------------------


def tree_block(model: Model) -> Dict:
    """Значения узлов дерева: круги — средние, корень — минимум."""
    chance = {s: model.expected_cost(s) for s in model.supplier_names}
    return {
        "chance_nodes": chance,
        "root": min(chance.values()),
        "chosen": min(chance, key=chance.get),
        "leaves": {s: model.costs(s) for s in model.supplier_names},
        "probabilities": {s: list(model.probs(s)) for s in model.supplier_names},
    }


def criteria_block(model: Model, alphas: Sequence[float] = (0, 0.25, 0.5, 0.75, 1.0)) -> Dict:
    """Классические критерии на матрице затрат.

    Матрица: строки — поставщики, столбцы — уровни брака (общие метки).
    Для затрат везде минимизация.
    """
    names = model.supplier_names
    matrix = {s: model.costs(s) for s in names}
    n_states = len(model.rates)

    wald = {s: max(matrix[s]) for s in names}
    minimin = {s: min(matrix[s]) for s in names}
    laplace = {s: sum(matrix[s]) / n_states for s in names}

    column_min = [min(matrix[s][j] for s in names) for j in range(n_states)]
    regret = {s: [matrix[s][j] - column_min[j] for j in range(n_states)] for s in names}
    savage = {s: max(regret[s]) for s in names}
    pairs = [(a, b) for a in matrix["A"] for b in matrix["B"]]
    joint_regret = {
        "A": [a - min(a, b) for a, b in pairs],
        "B": [b - min(a, b) for a, b in pairs],
    }
    joint_savage = {s: max(joint_regret[s]) for s in names}

    hurwicz = {}
    for a in alphas:
        hurwicz[str(a)] = {
            s: a * max(matrix[s]) + (1 - a) * min(matrix[s]) for s in names
        }

    def pick(scores: Dict[str, float]) -> str:
        m = min(scores.values())
        winners = [s for s, v in scores.items() if abs(v - m) < 1e-12]
        return winners[0] if len(winners) == 1 else "=".join(winners)

    return {
        "matrix": matrix,
        "wald": {"scores": wald, "choice": pick(wald)},
        "minimin": {"scores": minimin, "choice": pick(minimin)},
        "laplace": {"scores": laplace, "choice": pick(laplace)},
        "savage": {"scores": savage, "choice": pick(savage), "regret": regret,
                   "scope": "auxiliary_equal_rates"},
        "savage_joint": {"scores": joint_savage, "choice": pick(joint_savage),
                         "regret": joint_regret, "states": pairs,
                         "scope": "all_rate_pairs"},
        "hurwicz": {
            a: {"scores": sc, "choice": pick(sc)} for a, sc in hurwicz.items()
        },
        "emv": {
            "scores": {s: model.expected_cost(s) for s in names},
            "choice": core.best(model),
        },
    }


# ---------------------------------------------------------------------------
# Риск и полезность: совместимые таблицы
# ---------------------------------------------------------------------------


def risk_block(model: Model) -> Dict:
    """Прежняя схема risk: без новых полей Var(D)/SD(D)."""
    data = core.risk(model)
    keys = (
        "means", "variances", "sds", "cv", "cv_relative_to_price_a",
        "diff_distribution", "p_a_cheaper", "p_b_cheaper", "p_tie",
        "expected_diff", "worst", "best",
    )
    return {key: data[key] for key in keys}


def utility_block(model: Model, rhos: Sequence[float] = (2, 5, 10, 20, 50, 100, 1000)) -> Dict:
    """Прежняя таблица CE, mean--SD и бюджета для полного JSON."""
    names = model.supplier_names
    costs = {s: model.costs(s) for s in names}
    probs = {s: model.probs(s) for s in names}

    table = []
    for rho in rhos:
        row = {"rho": rho}
        util = core.utility(model, rho)
        for s in names:
            row[s] = {"ce": util[s]["ce"], "risk_premium": util[s]["risk_premium"]}
        # Прежняя совместимая таблица при равенстве оставляет A.
        row["choice"] = min(names, key=lambda s: row[s]["ce"])
        table.append(row)

    base_ms = core.mean_sd(model, 0)
    means = base_ms["means"]
    sds = base_ms["sds"]
    lam_star = base_ms["lambda_star"]

    mean_sd = []
    for lam in (0, 0.5, 1, 1.26, 1.5, 2):
        row = {"lambda": lam}
        row.update(core.mean_sd(model, lam)["scores"])
        row["choice"] = min(names, key=lambda s: row[s])
        mean_sd.append(row)

    budgets = []
    for b in (50, 60, 63, 100, 113, 150):
        row = {"budget": b, "absolute_budget": model.price_a + b}
        for s in names:
            row[s] = sum(p for c, p in zip(costs[s], probs[s])
                         if c <= row["absolute_budget"] + 1e-12)
        top = max(row[s] for s in names)
        winners = [s for s in names if abs(row[s] - top) < 1e-12]
        row["choice"] = winners[0] if len(winners) == 1 else "=".join(winners)
        budgets.append(row)

    return {
        "exponential": table,
        "mean_sd": mean_sd,
        "lambda_star": lam_star,
        "budget": budgets,
        "sds": sds,
        "means": means,
    }


# ---------------------------------------------------------------------------
# EVPI и EVSI
# ---------------------------------------------------------------------------


def evpi_block(model: Model) -> Dict:
    """Ценность информации при независимости и комонотонной связке партий."""
    names = model.supplier_names
    costs = {s: model.costs(s) for s in names}
    probs = {s: model.probs(s) for s in names}
    base = min(model.expected_cost(s) for s in names)

    # 1. Информация приходит после выбора — не влияет на решение.
    after_choice = 0.0

    # 2. Полная информация об одной партии (A или B) до выбора.
    def one_batch(s: str) -> float:
        other = "B" if s == "A" else "A"
        other_ev = model.expected_cost(other)
        e_perfect = sum(p * min(c, other_ev) for p, c in zip(probs[s], costs[s]))
        return base - e_perfect

    # 3. Полная информация об обеих партиях, качества независимы.
    e_both = 0.0
    for pa, ca in zip(probs["A"], costs["A"]):
        for pb, cb in zip(probs["B"], costs["B"]):
            e_both += pa * pb * min(ca, cb)
    evpi_both = base - e_both

    # 4. Максимальная положительная зависимость с исходными маргиналами.
    joint = core.joint_probabilities(model, "comonotonic")
    e_comonotonic = sum(joint[i][j] * min(ca, cb)
                        for i, ca in enumerate(costs["A"])
                        for j, cb in enumerate(costs["B"]))

    return {
        "base": base,
        "after_choice": after_choice,
        "one_batch_A": one_batch("A"),
        "one_batch_B": one_batch("B"),
        "both_independent": evpi_both,
        "both_comonotonic": _clamp_zero(base - e_comonotonic),
        "e_perfect_comonotonic": e_comonotonic,
        "comonotonic_joint": joint,
        "e_perfect_both": e_both,
    }


# ---------------------------------------------------------------------------
# Вероятностные модели и Байес
# ---------------------------------------------------------------------------


def _log_binom_pmf(n: int, k: int, p: float) -> float:
    """Логарифм биномиальной вероятности (устойчиво для больших n)."""
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def _sample_log_weights(
    rates: Sequence[float], priors: Sequence[float], n: int, k: int,
    population_size: int | None = None,
) -> List[float]:
    """Логарифмы совместных вероятностей состояния и результата выборки.

    Без ``population_size`` используется биномиальное правдоподобие;
    с ``population_size`` — точная выборка без возвращения из конечной партии.
    """
    if not isinstance(n, int) or not isinstance(k, int) or not 0 <= k <= n:
        raise ValueError("Выборка должна удовлетворять 0 <= k <= n для целых n и k")
    if population_size is not None and n > population_size:
        raise ValueError("Выборка без возвращения не может быть больше партии")
    logs = []
    for r, prior in zip(rates, priors):
        if prior == 0:
            logs.append(-math.inf)
            continue
        if population_size is None:
            likelihood = _log_binom_pmf(n, k, r)
        else:
            defects = population_size * r
            if not math.isclose(defects, round(defects), abs_tol=1e-8, rel_tol=0):
                raise ValueError("Для конечной партии N*r должно быть целым числом")
            defects = round(defects)
            if k > defects or n - k > population_size - defects:
                likelihood = -math.inf
            else:
                likelihood = (math.log(math.comb(defects, k))
                              + math.log(math.comb(population_size - defects, n - k))
                              - math.log(math.comb(population_size, n)))
        logs.append(likelihood + math.log(prior))
    return logs


def _posterior_weights(
    rates: Sequence[float], priors: Sequence[float], n: int, k: int,
    population_size: int | None = None,
) -> List[float]:
    """Нормированные апостериорные вероятности уровней брака."""
    logs = _sample_log_weights(rates, priors, n, k, population_size)
    m = max(logs)
    if m == -math.inf:
        raise ValueError("Наблюдение невозможно при заданных состояниях и априорных вероятностях")
    weights = [math.exp(x - m) for x in logs]
    z = sum(weights)
    return [w / z for w in weights]


def _posterior_mean(rates: Sequence[float], priors: Sequence[float], n: int, k: int) -> float:
    weights = _posterior_weights(rates, priors, n, k)
    return sum(r * w for r, w in zip(rates, weights))


def probability_block(model: Model, n_bayes: int = 50, k_bayes: int = 0) -> Dict:
    """Вводные вероятностные характеристики для папки «Теория вероятности»."""
    names = model.supplier_names
    batch_var = {s: core.variance(model.defects, model.probs(s)) for s in names}
    batch: Dict[str, Dict[str, float]] = {}
    binomial: Dict[str, Dict[str, object]] = {}

    for s in names:
        within = sum(
            p * model.N * r * (1 - r)
            for p, r in zip(model.probs(s), model.rates)
        )
        total_var = within + batch_var[s]
        batch[s] = {
            "defects_mean": model.mean_defects[s],
            "defects_var": batch_var[s],
            "defects_sd": math.sqrt(batch_var[s]),
            "cost_var": model.c ** 2 * batch_var[s],
            "cost_sd": model.c * math.sqrt(batch_var[s]),
        }
        binomial[s] = {
            "within_var": within,
            "between_var": batch_var[s],
            "variance": total_var,
            "sd": math.sqrt(total_var),
            "cost_var": model.c ** 2 * total_var,
            "cost_sd": model.c * math.sqrt(total_var),
            "pmf_at_mean": {
                str(r): math.exp(_log_binom_pmf(model.N, int(round(model.N * r)), r))
                for r in model.rates
            },
        }

    posterior = _posterior_weights(model.rates, model.p_a, n_bayes, k_bayes)
    post_mean = sum(r * w for r, w in zip(model.rates, posterior))
    cost_a = model.price_a + model.N * post_mean * model.c
    cost_b = model.expected_cost("B")
    finite = None
    if n_bayes <= model.N and all(math.isclose(x, round(x), abs_tol=1e-8, rel_tol=0)
                                  for x in model.defects):
        finite_weights = _posterior_weights(model.rates, model.p_a, n_bayes, k_bayes, model.N)
        finite_mean = sum(r * p for r, p in zip(model.rates, finite_weights))
        finite_cost = model.price_a + model.N * finite_mean * model.c
        finite = {"n": n_bayes, "k": k_bayes, "posterior": finite_weights,
                  "posterior_mean": finite_mean, "cost_a": finite_cost, "cost_b": cost_b,
                  "choice": "A" if finite_cost < cost_b else "B"}

    return {
        "batch": batch,
        "binomial": binomial,
        "bayes_finite": finite,
        "bayes": {
            "n": n_bayes,
            "k": k_bayes,
            "prior": list(model.p_a),
            "posterior": posterior,
            "posterior_mean": post_mean,
            "cost_a": cost_a,
            "cost_b": cost_b,
            "choice": "A" if cost_a < cost_b else "B",
        },
    }


def evsi_block(model: Model, ns: Sequence[int] = (10, 20, 50, 100, 200)) -> Dict:
    """EVSI для выборки у A, у B и у обоих (наивный перебор)."""
    rates = model.rates
    base = min(model.expected_cost(s) for s in model.supplier_names)

    def cost_a_from_mean(m: float) -> float:
        return model.N * model.c * m + model.price_a

    def cost_b_from_mean(m: float) -> float:
        return model.N * model.c * m + model.price_a - model.discount

    def outcomes(priors: Sequence[float], n: int) -> List[Tuple[float, float]]:
        """Вероятность результата выборки и условное среднее доли брака."""
        result = []
        for k in range(n + 1):
            logs = _sample_log_weights(rates, priors, n, k)
            m = max(logs)
            if m == -math.inf:
                continue
            weights = [math.exp(x - m) for x in logs]
            z = sum(weights)
            result.append((math.exp(m) * z, sum(r * w for r, w in zip(rates, weights)) / z))
        return result

    table = []
    for n in ns:
        if not isinstance(n, int) or n < 0:
            raise ValueError("Объём выборки должен быть неотрицательным целым числом")
        a_outcomes = [(p, cost_a_from_mean(mu)) for p, mu in outcomes(model.p_a, n)]
        b_outcomes = [(p, cost_b_from_mean(mu)) for p, mu in outcomes(model.p_b, n)]
        after_a = sum(p * min(c, model.expected_cost("B")) for p, c in a_outcomes)
        after_b = sum(p * min(c, model.expected_cost("A")) for p, c in b_outcomes)
        after_both = sum(pa * pb * min(ca, cb)
                         for pa, ca in a_outcomes for pb, cb in b_outcomes)
        table.append(
            {
                "n": n,
                "A": _clamp_zero(base - after_a),
                "B": _clamp_zero(base - after_b),
                "both": _clamp_zero(base - after_both),
            }
        )
    return {"base": base, "table": table}


# ---------------------------------------------------------------------------
# Дробление и Монте-Карло
# ---------------------------------------------------------------------------


def split_block(model: Model, steps: int = 21) -> Dict:
    """Дополнительная модель дробления с пропорциональными ценами и скидкой.

    Распределение доли брака не зависит от размера подзаказа; исходное условие
    таких возможностей не задаёт. Сравниваются независимость и комонотонная связка.
    """
    if steps < 2:
        raise ValueError("Для кривой дробления нужны хотя бы две точки")
    costs = {s: model.costs(s) for s in model.supplier_names}
    probs = {s: model.probs(s) for s in model.supplier_names}
    means = {s: model.expected_cost(s) for s in model.supplier_names}
    vars_ = {s: core.variance(costs[s], probs[s]) for s in model.supplier_names}

    def mean(x: float) -> float:
        return x * means["A"] + (1 - x) * means["B"]

    def sd_indep(x: float) -> float:
        return math.sqrt(x * x * vars_["A"] + (1 - x) ** 2 * vars_["B"])

    joint = core.joint_probabilities(model, "comonotonic")
    covariance = sum(joint[i][j] * (ca - means["A"]) * (cb - means["B"])
                     for i, ca in enumerate(costs["A"]) for j, cb in enumerate(costs["B"]))

    def sd_comonotonic(x: float) -> float:
        return math.sqrt(max(0.0, x * x * vars_["A"] + (1 - x) ** 2 * vars_["B"]
                             + 2 * x * (1 - x) * covariance))

    total_variance = vars_["A"] + vars_["B"]
    x_star = vars_["B"] / total_variance if total_variance > 0 else 0.0
    xs = [i / (steps - 1) for i in range(steps)]
    return {
        "x_star_min_variance": x_star,
        "mean_at_x_star": mean(x_star),
        "sd_at_x_star": sd_indep(x_star),
        "sd_all_A": sd_indep(1.0),
        "sd_all_B": sd_indep(0.0),
        "comonotonic_covariance": covariance,
        "assumptions": "proportional_prices_and_discount; size_invariant_rate_distributions",
        "curve": [
            {
                "x": x,
                "mean": mean(x),
                "sd_independent": sd_indep(x),
                "sd_comonotonic": sd_comonotonic(x),
            }
            for x in xs
        ],
    }


def simulate_block(model: Model, runs: int = 100_000, seed: int = 42) -> Dict:
    """Совместимое имя для ``solver_core.simulation``."""
    return core.simulation(model, runs, seed)


def sensitivity_block(model: Model) -> Dict:
    """Совместимое имя для ``solver_core.sensitivity``."""
    return core.sensitivity(model)
