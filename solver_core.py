#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Вычислительное ядро задачи «Луч».

Здесь собраны только те методы, которые нужны для доклада:
EMV, EOL, риск/mean--SD, ожидаемая полезность/CE, чувствительность
и Монте-Карло. Расширения (дерево, критерии неопределённости, EVPI/EVSI,
Байес, дробление) живут в ``solver_extras.py`` и не импортируются ядром.

Все денежные величины — в тыс. руб. Функции не печатают и не пишут файлы.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

Pair = Dict[str, float]
State = Dict[str, float]


def _is_finite_number(value: object) -> bool:
    """Число, которое можно использовать в расчётах (bool не число)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_distribution(values: Sequence[float], probs: Sequence[float]) -> None:
    """Проверить пару «исходы — вероятности» перед усреднением."""
    if len(values) != len(probs) or not values:
        raise ValueError("values и probs должны быть непустыми и одинаковой длины")
    if any(not _is_finite_number(v) for v in values):
        raise ValueError("values должны быть конечными числами")
    if any(not _is_finite_number(p) or not 0 <= p <= 1 for p in probs):
        raise ValueError("probs должны быть конечными числами в [0, 1]")
    if abs(sum(probs) - 1.0) > 1e-9:
        raise ValueError("сумма вероятностей должна быть 1")


def expected(values: Sequence[float], probs: Sequence[float]) -> float:
    """Математическое ожидание: ``sum(p_i * v_i)``."""
    _validate_distribution(values, probs)
    return sum(p * v for p, v in zip(probs, values))


def variance(values: Sequence[float], probs: Sequence[float]) -> float:
    """Дисперсия заданного распределения (без поправки на выборку)."""
    mean = expected(values, probs)
    return sum(p * (v - mean) ** 2 for p, v in zip(probs, values))


def stdev(values: Sequence[float], probs: Sequence[float]) -> float:
    """Стандартное отклонение — корень дисперсии."""
    return math.sqrt(variance(values, probs))


def ce_cost(values: Sequence[float], probs: Sequence[float], rho: float) -> float:
    """Эквивалент уверенности для затрат при экспоненциальной полезности.

    Для убывающей вогнутой полезности ``U(C) = -exp(C / rho)``
    максимизация ожидаемой полезности эквивалентна минимизации
    ``CE = rho * ln E[exp(C / rho)]``. Сдвиг на максимум нужен,
    чтобы не переполнить экспоненту.
    """
    if not _is_finite_number(rho) or rho <= 0:
        raise ValueError("rho должно быть конечным положительным числом")
    _validate_distribution(values, probs)
    positive = [(v, p) for v, p in zip(values, probs) if p > 0]
    if not positive:
        raise ValueError("нужна хотя бы одна положительная вероятность")
    m = max(v for v, _ in positive)
    s = sum(p * math.exp((v - m) / rho) for v, p in positive)
    return m + rho * math.log(s)


@dataclass
class Model:
    """Параметры задачи. Все деньги — в тыс. руб."""

    N: int = 10_000
    c: float = 0.5
    discount: float = 37.0
    price_a: float = 0.0
    rates: Tuple[float, ...] = (0.01, 0.02, 0.03)
    p_a: Tuple[float, ...] = (0.7, 0.2, 0.1)
    p_b: Tuple[float, ...] = (0.3, 0.4, 0.3)
    supplier_names: Tuple[str, str] = ("A", "B")

    def validate(self) -> None:
        """Проверить размеры, конечность чисел и допустимость вероятностей."""
        if not isinstance(self.N, int) or isinstance(self.N, bool) or self.N <= 0:
            raise ValueError("N должно быть положительным целым числом")
        if any(not _is_finite_number(x) or x < 0 for x in (self.c, self.discount, self.price_a)):
            raise ValueError("c, discount и price_a должны быть конечными и неотрицательными")
        if not self.rates or any(not _is_finite_number(r) or not 0 <= r <= 1 for r in self.rates):
            raise ValueError("rates: доли брака должны лежать в интервале [0, 1]")
        if self.supplier_names != ("A", "B"):
            raise ValueError("Поддерживаются поставщики A и B")
        if len(self.rates) != len(self.p_a) or len(self.rates) != len(self.p_b):
            raise ValueError("Размеры rates, p_a и p_b должны совпадать")
        for name, p in (("p_a", self.p_a), ("p_b", self.p_b)):
            if any(not _is_finite_number(x) or not 0 <= x <= 1 for x in p):
                raise ValueError(f"{name}: вероятности должны быть конечными числами в [0, 1]")
            if abs(sum(p) - 1.0) > 1e-9:
                raise ValueError(f"{name}: сумма вероятностей должна быть 1, получено {sum(p)}")

    @property
    def defects(self) -> List[float]:
        """Число бракованных переключателей для каждого уровня брака."""
        return [self.N * r for r in self.rates]

    @property
    def repair_costs(self) -> List[float]:
        """Стоимость ремонта партии для каждого уровня брака."""
        return [self.c * x for x in self.defects]

    def costs(self, supplier: str) -> List[float]:
        """Затраты (ремонт + цена) для каждого уровня брака.

        При ``price_a=0`` получаются затраты относительно цены A;
        при ненулевом ``price_a`` цена включена в обе альтернативы.
        """
        if supplier not in self.supplier_names:
            raise ValueError(f"Неизвестный поставщик: {supplier}")
        base = self.price_a if supplier == "A" else self.price_a - self.discount
        return [base + r for r in self.repair_costs]

    def probs(self, supplier: str) -> Tuple[float, ...]:
        if supplier == "A":
            return self.p_a
        if supplier == "B":
            return self.p_b
        raise ValueError(f"Неизвестный поставщик: {supplier}")

    def expected_repair(self, supplier: str) -> float:
        return expected(self.repair_costs, self.probs(supplier))

    def expected_cost(self, supplier: str) -> float:
        return expected(self.costs(supplier), self.probs(supplier))

    @property
    def expected_rate(self) -> Dict[str, float]:
        return {
            "A": expected(self.rates, self.p_a),
            "B": expected(self.rates, self.p_b),
        }

    @property
    def mean_defects(self) -> Dict[str, float]:
        return {
            "A": expected(self.defects, self.p_a),
            "B": expected(self.defects, self.p_b),
        }


def best(model: Model) -> str:
    """Поставщик с минимальными ожидаемыми затратами (прежнее правило)."""
    return min(model.supplier_names, key=model.expected_cost)


def choose(scores: Pair) -> str:
    """Выбор минимума с явной ничьей: ``A``/``B``/``tie``."""
    m = min(scores.values())
    tolerance = 1e-9 + 1e-12 * max(abs(v) for v in scores.values())
    winners = [s for s, v in scores.items() if abs(v - m) <= tolerance]
    return winners[0] if len(winners) == 1 else "tie"


def joint_probabilities(model: Model, coupling: str = "independent") -> List[List[float]]:
    """Совместные вероятности уровней A и B с сохранением обоих маргиналов.

    Комонотонная связка сопоставляет квантили через один общий U ~ Uniform(0, 1).
    Она даёт максимальную положительную ковариацию, но не обязательно корреляцию 1.
    """
    if coupling == "independent":
        return [[pa * pb for pb in model.p_b] for pa in model.p_a]
    if coupling != "comonotonic":
        raise ValueError("Неизвестная модель зависимости")
    order = sorted(range(len(model.rates)), key=lambda i: model.rates[i])

    def intervals(probs: Sequence[float]) -> Dict[int, Tuple[float, float]]:
        left = 0.0
        result = {}
        for i in order:
            right = left + probs[i]
            result[i] = (left, right)
            left = right
        return result

    a, b = intervals(model.p_a), intervals(model.p_b)
    return [
        [max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
         for j in range(len(model.rates))]
        for i in range(len(model.rates))
    ]


def joint_states(model: Model) -> List[State]:
    """Девять независимых состояний ``(r_A, r_B)`` с затратами и сожалениями."""
    model.validate()
    costs = {s: model.costs(s) for s in model.supplier_names}
    probs = {s: model.probs(s) for s in model.supplier_names}
    states: List[State] = []
    for i, rate_a in enumerate(model.rates):
        for j, rate_b in enumerate(model.rates):
            cost_a = costs["A"][i]
            cost_b = costs["B"][j]
            minimum = min(cost_a, cost_b)
            states.append({
                "i": i,
                "j": j,
                "rate_a": rate_a,
                "rate_b": rate_b,
                "probability": probs["A"][i] * probs["B"][j],
                "cost_a": cost_a,
                "cost_b": cost_b,
                "minimum": minimum,
                "regret_a": cost_a - minimum,
                "regret_b": cost_b - minimum,
            })
    return states


def emv(model: Model) -> Dict:
    """Ожидаемая денежная стоимость: ``E[C_a] = sum_i p_ai C_ai -> min``."""
    model.validate()
    costs = {s: model.costs(s) for s in model.supplier_names}
    expected_costs = {s: model.expected_cost(s) for s in model.supplier_names}
    saving = expected_costs["A"] - expected_costs["B"]
    return {
        "costs": costs,
        "expected_costs": expected_costs,
        "expected_repair": {s: model.expected_repair(s) for s in model.supplier_names},
        "expected_defects": model.mean_defects,
        "expected_rates": model.expected_rate,
        "choice": choose(expected_costs),
        "saving_vs_A": saving,
        "per_switch_saving_rub": saving * 1000 / model.N,
        "excess_over_best_mean": {
            s: expected_costs[s] - min(expected_costs.values())
            for s in model.supplier_names
        },
    }


def eol(model: Model) -> Dict:
    """Ожидаемая упущенная выгода в совместных состояниях.

    ``EOL_a = sum_s p(s) [C_a(s) - min_b C_b(s)]``.
    При одном совместном распределении EOL отличается от EMV на общую
    константу ``E[C_min]``, поэтому ранжирование совпадает.
    """
    states = joint_states(model)
    expected_minimum = sum(s["probability"] * s["minimum"] for s in states)
    # Вычитание одной константы сохраняет ранжирование EMV и не накапливает
    # погрешность прямой суммы сожалений (например, при c=0).
    expected_costs = {s: model.expected_cost(s) for s in model.supplier_names}
    expected_regret = {s: expected_costs[s] - expected_minimum for s in model.supplier_names}
    return {
        "coupling": "independent",
        "states": states,
        "expected_minimum": expected_minimum,
        "expected_regret": expected_regret,
        "choice": choose(expected_regret),
        "excess_over_best_mean": {
            "A": expected_regret["A"] - min(expected_regret.values()),
            "B": expected_regret["B"] - min(expected_regret.values()),
        },
    }


def risk(model: Model) -> Dict:
    """Дисперсия/SD, распределение разницы ``D=C_B-C_A`` и вероятности."""
    model.validate()
    names = model.supplier_names
    costs = {s: model.costs(s) for s in names}
    probs = {s: model.probs(s) for s in names}
    means = {s: model.expected_cost(s) for s in names}
    variances = {s: variance(costs[s], probs[s]) for s in names}
    sds = {s: math.sqrt(variances[s]) for s in names}

    diff_dist: Dict[float, float] = {}
    for pa, ca in zip(probs["A"], costs["A"]):
        for pb, cb in zip(probs["B"], costs["B"]):
            d = round(cb - ca, 10)
            diff_dist[d] = diff_dist.get(d, 0.0) + pa * pb

    p_a_cheaper = sum(p for d, p in diff_dist.items() if d > 0)
    p_b_cheaper = sum(p for d, p in diff_dist.items() if d < 0)
    p_tie = sum(p for d, p in diff_dist.items() if abs(d) < 1e-12)
    expected_diff = sum(d * p for d, p in diff_dist.items())
    diff_variance = sum(p * (d - expected_diff) ** 2 for d, p in diff_dist.items())

    return {
        "means": means,
        "variances": variances,
        "sds": sds,
        "cv": {s: sds[s] / means[s] if means[s] > 0 else None for s in names},
        "cv_relative_to_price_a": {
            s: sds[s] / (means[s] - model.price_a)
            if means[s] - model.price_a > 0 else None for s in names
        },
        "diff_distribution": {str(k): v for k, v in sorted(diff_dist.items())},
        "p_a_cheaper": p_a_cheaper,
        "p_b_cheaper": p_b_cheaper,
        "p_tie": p_tie,
        "expected_diff": expected_diff,
        "diff_variance": diff_variance,
        "diff_sd": math.sqrt(diff_variance),
        "worst": {s: max(costs[s]) for s in names},
        "best": {s: min(costs[s]) for s in names},
    }


def mean_sd(model: Model, lambda_: float = 0.0) -> Dict:
    """Критерий ``J_a = E[C_a] + lambda * SD_a -> min``."""
    if not _is_finite_number(lambda_) or lambda_ < 0:
        raise ValueError("lambda должно быть конечным неотрицательным числом")
    model.validate()
    names = model.supplier_names
    means = {s: model.expected_cost(s) for s in names}
    sds = {s: stdev(model.costs(s), model.probs(s)) for s in names}
    scores = {s: means[s] + lambda_ * sds[s] for s in names}
    denom = sds["B"] - sds["A"]
    if abs(denom) <= 1e-12:
        lambda_star = None
        threshold = {
            "feasible": False,
            "reason": "стандартные отклонения совпадают; знак разницы не меняется",
        }
    else:
        lambda_star = (means["A"] - means["B"]) / denom
        threshold = {
            "feasible": lambda_star >= 0,
            "reason": "порог вне области lambda >= 0" if lambda_star < 0 else "порог достижим",
        }
    return {
        "lambda": lambda_,
        "means": means,
        "sds": sds,
        "scores": scores,
        "choice": choose(scores),
        "lambda_star": lambda_star,
        "threshold": threshold,
    }


def utility(model: Model, rho: float = 100.0) -> Dict:
    """Экспоненциальная полезность затрат и эквивалент уверенности."""
    if not _is_finite_number(rho) or rho <= 0:
        raise ValueError("rho должно быть конечным положительным числом")
    model.validate()
    names = model.supplier_names
    costs = {s: model.costs(s) for s in names}
    probs = {s: model.probs(s) for s in names}
    means = {s: model.expected_cost(s) for s in names}
    result: Dict = {"rho": rho}
    for s in names:
        ce = ce_cost(costs[s], probs[s], rho)
        result[s] = {"ce": ce, "risk_premium": ce - means[s]}
    result["choice"] = choose({s: result[s]["ce"] for s in names})
    return result


def sensitivity(model: Model) -> Dict:
    """Пороги равенства средних и одномерные сценарии."""
    model.validate()
    ev_a = model.expected_cost("A")
    ev_b = model.expected_cost("B")
    delta_star = model.expected_repair("B") - model.expected_repair("A")

    dx = (sum(p * x for p, x in zip(model.p_b, model.defects))
          - sum(p * x for p, x in zip(model.p_a, model.defects)))
    c_star = model.discount / dx if abs(dx) > 1e-12 else None
    rate_star = (model.expected_rate["A"] + model.discount / (model.N * model.c)
                 if model.c > 0 else None)

    scenarios = []
    for d in (0, 10, 20, 30, 37, 50):
        m = replace(model, discount=d)
        m.validate()
        scenarios.append({
            "discount": d,
            "ev_A": m.expected_cost("A"),
            "ev_B": m.expected_cost("B"),
            "choice": best(m),
        })

    repair_scenarios = []
    for c in (0.3, 0.5, 0.6167, 0.8, 1.0):
        m = replace(model, c=c)
        m.validate()
        repair_scenarios.append({
            "c": c,
            "ev_A": m.expected_cost("A"),
            "ev_B": m.expected_cost("B"),
            "choice": best(m),
        })

    return {
        "delta_star": delta_star,
        "c_star": c_star,
        "rate_b_star": rate_star,
        "discount_scenarios": scenarios,
        "repair_scenarios": repair_scenarios,
        "current": {"ev_A": ev_a, "ev_B": ev_b, "choice": best(model)},
    }


def simulation(model: Model, runs: int = 100_000, seed: int = 42) -> Dict:
    """Монте-Карло: независимые пары партий и выборочные средние."""
    if not isinstance(runs, int) or isinstance(runs, bool) or runs <= 0:
        raise ValueError("runs должно быть положительным целым числом")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed должно быть целым числом")
    model.validate()
    rng = random.Random(seed)
    costs = {s: model.costs(s) for s in model.supplier_names}
    probs = {s: model.probs(s) for s in model.supplier_names}

    def sample(supplier: str) -> float:
        u = rng.random()
        acc = 0.0
        for p, c in zip(probs[supplier], costs[supplier]):
            acc += p
            if u <= acc:
                return c
        return costs[supplier][-1]

    sum_a = sum_b = 0.0
    p_a_cheaper = 0
    diff_counts: Dict[float, int] = {}
    for _ in range(runs):
        a = sample("A")
        b = sample("B")
        sum_a += a
        sum_b += b
        if a < b:
            p_a_cheaper += 1
        d = round(b - a, 10)
        diff_counts[d] = diff_counts.get(d, 0) + 1

    return {
        "runs": runs,
        "seed": seed,
        "mean_A": sum_a / runs,
        "mean_B": sum_b / runs,
        "p_a_cheaper": p_a_cheaper / runs,
        "diff_distribution": {str(k): v / runs for k, v in sorted(diff_counts.items())},
    }


def build_core_results(
    model: Model,
    lambda_: float = 0.0,
    rho: float = 100.0,
    runs: int = 100_000,
    seed: int = 42,
    include_simulation: bool = True,
) -> Dict:
    """Компактный результат для CLI и демонстрации (не заменяет полный JSON)."""
    model.validate()
    result = {
        "schema_version": 1,
        "scope": "core",
        "model": {
            "N": model.N,
            "c": model.c,
            "discount": model.discount,
            "price_a": model.price_a,
            "rates": list(model.rates),
            "p_a": list(model.p_a),
            "p_b": list(model.p_b),
        },
        "emv": emv(model),
        "eol": eol(model),
        "risk": risk(model),
        "mean_sd": mean_sd(model, lambda_),
        "utility": utility(model, rho),
        "sensitivity": sensitivity(model),
    }
    if include_simulation:
        result["simulation"] = simulation(model, runs, seed)
    return result


# ---------------------------------------------------------------------------
# Общая постановка: m поставщиков с собственными наборами уровней
# ---------------------------------------------------------------------------

MAX_JOINT_STATES = 100_000


@dataclass(frozen=True)
class Supplier:
    """Поставщик с собственной сеткой долей брака и вероятностями.

    ``rates`` — доли брака (безразмерные, строго возрастают), ``probabilities`` —
    вероятности уровней, ``price_offset`` — смещение цены относительно общей
    опорной цены ``reference_price``, тыс. руб.
    """

    id: str
    rates: Tuple[float, ...]
    probabilities: Tuple[float, ...]
    price_offset: float = 0.0

    def validate(self) -> None:
        """Строгая проверка нового формата: уровни и вероятности."""
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("id поставщика должен быть непустой строкой")
        if self.id == "tie":
            raise ValueError("id 'tie' зарезервирован для ничьей")
        if not self.rates or not self.probabilities:
            raise ValueError(f"{self.id}: нужны хотя бы один уровень и одна вероятность")
        if len(self.rates) != len(self.probabilities):
            raise ValueError(f"{self.id}: длины rates и probabilities должны совпадать")
        if any(not _is_finite_number(r) or not 0 <= r <= 1 for r in self.rates):
            raise ValueError(f"{self.id}: доли брака должны быть конечными числами в [0, 1]")
        if any(b <= a for a, b in zip(self.rates, self.rates[1:])):
            raise ValueError(f"{self.id}: уровни должны строго возрастать и не повторяться")
        if any(not _is_finite_number(p) or not 0 <= p <= 1 for p in self.probabilities):
            raise ValueError(f"{self.id}: вероятности должны быть конечными числами в [0, 1]")
        if abs(sum(self.probabilities) - 1.0) > 1e-9:
            raise ValueError(f"{self.id}: сумма вероятностей должна быть 1")
        if not _is_finite_number(self.price_offset):
            raise ValueError(f"{self.id}: смещение цены должно быть конечным числом")


@dataclass
class GeneralModel:
    """Общая модель: m поставщиков, у каждого своя сетка уровней.

    ``reference_price`` — общая опорная цена $P_0$ (тыс. руб.), относительно неё
    заданы смещения ``price_offset``. Флаг ``price_known`` относится к отчёту:
    при известной цене проверяется неотрицательность абсолютных цен.
    """

    N: int = 10_000
    c: float = 0.5
    reference_price: float = 0.0
    suppliers: Tuple[Supplier, ...] = (
        Supplier("A", (0.01, 0.02, 0.03), (0.7, 0.2, 0.1), 0.0),
        Supplier("B", (0.01, 0.02, 0.03), (0.3, 0.4, 0.3), -37.0),
    )
    price_known: bool = False

    def validate(self) -> None:
        """Проверить новый формат целиком, включая целые дефекты и цены."""
        if not isinstance(self.N, int) or isinstance(self.N, bool) or self.N <= 0:
            raise ValueError("N должно быть положительным целым числом")
        if not _is_finite_number(self.c) or self.c < 0:
            raise ValueError("c должно быть конечным неотрицательным числом")
        if not _is_finite_number(self.reference_price) or self.reference_price < 0:
            raise ValueError("reference_price должно быть конечным неотрицательным числом")
        if not isinstance(self.suppliers, tuple) or not self.suppliers:
            raise ValueError("suppliers должен быть непустым tuple")
        ids = [s.id for s in self.suppliers]
        if len(set(ids)) != len(ids):
            raise ValueError("id поставщиков должны быть уникальны")
        for supplier in self.suppliers:
            supplier.validate()
            defects = [self.N * r for r in supplier.rates]
            if any(abs(x - round(x)) > 1e-9 for x in defects):
                raise ValueError(f"{supplier.id}: N * rate должно давать целое число дефектов")
            for r in supplier.rates:
                cost = self.reference_price + supplier.price_offset + self.N * self.c * r
                if not math.isfinite(cost):
                    raise ValueError(f"{supplier.id}: затраты не представимы числом double")
            if self.price_known and self.reference_price + supplier.price_offset < -1e-9:
                raise ValueError(f"{supplier.id}: абсолютная цена партии отрицательна")

    @property
    def state_count(self) -> int:
        """Число строк полного декартова произведения уровней."""
        total = 1
        for supplier in self.suppliers:
            total *= len(supplier.rates)
        return total


def _general_mean_rate(supplier: Supplier) -> float:
    """Средняя доля брака поставщика по его собственным уровням."""
    return sum(p * r for p, r in zip(supplier.probabilities, supplier.rates))


def _general_norm_costs(model: GeneralModel, supplier: Supplier) -> List[float]:
    """Затраты без общей цены $P_0$: ``d_a + N c r``.

    Нормировка нужна, чтобы разности и допуски не теряли точность при
    огромной общей цене: сравниваются только смещение и ремонт.
    """
    return [supplier.price_offset + model.N * model.c * r for r in supplier.rates]


def _general_pair(model: GeneralModel, pair: Sequence[str] | None) -> Tuple[Supplier, Supplier]:
    """Выбрать пару поставщиков: по умолчанию первые две альтернативы."""
    by_id = {s.id: s for s in model.suppliers}
    if pair is None:
        if len(model.suppliers) < 2:
            raise ValueError("для сравнения нужны две альтернативы")
        if len(model.suppliers) > 2:
            raise ValueError("при трёх и более альтернативах укажите пару явно")
        return model.suppliers[0], model.suppliers[1]
    if len(pair) != 2:
        raise ValueError("пара задаётся двумя id")
    if pair[0] == pair[1] or pair[0] not in by_id or pair[1] not in by_id:
        raise ValueError("пара должна содержать два разных известных id")
    return by_id[pair[0]], by_id[pair[1]]


def general_emv(model: GeneralModel) -> Dict:
    """EMV общего вида: средние затраты и множество всех лучших id."""
    model.validate()
    ids = [s.id for s in model.suppliers]
    means_norm = {s.id: _general_mean_rate(s) * model.N * model.c + s.price_offset
                  for s in model.suppliers}
    expected_costs = {s.id: means_norm[s.id] + model.reference_price for s in model.suppliers}
    reference = ids[0]
    best_value = min(means_norm.values())
    winners = [s for s in ids if abs(means_norm[s] - best_value) <= 1e-9 + 1e-12 * abs(best_value)]
    return {
        "reference_supplier_id": reference,
        "costs": {s.id: _general_norm_costs(model, s) for s in model.suppliers},
        "expected_costs": expected_costs,
        "expected_costs_normalized": means_norm,
        "expected_repair": {s.id: model.N * model.c * _general_mean_rate(s) for s in model.suppliers},
        "expected_defects": {s.id: model.N * _general_mean_rate(s) for s in model.suppliers},
        "expected_rates": {s.id: _general_mean_rate(s) for s in model.suppliers},
        "winners": winners,
        "choice": winners[0] if len(winners) == 1 else "tie",
        "saving_vs_reference": {s: means_norm[reference] - means_norm[s] for s in ids},
        "excess_over_best_mean": {s: means_norm[s] - best_value for s in ids},
    }


def general_eol(
    model: GeneralModel,
    include_states: bool = False,
    max_states: int = MAX_JOINT_STATES,
) -> Dict:
    """EOL по всем совместным состояниям при независимости.

    Совместные строки обходятся потоково через ``itertools.product``; полные
    строки возвращаются только по явному запросу. Число состояний ограничено
    ``max_states`` — это ограничение ресурсов, а не математики.
    """
    model.validate()
    total = model.state_count
    if total > max_states:
        raise ValueError(
            f"совместных состояний {total} больше лимита {max_states}; "
            "увеличьте max_states явно или уменьшите модель"
        )
    ids = [s.id for s in model.suppliers]
    norm_costs = [_general_norm_costs(model, s) for s in model.suppliers]
    probs = [s.probabilities for s in model.suppliers]
    expected_minimum = 0.0
    regrets = {s: 0.0 for s in ids}
    positive_states = 0
    states: List[Dict] = []
    ranges = [range(len(s.rates)) for s in model.suppliers]
    for indices in itertools.product(*ranges):
        q = 1.0
        for i, idx in enumerate(indices):
            q *= probs[i][idx]
        if q == 0.0 and all(probs[i][idx] > 0 for i, idx in enumerate(indices)):
            raise ValueError(
                "произведение вероятностей не представимо числом double; "
                "точный потоковый результат невозможен"
            )
        costs = [norm_costs[i][idx] for i, idx in enumerate(indices)]
        minimum = min(costs)
        expected_minimum += q * minimum
        if q > 0:
            positive_states += 1
        for i, s in enumerate(ids):
            regrets[s] += q * (costs[i] - minimum)
        if include_states:
            states.append({
                "indices": list(indices),
                "rates": [model.suppliers[i].rates[idx] for i, idx in enumerate(indices)],
                "costs_normalized": {ids[i]: costs[i] for i in range(len(ids))},
                "regrets": {ids[i]: costs[i] - minimum for i in range(len(ids))},
                "probability": q,
                "minimum": minimum,
            })
    best_value = min(regrets.values())
    winners = [s for s in ids if abs(regrets[s] - best_value) <= 1e-9 + 1e-12 * abs(best_value)]
    result = {
        "coupling": "independent",
        "assumption": "качества партий независимы; это дополнительное допущение",
        "state_count": total,
        "positive_state_count": positive_states,
        "expected_minimum": expected_minimum + model.reference_price,
        "expected_regret": regrets,
        "winners": winners,
        "choice": winners[0] if len(winners) == 1 else "tie",
    }
    if include_states:
        result["states"] = states
    return result


def _general_marginals(model: GeneralModel) -> Dict:
    """Средние (без $P_0$), дисперсии, SD и CV каждого поставщика."""
    ids = [s.id for s in model.suppliers]
    means: Dict[str, float] = {}
    variances: Dict[str, float] = {}
    sds: Dict[str, float] = {}
    for supplier in model.suppliers:
        costs = _general_norm_costs(model, supplier)
        mean = sum(p * x for p, x in zip(supplier.probabilities, costs))
        var = sum(p * (x - mean) ** 2 for p, x in zip(supplier.probabilities, costs))
        means[supplier.id] = mean
        variances[supplier.id] = var
        sds[supplier.id] = math.sqrt(var)
    return {
        "means_normalized": means,
        "variances": variances,
        "sds": sds,
        "cv": {s: (sds[s] / means[s] if means[s] > 0 else None) for s in ids},
    }


def general_risk(model: GeneralModel, pair: Sequence[str] | None = None) -> Dict:
    """Маргинальный риск всех id и распределение разницы выбранной пары."""
    model.validate()
    result: Dict = _general_marginals(model)
    if len(model.suppliers) < 2:
        result["diff"] = None
        return result
    first, second = _general_pair(model, pair)
    diff_dist: Dict[float, float] = {}
    for p1, c1 in zip(first.probabilities, _general_norm_costs(model, first)):
        for p2, c2 in zip(second.probabilities, _general_norm_costs(model, second)):
            d = round(c2 - c1, 10)
            diff_dist[d] = diff_dist.get(d, 0.0) + p1 * p2
    p_first_cheaper = sum(p for d, p in diff_dist.items() if d > 0)
    p_second_cheaper = sum(p for d, p in diff_dist.items() if d < 0)
    p_tie = sum(p for d, p in diff_dist.items() if abs(d) < 1e-12)
    expected_diff = sum(d * p for d, p in diff_dist.items())
    diff_variance = sum(p * (d - expected_diff) ** 2 for d, p in diff_dist.items())
    result["diff"] = {
        "first": first.id,
        "second": second.id,
        "distribution": {str(k): v for k, v in sorted(diff_dist.items())},
        "p_first_cheaper": p_first_cheaper,
        "p_second_cheaper": p_second_cheaper,
        "p_tie": p_tie,
        "expected_diff": expected_diff,
        "diff_variance": diff_variance,
        "diff_sd": math.sqrt(diff_variance),
    }
    return result


def general_mean_sd(model: GeneralModel, lambda_: float = 0.0) -> Dict:
    """Критерий ``J_a = E[C_a] + lambda * SD_a`` для всех id."""
    if not _is_finite_number(lambda_) or lambda_ < 0:
        raise ValueError("lambda должно быть конечным неотрицательным числом")
    model.validate()
    risk_data = _general_marginals(model)
    scores_norm = {s: risk_data["means_normalized"][s] + lambda_ * risk_data["sds"][s]
                   for s in risk_data["means_normalized"]}
    best_value = min(scores_norm.values())
    winners = [s for s, v in scores_norm.items()
               if abs(v - best_value) <= 1e-9 + 1e-12 * abs(best_value)]
    return {
        "lambda": lambda_,
        "means": {s: v + model.reference_price for s, v in risk_data["means_normalized"].items()},
        "sds": risk_data["sds"],
        "scores": {s: v + model.reference_price for s, v in scores_norm.items()},
        "winners": winners,
        "choice": winners[0] if len(winners) == 1 else "tie",
    }


def general_utility(model: GeneralModel, rho: float = 100.0) -> Dict:
    """Экспоненциальная полезность: CE каждого id и множество лучших."""
    if not _is_finite_number(rho) or rho <= 0:
        raise ValueError("rho должно быть конечным положительным числом")
    model.validate()
    result: Dict = {"rho": rho}
    for supplier in model.suppliers:
        costs = _general_norm_costs(model, supplier)
        ce = ce_cost(costs, supplier.probabilities, rho) + model.reference_price
        mean = sum(p * x for p, x in zip(supplier.probabilities, costs)) + model.reference_price
        result[supplier.id] = {"ce": ce, "risk_premium": ce - mean}
    ids = [s.id for s in model.suppliers]
    best_value = min(result[s]["ce"] for s in ids)
    winners = [s for s in ids if abs(result[s]["ce"] - best_value) <= 1e-9 + 1e-12 * abs(best_value)]
    result["winners"] = winners
    result["choice"] = winners[0] if len(winners) == 1 else "tie"
    return result


def _threshold(a_coef: float, b_coef: float, lower: float,
               upper: float | None, *, integer: bool = False) -> Dict:
    """Классифицировать порог для разности ``D(x) = a*x + b``.

    Возвращает ``kind``: ``single`` (один корень), ``none`` (переноса нет),
    ``all`` (равенство при всех допустимых значениях). ``feasible`` показывает,
    попадает ли корень в объявленную область параметра.
    """
    region = {"lower": lower, "upper": upper, "integer": integer}
    if abs(a_coef) <= 0:
        if abs(b_coef) <= 0:
            return {"kind": "all", "root": None, "feasible": True, "region": region}
        return {"kind": "none", "root": None, "feasible": False, "region": region}
    root = -b_coef / a_coef
    feasible = root >= lower - 1e-12 and (upper is None or root <= upper + 1e-12)
    if integer and feasible:
        feasible = abs(root - round(root)) <= 1e-9
    return {"kind": "single", "root": root, "feasible": feasible, "region": region}


def general_sensitivity(model: GeneralModel, pair: Sequence[str] | None = None) -> Dict:
    """Попарные пороги равенства средних с классификацией корня."""
    model.validate()
    if len(model.suppliers) < 2:
        return {"pair": None, "reason": "для порогов нужны две альтернативы", "thresholds": {}}
    first, second = _general_pair(model, pair)
    mu_first = _general_mean_rate(first)
    mu_second = _general_mean_rate(second)
    h = mu_second - mu_first
    delta = first.price_offset - second.price_offset
    nc = model.N * model.c
    discount_upper = model.reference_price + first.price_offset if model.price_known else None
    rate_upper = max(second.rates)
    rate_lower = min(second.rates)
    return {
        "pair": {"a": first.id, "b": second.id, "mu_a": mu_first, "mu_b": mu_second,
                 "h_ab": h, "delta_ab": delta},
        "thresholds": {
            "discount": _threshold(-1.0, nc * h, 0.0, discount_upper),
            "repair_cost": _threshold(model.N * h, -delta, 0.0, None),
            "mean_rate_b": _threshold(nc, -nc * mu_first - delta, rate_lower, rate_upper),
            "batch_size": _threshold(model.c * h, -delta, 1.0, None, integer=True),
        },
    }


def general_simulation(model: GeneralModel, runs: int = 100_000, seed: int = 42) -> Dict:
    """Монте-Карло общего вида: общий seed, поставщики в порядке tuple."""
    if not isinstance(runs, int) or isinstance(runs, bool) or runs <= 0:
        raise ValueError("runs должно быть положительным целым числом")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed должно быть целым числом")
    model.validate()
    rng = random.Random(seed)
    ids = [s.id for s in model.suppliers]
    costs = [_general_norm_costs(model, s) for s in model.suppliers]
    probs = [s.probabilities for s in model.suppliers]

    def sample(index: int) -> float:
        u = rng.random()
        acc = 0.0
        for p, value in zip(probs[index], costs[index]):
            acc += p
            if u <= acc:
                return value
        return costs[index][-1]

    sums = [0.0] * len(ids)
    wins = {s: 0 for s in ids}
    for _ in range(runs):
        values = [sample(i) for i in range(len(ids))]
        for i, value in enumerate(values):
            sums[i] += value
        minimum = min(values)
        winner = ids[values.index(minimum)]
        wins[winner] += 1
    return {
        "runs": runs,
        "seed": seed,
        "means": {s: sums[i] / runs + model.reference_price for i, s in enumerate(ids)},
        "p_best": {s: wins[s] / runs for s in ids},
    }


def build_general_results(
    model: GeneralModel,
    lambda_: float = 0.0,
    rho: float = 100.0,
    runs: int = 100_000,
    seed: int = 42,
    include_simulation: bool = True,
) -> Dict:
    """Результат одного сценария общего вида (схема версии 2)."""
    model.validate()
    ids = [s.id for s in model.suppliers]
    report_pair = (ids[0], ids[1]) if len(ids) > 2 else None
    result = {
        "schema_version": 2,
        "scope": "general",
        "level_semantics": "exact_count",
        "model": {
            "N": model.N,
            "c": model.c,
            "reference_price": model.reference_price,
            "price_known": model.price_known,
            "suppliers": [
                {
                    "id": s.id,
                    "rates": list(s.rates),
                    "probabilities": list(s.probabilities),
                    "price_offset": s.price_offset,
                }
                for s in model.suppliers
            ],
        },
        "preferences": {"lambda": lambda_, "rho": rho},
        "assumptions": {"coupling": "independent"},
        "state_count": model.state_count,
        "emv": general_emv(model),
        "eol": general_eol(model),
        "risk": general_risk(model, pair=report_pair),
        "mean_sd": general_mean_sd(model, lambda_),
        "utility": general_utility(model, rho),
        "sensitivity": general_sensitivity(model, pair=report_pair),
    }
    if include_simulation:
        result["simulation"] = general_simulation(model, runs, seed)
    return result


def _general_suppliers_base() -> Tuple[Supplier, Supplier]:
    return (
        Supplier("A", (0.01, 0.02, 0.03), (0.7, 0.2, 0.1), 0.0),
        Supplier("B", (0.01, 0.02, 0.03), (0.3, 0.4, 0.3), -37.0),
    )


def build_general_examples() -> Dict:
    """Синтетические эталоны общего вида; файлы не пишет.

    ``base`` повторяет исходную задачу, ``unequal_2x3`` и ``teacher_grids``
    проверяют разные длины и разрывы уровней, ``three_2x3x1`` — три
    альтернативы. Веса примеров выбраны для проверки, а не даны условием.
    """
    a, b = _general_suppliers_base()
    unequal = (
        Supplier("A", (0.01, 0.03), (0.75, 0.25), 0.0),
        Supplier("B", (0.02, 0.03, 0.11), (0.5, 0.25, 0.25), -37.0),
    )
    third = Supplier("C", (0.01,), (1.0,), 10.0)
    teacher = (
        Supplier("A", (0.01, 0.02, 0.03, 0.05, 0.07), (0.2, 0.2, 0.2, 0.2, 0.2), 0.0),
        Supplier("B", (0.02, 0.03, 0.11), (1 / 3, 1 / 3, 1 / 3), -37.0),
    )
    cases = {
        "base": GeneralModel(suppliers=(a, b)),
        "unequal_2x3": GeneralModel(suppliers=unequal),
        "three_2x3x1": GeneralModel(suppliers=unequal + (third,)),
        "teacher_grids": GeneralModel(suppliers=teacher),
    }
    return {
        "schema_version": 2,
        "scope": "general_examples",
        "note": "Синтетические учебные примеры для проверки разных сеток; не данные компании «Луч».",
        "cases": {name: build_general_results(model) for name, model in cases.items()},
    }
