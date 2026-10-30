#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Совместимый вход задачи «Луч» и командный интерфейс.

Полный режим без аргументов воспроизводит прежний отчёт и ``results.json``.
Компактный режим ``--scope core`` показывает только методы доклада и
не перезаписывает исторический JSON без явного ``--output``.

Формулы живут в :mod:`solver_core`; справочные блоки — в
:mod:`solver_extras`. Этот файл — фасад и CLI, а не второй вычислитель.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import solver_core as core
import solver_extras as extras
from solver_core import (  # re-export прежних имён
    Model,
    best,
    ce_cost,
    choose,
    emv,
    eol,
    expected,
    joint_probabilities,
    joint_states,
    mean_sd,
    risk,
    sensitivity,
    simulation,
    stdev,
    utility,
    variance,
)
from solver_core import (  # общая постановка: m поставщиков и разные k_a
    MAX_JOINT_STATES,
    GeneralModel,
    Supplier,
    build_general_examples,
    build_general_results,
    general_emv,
    general_eol,
    general_mean_sd,
    general_risk,
    general_sensitivity,
    general_simulation,
    general_utility,
)
from solver_extras import (  # re-export справочных блоков и helpers
    _clamp_zero,
    _log_binom_pmf,
    _posterior_mean,
    _posterior_weights,
    _sample_log_weights,
    criteria_block,
    evpi_block,
    evsi_block,
    probability_block,
    risk_block,
    sensitivity_block,
    simulate_block,
    split_block,
    tree_block,
    utility_block,
)

OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Совместимые блоки
# ---------------------------------------------------------------------------


def emv_block(model: Model) -> Dict:
    """Прежняя схема EMV/EOL; новые методы остаются в ``solver_core``."""
    data = emv(model)
    eol_data = eol(model)
    ev = data["expected_costs"]
    eol_values = {s: ev[s] - eol_data["expected_minimum"] for s in model.supplier_names}
    return {
        "costs": data["costs"],
        "expected_costs": ev,
        "expected_repair": data["expected_repair"],
        "expected_defects": data["expected_defects"],
        "expected_rates": data["expected_rates"],
        "best": best(model),
        "saving_vs_A": data["saving_vs_A"],
        "eol": eol_values,
        "eol_coupling": "independent",
        "excess_over_best_mean": data["excess_over_best_mean"],
        "per_switch_saving_rub": data["per_switch_saving_rub"],
    }


def build_results(model: Model) -> Dict:
    """Полный исторический JSON со всеми расширениями."""
    model.validate()
    return {
        "model": {
            "N": model.N,
            "c": model.c,
            "discount": model.discount,
            "price_a": model.price_a,
            "rates": list(model.rates),
            "p_a": list(model.p_a),
            "p_b": list(model.p_b),
        },
        "emv": emv_block(model),
        "tree": tree_block(model),
        "criteria": criteria_block(model),
        "risk": risk_block(model),
        "probability": probability_block(model),
        "utility": utility_block(model),
        "evpi": evpi_block(model),
        "evsi": evsi_block(model),
        "sensitivity": sensitivity_block(model),
        "split": split_block(model),
        "simulation": simulate_block(model),
    }


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def _choice_label(choice: str) -> str:
    return "ничья" if choice == "tie" else choice


def print_report(res: Dict, price_known: Optional[bool] = None) -> None:
    """Прежний текстовый отчёт полного режима."""
    emv_data = res["emv"]
    print("=" * 72)
    print("ЗАДАЧА «ЛУЧ»: ВЫБОР ПОСТАВЩИКА")
    print("=" * 72)
    print(f"Партия: {fmt(res['model']['N'], 0)} шт.; ремонт: {fmt(res['model']['c'])} тыс. руб.; "
          f"скидка B: {fmt(res['model']['discount'])} тыс. руб.")
    relative = res["model"]["price_a"] == 0 if price_known is None else not price_known
    print("Затраты относительно P_A; абсолютная цена A не задана."
          if relative else f"Цена A включена: {fmt(res['model']['price_a'])} тыс. руб.")
    print()
    for s in ("A", "B"):
        print(f"Поставщик {s}:")
        print(f"  возможные затраты: {'; '.join(fmt(x) for x in emv_data['costs'][s])} тыс. руб.")
        print(f"  ожидаемый брак:   {fmt(emv_data['expected_defects'][s], 0)} шт.")
        print(f"  ожидаемый ремонт: {fmt(emv_data['expected_repair'][s])} тыс. руб.")
        print(f"  ожидаемые затраты:{fmt(emv_data['expected_costs'][s])} тыс. руб.")
    print()
    print(f"ЛУЧШЕЕ РЕШЕНИЕ: {emv_data['best']}; "
          f"экономия относительно A {fmt(emv_data['expected_costs']['A'] - min(emv_data['expected_costs'].values()))} тыс. руб.")
    value = min(emv_data["expected_costs"].values())
    label = f"P_A {'+' if value >= 0 else '-'} {fmt(abs(value))}" if relative else fmt(value)
    print(f"Ожидаемая стоимость лучшего решения: {label} тыс. руб.")
    print(f"Экономия B относительно A на переключатель: {fmt(emv_data['per_switch_saving_rub'], 1)} руб.")
    print(f"EOL (независимые партии): A={fmt(emv_data['eol']['A'])}, B={fmt(emv_data['eol']['B'])} тыс. руб.")
    print()
    print("Критерии (матрица общих состояний):")
    for name in ("emv", "wald", "minimin", "laplace", "savage"):
        block = res["criteria"][name]
        print(f"  {name:8s}: A={fmt(block['scores']['A'])}, B={fmt(block['scores']['B'])} -> {block['choice']}")
    scores = res["criteria"]["savage_joint"]
    print(f"  Сэвидж, все пары: A={fmt(scores['scores']['A'])}, B={fmt(scores['scores']['B'])} -> {scores['choice']}")
    print()
    print("Риск:")
    print(f"  sigma: A={fmt(res['risk']['sds']['A'])}, B={fmt(res['risk']['sds']['B'])} тыс. руб.")
    print(f"  P(A дешевле B) = {fmt(res['risk']['p_a_cheaper'], 3)}")
    print(f"  P(B дешевле A) = {fmt(res['risk']['p_b_cheaper'], 3)}")
    print()
    print("Пороги чувствительности:")
    print(f"  скидка > {fmt(res['sensitivity']['delta_star'])} тыс. руб.")
    c_star, rate_star = res["sensitivity"]["c_star"], res["sensitivity"]["rate_b_star"]
    print(f"  порог ремонта: {fmt(c_star, 6) if c_star is not None else 'не определён'} тыс. руб.")
    print(f"  порог среднего брака B: {fmt(100 * rate_star) + ' %' if rate_star is not None else 'не определён при c=0'}")
    print()
    print("EVPI:")
    for k in ("after_choice", "one_batch_A", "one_batch_B", "both_independent", "both_comonotonic"):
        print(f"  {k}: {fmt(res['evpi'][k], 3)} тыс. руб.")
    print()
    print("EVSI (тыс. руб.):")
    for row in res["evsi"]["table"]:
        print(f"  n={row['n']:3d}: A={fmt(row['A'], 3)}, B={fmt(row['B'], 3)}, обе={fmt(row['both'], 3)}")
    print()
    print(f"Монте-Карло: среднее A={fmt(res['simulation']['mean_A'], 3)}, "
          f"B={fmt(res['simulation']['mean_B'], 3)} тыс. руб.; "
          f"P(A<B)={fmt(res['simulation']['p_a_cheaper'], 3)}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Компактный режим
# ---------------------------------------------------------------------------


def print_core_report(res: Dict, price_known: Optional[bool] = None) -> None:
    """Короткий отчёт только по методам доклада."""
    model = res["model"]
    relative = model["price_a"] == 0 if price_known is None else not price_known
    print("Режим цены: " + ("относительно P_A" if relative else f"цена A={fmt(model['price_a'])} тыс. руб."))
    emv_data, eol_data = res["emv"], res["eol"]
    print(f"EMV: A={fmt(emv_data['expected_costs']['A'])}; B={fmt(emv_data['expected_costs']['B'])}; "
          f"выбор {_choice_label(emv_data['choice'])}")
    print(f"EOL: A={fmt(eol_data['expected_regret']['A'])}; B={fmt(eol_data['expected_regret']['B'])}; "
          f"выбор {_choice_label(eol_data['choice'])}")
    ms = res["mean_sd"]
    print(f"mean–SD (λ={fmt(ms['lambda'])}): A={fmt(ms['scores']['A'])}; B={fmt(ms['scores']['B'])}; "
          f"выбор {_choice_label(ms['choice'])}")
    ut = res["utility"]
    print(f"CE (ρ={fmt(ut['rho'])}): A={fmt(ut['A']['ce'])}; B={fmt(ut['B']['ce'])}; "
          f"выбор {_choice_label(ut['choice'])}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_float(text: str) -> float:
    """Принять десятичную запятую или точку, но не мусор вокруг числа."""
    value = text.strip().replace(",", ".")
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось число, получено {text!r}") from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("число должно быть конечным")
    return result


def _parse_int(text: str) -> int:
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось целое число, получено {text!r}") from exc


def _load_config(path: str) -> Dict:
    """Прочитать и строго проверить JSON-конфигурацию."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("config: верхний уровень должен быть объектом")
    allowed = {"model", "preferences", "simulation", "scope", "format", "price_known"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"config: неизвестные ключи: {', '.join(sorted(unknown))}")
    model_allowed = {"N", "c", "discount", "price_a", "p_a", "p_b"}
    if "model" in data:
        if not isinstance(data["model"], dict):
            raise ValueError("config.model должен быть объектом")
        unknown = set(data["model"]) - model_allowed
        if unknown:
            raise ValueError(f"config.model: неизвестные ключи: {', '.join(sorted(unknown))}")
    pref_allowed = {"lambda", "rho"}
    if "preferences" in data:
        if not isinstance(data["preferences"], dict):
            raise ValueError("config.preferences должен быть объектом")
        unknown = set(data["preferences"]) - pref_allowed
        if unknown:
            raise ValueError(f"config.preferences: неизвестные ключи: {', '.join(sorted(unknown))}")
    sim_allowed = {"runs", "seed"}
    if "simulation" in data:
        if not isinstance(data["simulation"], dict):
            raise ValueError("config.simulation должен быть объектом")
        unknown = set(data["simulation"]) - sim_allowed
        if unknown:
            raise ValueError(f"config.simulation: неизвестные ключи: {', '.join(sorted(unknown))}")
    if "scope" in data and data["scope"] not in ("full", "core"):
        raise ValueError("config.scope: ожидалось full или core")
    if "format" in data and data["format"] not in ("text", "json"):
        raise ValueError("config.format: ожидалось text или json")
    if "price_known" in data and not isinstance(data["price_known"], bool):
        raise ValueError("config.price_known должен быть bool")
    return data


def _model_from_args(args: argparse.Namespace, config: Dict) -> tuple[Model, bool]:
    """Собрать Model и флаг известной цены с приоритетом CLI > config."""
    values = {
        "N": 10_000,
        "c": 0.5,
        "discount": 37.0,
        "price_a": 0.0,
        "rates": (0.01, 0.02, 0.03),
        "p_a": (0.7, 0.2, 0.1),
        "p_b": (0.3, 0.4, 0.3),
    }
    config_model = config.get("model", {})
    for key in ("N", "c", "discount", "price_a", "p_a", "p_b"):
        if key in config_model:
            values[key] = config_model[key]
    if args.n is not None:
        values["N"] = args.n
    if args.repair_cost is not None:
        values["c"] = args.repair_cost
    if args.discount is not None:
        values["discount"] = args.discount
    if args.price_a is not None:
        values["price_a"] = args.price_a
    if args.p_a is not None:
        values["p_a"] = tuple(args.p_a)
    if args.p_b is not None:
        values["p_b"] = tuple(args.p_b)

    price_present = "price_a" in config_model or args.price_a is not None
    price_known = config.get("price_known")
    if args.price_a is not None:
        price_known = True
    if price_known is None:
        price_known = price_present
    model = Model(**values)
    model.validate()
    if not price_known and model.price_a != 0:
        raise ValueError("price_a задана, но price_known=false; укажите цену явно или оставьте 0")
    if price_known and model.price_a < model.discount:
        raise ValueError("известная цена A должна быть не меньше скидки B")
    return model, bool(price_known)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Задача «Луч»: полный отчёт или компактное ядро.",
    )
    parser.add_argument("--scope", choices=("full", "core"), default=None,
                        help="full — прежний JSON и отчёт; core — только методы доклада")
    parser.add_argument("--n", type=_parse_int, default=None, help="размер партии N")
    parser.add_argument("--repair-cost", type=_parse_float, default=None, help="ремонт одного, тыс. руб.")
    parser.add_argument("--discount", type=_parse_float, default=None, help="скидка B на партию, тыс. руб.")
    parser.add_argument("--price-a", type=_parse_float, default=None, help="известная цена A, тыс. руб.")
    parser.add_argument("--p-a", nargs=3, type=_parse_float, default=None, metavar=("P1", "P2", "P3"),
                        help="три вероятности A")
    parser.add_argument("--p-b", nargs=3, type=_parse_float, default=None, metavar=("P1", "P2", "P3"),
                        help="три вероятности B")
    parser.add_argument("--lambda", dest="lambda_", type=_parse_float, default=None,
                        help="вес SD для mean–SD (core)")
    parser.add_argument("--rho", type=_parse_float, default=None,
                        help="толерантность к риску для CE (core)")
    parser.add_argument("--runs", type=_parse_int, default=None, help="число прогонов Монте-Карло (core)")
    parser.add_argument("--seed", type=_parse_int, default=None, help="seed Монте-Карло (core)")
    parser.add_argument("--format", choices=("text", "json"), default=None,
                        help="формат stdout")
    parser.add_argument("--output", default=None, help="явный путь записи результата")
    parser.add_argument("--config", default=None, help="JSON-конфигурация")
    return parser


def _write_json(path: str, data: Dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    tmp.replace(target)


def write_general_export(path: str) -> Path:
    """Записать общий набор эталонов в отдельный файл (не ``results.json``).

    Канонический ``results.json`` — legacy-эталон схемы v1; писать в него
    общие результаты нельзя даже при ошибочно указанном пути. Собственный
    сериализатор добавляет перевод строки и не меняет прежний writer.
    """
    target = Path(path)
    legacy = (OUT_DIR / "results.json").resolve()
    if target.resolve() == legacy:
        raise ValueError("общий экспорт не перезаписывает канонический results.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(build_general_examples(), ensure_ascii=False, indent=2, allow_nan=False)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(target)
    return target


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Точка входа CLI. Возвращает код завершения, не вызывает ``sys.exit``."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        config = _load_config(args.config) if args.config else {}
        scope = args.scope or config.get("scope") or "full"
        output_format = args.format or config.get("format") or "text"

        # Полный режим не должен молча менять фиксированные таблицы.
        if scope == "full":
            if args.lambda_ is not None or args.rho is not None or args.runs is not None or args.seed is not None:
                raise ValueError("--lambda/--rho/--runs/--seed применимы только с --scope core")
            if "preferences" in config or "simulation" in config:
                raise ValueError("preferences/simulation в config применимы только с scope=core")

        model, price_known = _model_from_args(args, config)

        if scope == "core":
            lambda_ = args.lambda_ if args.lambda_ is not None else config.get("preferences", {}).get("lambda", 0.0)
            rho = args.rho if args.rho is not None else config.get("preferences", {}).get("rho", 100.0)
            runs = args.runs if args.runs is not None else config.get("simulation", {}).get("runs", 100_000)
            seed = args.seed if args.seed is not None else config.get("simulation", {}).get("seed", 42)
            if not isinstance(runs, int) or isinstance(runs, bool) or runs <= 0:
                raise ValueError("runs должно быть положительным целым числом")
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError("seed должно быть целым числом")
            result = core.build_core_results(
                model, lambda_, rho, runs, seed, include_simulation=(output_format == "json"),
            )
            result["price_known"] = price_known
            if output_format == "json":
                text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
                print(text)
            else:
                print_core_report(result, price_known)
            if args.output:
                _write_json(args.output, result)
            return 0

        result = build_results(model)
        out = Path(args.output) if args.output else OUT_DIR / "results.json"
        _write_json(str(out), result)
        if output_format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            print_report(result, price_known)
            print(f"JSON сохранён: {out}")
        return 0
    except (ValueError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
