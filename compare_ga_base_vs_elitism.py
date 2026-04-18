import argparse
import csv
import random
import time
from pathlib import Path
from statistics import mean

from geneticAlgorithmWithoutModific import (
    BORDER,
    ELITISM_COUNT,
    FUNCTION,
    MAX_GENERATIONS,
    POPULATION_SIZE,
    P_CROSSOVER,
    P_MUTATION,
    run_genetic_algorithm,
)


def _iterations_to_threshold(best_history, threshold, fallback):
    for idx, value in enumerate(best_history, start=1):
        if value < threshold:
            return idx
    return fallback


def _run_variant(*, variant_name, elitism_count, seed, args):
    random.seed(seed)
    started = time.perf_counter()
    result = run_genetic_algorithm(
        population_size=args.population_size,
        p_crossover=args.p_crossover,
        p_mutation=args.p_mutation,
        max_generations=args.max_generations,
        elitism_count=elitism_count,
        border=BORDER,
        function=FUNCTION,
    )
    elapsed = time.perf_counter() - started
    iterations_to_threshold = _iterations_to_threshold(
        result["best_fitness_history"], args.success_threshold, args.max_generations
    )
    best_value = result["best_value"]
    return {
        "variant": variant_name,
        "seed": seed,
        "best_value": best_value,
        "elapsed_time_s": elapsed,
        "iterations_to_threshold": iterations_to_threshold,
        "success": int(best_value < args.success_threshold),
    }


def _summarize(rows):
    values = [row["best_value"] for row in rows]
    times = [row["elapsed_time_s"] for row in rows]
    iters = [row["iterations_to_threshold"] for row in rows]
    successes = [row["success"] for row in rows]
    return {
        "best_found_value": min(values),
        "mean_best_value": mean(values),
        "mean_time_s": mean(times),
        "mean_iterations_to_threshold": mean(iters),
        "success_rate": mean(successes),
    }


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _winner_line(summary_by_variant):
    base = summary_by_variant["ga_base"]
    elitism = summary_by_variant["ga_elitism"]

    quality_winner = "ga_base" if base["mean_best_value"] < elitism["mean_best_value"] else "ga_elitism"
    speed_winner = "ga_base" if base["mean_time_s"] < elitism["mean_time_s"] else "ga_elitism"
    success_winner = "ga_base" if base["success_rate"] > elitism["success_rate"] else "ga_elitism"

    print("\nСравнение победителей по ключевым метрикам:")
    print(f"  Качество (mean_best_value): {quality_winner}")
    print(f"  Скорость (mean_time_s):     {speed_winner}")
    print(f"  Успешность (success_rate):  {success_winner}")


def main():
    parser = argparse.ArgumentParser(
        description="Сравнение ГА без модификации и ГА с элитизмом."
    )
    parser.add_argument("--runs", type=int, default=30, help="Число запусков на вариант")
    parser.add_argument("--success-threshold", type=float, default=-19.0, help="Порог успеха: f(x) < threshold")
    parser.add_argument("--seed-base", type=int, default=42, help="Базовый seed; на i-м запуске используется seed_base + i")
    parser.add_argument("--population-size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--max-generations", type=int, default=MAX_GENERATIONS)
    parser.add_argument("--p-crossover", type=float, default=P_CROSSOVER)
    parser.add_argument("--p-mutation", type=float, default=P_MUTATION)
    parser.add_argument("--elitism-count", type=int, default=ELITISM_COUNT, help="Top-k для варианта с элитизмом")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Папка для CSV-отчётов",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs должен быть > 0")
    if args.elitism_count < 0 or args.elitism_count >= args.population_size:
        parser.error("--elitism-count должен быть в диапазоне [0, population_size)")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs_rows = []
    by_variant = {"ga_base": [], "ga_elitism": []}

    for run_idx in range(args.runs):
        seed = args.seed_base + run_idx
        base_row = _run_variant(
            variant_name="ga_base",
            elitism_count=0,
            seed=seed,
            args=args,
        )
        elitism_row = _run_variant(
            variant_name="ga_elitism",
            elitism_count=args.elitism_count,
            seed=seed,
            args=args,
        )
        for row in (base_row, elitism_row):
            row["run_index"] = run_idx
            runs_rows.append(row)
            by_variant[row["variant"]].append(row)

    summary_rows = []
    summary_by_variant = {}
    for variant_name, rows in by_variant.items():
        summary = _summarize(rows)
        summary_by_variant[variant_name] = summary
        summary_rows.append({"variant": variant_name, **summary})

    runs_csv = args.output_dir / "ga_base_vs_elitism_runs.csv"
    summary_csv = args.output_dir / "ga_base_vs_elitism_summary.csv"

    _write_csv(
        runs_csv,
        [
            "variant",
            "run_index",
            "seed",
            "best_value",
            "elapsed_time_s",
            "iterations_to_threshold",
            "success",
        ],
        runs_rows,
    )
    _write_csv(
        summary_csv,
        [
            "variant",
            "best_found_value",
            "mean_best_value",
            "mean_time_s",
            "mean_iterations_to_threshold",
            "success_rate",
        ],
        summary_rows,
    )

    print("=" * 72)
    print("Сравнение GA: base vs elitism")
    print(f"Запусков на вариант: {args.runs}")
    print(f"Порог успеха: f < {args.success_threshold}")
    print(f"CSV (прогоны): {runs_csv}")
    print(f"CSV (сводка):  {summary_csv}")
    print("=" * 72)
    for row in summary_rows:
        print(
            f"{row['variant']:<12} | best_found={row['best_found_value']:.8f} | "
            f"mean_best={row['mean_best_value']:.8f} | mean_time={row['mean_time_s']:.4f}s | "
            f"mean_iter={row['mean_iterations_to_threshold']:.2f} | success_rate={row['success_rate']:.2%}"
        )

    _winner_line(summary_by_variant)


if __name__ == "__main__":
    main()
