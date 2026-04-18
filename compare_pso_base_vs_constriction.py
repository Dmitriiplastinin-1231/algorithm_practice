import argparse
import csv
import math
import time
from pathlib import Path
from statistics import mean

from pso import PSO


FUNCTION = lambda x, y: -(
    math.fabs(
        math.sin(x)
        * math.cos(y)
        * math.exp(math.fabs(1 - (((x ** 2 + y ** 2) ** 0.5) / math.pi)))
    )
)
BOUNDS = [(-10, 10), (-10, 10)]
DIMENSIONS = 2


def _iterations_to_threshold(history, threshold, fallback):
    for idx, value in enumerate(history, start=1):
        if value < threshold:
            return idx
    return fallback


def _run_variant(*, variant_name, use_constriction, seed, args):
    started = time.perf_counter()
    optimizer = PSO(
        func=FUNCTION,
        dimensions=DIMENSIONS,
        bounds=BOUNDS,
        num_particles=args.num_particles,
        max_iterations=args.max_iterations,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        seed=seed,
        use_constriction=use_constriction,
    )
    _, best_value = optimizer.optimize()
    elapsed = time.perf_counter() - started
    iterations_to_threshold = _iterations_to_threshold(
        optimizer.history,
        args.success_threshold,
        args.max_iterations,
    )
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
    base = summary_by_variant["pso_base"]
    constr = summary_by_variant["pso_constriction"]

    quality_winner = "pso_base" if base["mean_best_value"] < constr["mean_best_value"] else "pso_constriction"
    speed_winner = "pso_base" if base["mean_time_s"] < constr["mean_time_s"] else "pso_constriction"
    success_winner = "pso_base" if base["success_rate"] > constr["success_rate"] else "pso_constriction"

    print("\nСравнение победителей по ключевым метрикам:")
    print(f"  Качество (mean_best_value): {quality_winner}")
    print(f"  Скорость (mean_time_s):     {speed_winner}")
    print(f"  Успешность (success_rate):  {success_winner}")


def main():
    parser = argparse.ArgumentParser(
        description="Сравнение PSO без модификации и PSO с коэффициентом сжатия."
    )
    parser.add_argument("--runs", type=int, default=30, help="Число запусков на вариант")
    parser.add_argument("--success-threshold", type=float, default=-19.0, help="Порог успеха: f(x) < threshold")
    parser.add_argument("--seed-base", type=int, default=42, help="Базовый seed; на i-м запуске используется seed_base + i")
    parser.add_argument("--num-particles", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--w", type=float, default=0.7298, help="Инерционный вес для базового PSO")
    parser.add_argument("--c1", type=float, default=2.05)
    parser.add_argument("--c2", type=float, default=2.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Папка для CSV-отчётов",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs должен быть > 0")
    if args.num_particles <= 0:
        parser.error("--num-particles должен быть > 0")
    if args.max_iterations <= 0:
        parser.error("--max-iterations должен быть > 0")
    if args.c1 + args.c2 <= 4.0:
        parser.error("Для варианта с коэффициентом сжатия требуется c1 + c2 > 4")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs_rows = []
    by_variant = {"pso_base": [], "pso_constriction": []}

    for run_idx in range(args.runs):
        seed = args.seed_base + run_idx
        base_row = _run_variant(
            variant_name="pso_base",
            use_constriction=False,
            seed=seed,
            args=args,
        )
        constr_row = _run_variant(
            variant_name="pso_constriction",
            use_constriction=True,
            seed=seed,
            args=args,
        )
        for row in (base_row, constr_row):
            row["run_index"] = run_idx
            runs_rows.append(row)
            by_variant[row["variant"]].append(row)

    summary_rows = []
    summary_by_variant = {}
    for variant_name, rows in by_variant.items():
        summary = _summarize(rows)
        summary_by_variant[variant_name] = summary
        summary_rows.append({"variant": variant_name, **summary})

    runs_csv = args.output_dir / "pso_base_vs_constriction_runs.csv"
    summary_csv = args.output_dir / "pso_base_vs_constriction_summary.csv"

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
    print("Сравнение PSO: base vs constriction")
    print(f"Запусков на вариант: {args.runs}")
    print(f"Порог успеха: f < {args.success_threshold}")
    print(f"Параметры: c1={args.c1}, c2={args.c2}, w={args.w}")
    print(f"CSV (прогоны): {runs_csv}")
    print(f"CSV (сводка):  {summary_csv}")
    print("=" * 72)
    for row in summary_rows:
        print(
            f"{row['variant']:<18} | best_found={row['best_found_value']:.8f} | "
            f"mean_best={row['mean_best_value']:.8f} | mean_time={row['mean_time_s']:.4f}s | "
            f"mean_iter={row['mean_iterations_to_threshold']:.2f} | success_rate={row['success_rate']:.2%}"
        )

    _winner_line(summary_by_variant)


if __name__ == "__main__":
    main()
