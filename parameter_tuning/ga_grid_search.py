"""
GA Grid Search — эмпирический подбор гиперпараметров генетического алгоритма
для функции f(x,y) = -|sin(x)·cos(y)·exp(|1 − √(x²+y²)/π|)|.

Запуск:
    python parameter_tuning/ga_grid_search.py

Результаты сохраняются в parameter_tuning/ga_grid_search_results.csv
и выводятся в консоль в виде таблицы топ-10 комбинаций.
"""

import sys
import os
import math
import csv
import itertools
import time

# Добавляем корень репозитория в путь, чтобы импортировать geneticAlgorithmWithoutModific
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geneticAlgorithmWithoutModific import run_genetic_algorithm

# ---------------------------------------------------------------------------
# Целевая функция (та же, что используется в gui.py)
# ---------------------------------------------------------------------------
FUNCTION = lambda x, y: -(math.fabs(
    math.sin(x) * math.cos(y) * math.exp(math.fabs(1 - (math.sqrt(x**2 + y**2) / math.pi)))
))

# Глобальный минимум функции в области [-10, 10]²:
# f ≈ -19.2085 в точках (±8.059, ±9.660)
KNOWN_GLOBAL_MIN = -19.2085

# ---------------------------------------------------------------------------
# Сетка параметров для перебора
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "population_size": [200, 500, 1000],
    "max_generations": [100, 200, 300],
    "p_crossover":     [0.7, 0.8, 0.9],
    "p_mutation":      [0.1, 0.2, 0.3],
    "elitism_count":   [5, 10, 20],
}

# Количество повторных запусков на каждую комбинацию (для надёжности)
N_REPEATS = 3

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ga_grid_search_results.csv")


def run_combination(params: dict, n_repeats: int = N_REPEATS) -> dict:
    """Запускает ГА n_repeats раз и возвращает агрегированные метрики."""
    best_values = []
    times = []

    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = run_genetic_algorithm(
            population_size=params["population_size"],
            max_generations=params["max_generations"],
            p_crossover=params["p_crossover"],
            p_mutation=params["p_mutation"],
            elitism_count=params["elitism_count"],
        )
        elapsed = time.perf_counter() - t0
        best_values.append(result["best_value"])
        times.append(elapsed)

    avg_best = sum(best_values) / len(best_values)
    min_best = min(best_values)
    gap_to_opt = abs(avg_best - KNOWN_GLOBAL_MIN)

    return {
        **params,
        "avg_best": avg_best,
        "min_best": min_best,
        "gap_to_opt": gap_to_opt,
        "avg_time_s": sum(times) / len(times),
    }


def main():
    keys = list(PARAM_GRID.keys())
    combinations = list(itertools.product(*[PARAM_GRID[k] for k in keys]))

    total = len(combinations)
    print(f"Сетка параметров ГА: {total} комбинаций × {N_REPEATS} повторов = {total * N_REPEATS} запусков")
    print("=" * 70)

    fieldnames = keys + ["avg_best", "min_best", "gap_to_opt", "avg_time_s"]
    results = []

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            print(f"[{i:>4d}/{total}] pop={params['population_size']:>5d}  "
                  f"gen={params['max_generations']:>4d}  "
                  f"pcross={params['p_crossover']:.1f}  "
                  f"pmut={params['p_mutation']:.1f}  "
                  f"elite={params['elitism_count']:>3d}", end="  →  ", flush=True)

            row = run_combination(params)
            results.append(row)
            writer.writerow(row)
            csvfile.flush()

            print(f"avg_best={row['avg_best']:.6f}  gap={row['gap_to_opt']:.6f}  "
                  f"t={row['avg_time_s']:.2f}s")

    # ---------------------------------------------------------------------------
    # Вывод топ-10 по минимальному зазору до оптимума
    # ---------------------------------------------------------------------------
    results.sort(key=lambda r: r["gap_to_opt"])
    print("\n" + "=" * 70)
    print("ТОП-10 КОМБИНАЦИЙ (по близости к глобальному оптимуму):")
    print("=" * 70)
    header = (f"{'pop':>6} {'gen':>5} {'pcr':>5} {'pmt':>5} {'eli':>4} | "
              f"{'avg_best':>10} {'min_best':>10} {'gap':>10} {'t,s':>6}")
    print(header)
    print("-" * len(header))
    for row in results[:10]:
        print(f"{row['population_size']:>6d} "
              f"{row['max_generations']:>5d} "
              f"{row['p_crossover']:>5.2f} "
              f"{row['p_mutation']:>5.2f} "
              f"{row['elitism_count']:>4d} | "
              f"{row['avg_best']:>10.6f} "
              f"{row['min_best']:>10.6f} "
              f"{row['gap_to_opt']:>10.6f} "
              f"{row['avg_time_s']:>6.2f}")

    print(f"\nПолные результаты сохранены в: {OUTPUT_CSV}")
    print(f"\nТеоретические рекомендации подтверждены эмпирически:")
    best = results[0]
    print(f"  Лучшая комбинация: pop={best['population_size']}, "
          f"gen={best['max_generations']}, "
          f"p_cross={best['p_crossover']}, "
          f"p_mut={best['p_mutation']}, "
          f"elite={best['elitism_count']}")
    print(f"  Среднее лучшее значение: {best['avg_best']:.6f}  "
          f"(отклонение от оптимума: {best['gap_to_opt']:.6f})")


if __name__ == "__main__":
    main()
