"""
Подбор лучших параметров генетического алгоритма методом сеточного поиска.

Перебираются комбинации:
  - population_size
  - p_crossover
  - p_mutation
  - max_generations
  - elitism_count

Для каждой комбинации алгоритм запускается несколько раз (RUNS_PER_COMBO),
результат усредняется — это сглаживает влияние случайности.
"""

import itertools
import math
import time

from geneticAlgorithmWithoutModific import run_genetic_algorithm

# ---------------------------------------------------------------
#  Целевая функция (та же, что используется в проекте)
# ---------------------------------------------------------------
FUNCTION = lambda x, y: -(
    math.fabs(
        math.sin(x)
        * math.cos(y)
        * math.exp(math.fabs(1 - ((x ** 2 + y ** 2) ** 0.5) / math.pi))
    )
)

BORDER = [[-10, 10], [-10, 10]]

# ---------------------------------------------------------------
#  Сетка параметров для перебора
# ---------------------------------------------------------------
PARAM_GRID = {
    "population_size": [200, 500, 1000],
    "p_crossover":     [0.7, 0.8, 0.9],
    "p_mutation":      [0.1, 0.2, 0.3, 0.5],
    "max_generations": [100, 200, 300],
    "elitism_count":   [5, 10, 20],
}

# Сколько раз запускать алгоритм на каждой комбинации параметров
RUNS_PER_COMBO = 3

# ---------------------------------------------------------------
#  Вспомогательные функции
# ---------------------------------------------------------------

def _grid_combinations(grid: dict) -> list[dict]:
    """Возвращает список словарей — все комбинации значений из сетки."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def evaluate_params(params: dict) -> dict:
    """Запускает ГА *RUNS_PER_COMBO* раз и возвращает агрегированную статистику."""
    results = []
    for _ in range(RUNS_PER_COMBO):
        res = run_genetic_algorithm(
            population_size=params["population_size"],
            p_crossover=params["p_crossover"],
            p_mutation=params["p_mutation"],
            max_generations=params["max_generations"],
            elitism_count=params["elitism_count"],
            border=BORDER,
            function=FUNCTION,
        )
        results.append(res["best_value"])

    return {
        "params": params,
        "best_of_runs": min(results),
        "mean_of_runs": sum(results) / len(results),
        "all_runs": results,
    }


# ---------------------------------------------------------------
#  Основной блок
# ---------------------------------------------------------------

def main():
    combos = _grid_combinations(PARAM_GRID)
    total = len(combos)
    print(f"Всего комбинаций параметров: {total}")
    print(f"Запусков на комбинацию:       {RUNS_PER_COMBO}")
    print(f"Общее число запусков:         {total * RUNS_PER_COMBO}")
    print("=" * 70)

    all_results = []
    start = time.time()

    for idx, params in enumerate(combos, 1):
        t0 = time.time()
        result = evaluate_params(params)
        dt = time.time() - t0
        all_results.append(result)

        print(
            f"[{idx}/{total}]  "
            f"pop={params['population_size']:>5}, "
            f"cx={params['p_crossover']:.2f}, "
            f"mut={params['p_mutation']:.2f}, "
            f"gen={params['max_generations']:>4}, "
            f"elite={params['elitism_count']:>3}  →  "
            f"лучшее={result['best_of_runs']:.8f}  "
            f"среднее={result['mean_of_runs']:.8f}  "
            f"({dt:.1f} с)"
        )

    elapsed = time.time() - start

    # Сортировка: лучшая средняя → первая
    all_results.sort(key=lambda r: r["mean_of_runs"])

    print("\n" + "=" * 70)
    print("  ТОП-10 ЛУЧШИХ КОМБИНАЦИЙ ПАРАМЕТРОВ (по среднему результату)")
    print("=" * 70)

    for rank, r in enumerate(all_results[:10], 1):
        p = r["params"]
        print(
            f"  #{rank:>2}  "
            f"pop={p['population_size']:>5}, "
            f"cx={p['p_crossover']:.2f}, "
            f"mut={p['p_mutation']:.2f}, "
            f"gen={p['max_generations']:>4}, "
            f"elite={p['elitism_count']:>3}  |  "
            f"лучшее={r['best_of_runs']:.8f}  "
            f"среднее={r['mean_of_runs']:.8f}"
        )

    best = all_results[0]
    print("\n" + "=" * 70)
    print("  ЛУЧШИЕ ПАРАМЕТРЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА:")
    print("=" * 70)
    for k, v in best["params"].items():
        print(f"    {k:>20s} = {v}")
    print(f"    {'лучшее значение':>20s} = {best['best_of_runs']:.10f}")
    print(f"    {'среднее значение':>20s} = {best['mean_of_runs']:.10f}")
    print(f"\n  Общее время: {elapsed:.1f} с")


if __name__ == "__main__":
    main()
