"""
Подбор лучших параметров алгоритма роя частиц (PSO) методом сеточного поиска.

Перебираются комбинации:
  - num_particles
  - max_iterations
  - w  (инерционный вес, для стандартного режима)
  - c1 (когнитивный коэффициент)
  - c2 (социальный коэффициент)
  - use_constriction  (режим коэффициента сжатия)
  - kappa             (для режима сжатия)

Для каждой комбинации PSO запускается несколько раз (RUNS_PER_COMBO),
результат усредняется — это сглаживает влияние случайности.
"""

import itertools
import math
import time

from pso import PSO

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

BOUNDS = [(-10, 10), (-10, 10)]
DIMENSIONS = 2

# ---------------------------------------------------------------
#  Сетка параметров: стандартный PSO (инерционный вес)
# ---------------------------------------------------------------
STANDARD_GRID = {
    "num_particles":  [20, 40, 60],
    "max_iterations": [50, 100, 200],
    "w":              [0.4, 0.7298, 0.9],
    "c1":             [1.0, 1.49618, 2.0],
    "c2":             [1.0, 1.49618, 2.0],
}

# ---------------------------------------------------------------
#  Сетка параметров: PSO с коэффициентом сжатия (constriction)
# ---------------------------------------------------------------
CONSTRICTION_GRID = {
    "num_particles":  [20, 40, 60],
    "max_iterations": [50, 100, 200],
    "c1":             [2.05, 2.2, 2.5],
    "c2":             [2.05, 2.2, 2.5],
    "kappa":          [0.5, 0.75, 1.0],
}

# Сколько раз запускать PSO на каждой комбинации параметров
RUNS_PER_COMBO = 3

# ---------------------------------------------------------------
#  Вспомогательные функции
# ---------------------------------------------------------------

def _grid_combinations(grid: dict) -> list[dict]:
    """Возвращает список словарей — все комбинации значений из сетки."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def evaluate_standard(params: dict) -> dict:
    """Запускает стандартный PSO *RUNS_PER_COMBO* раз."""
    results = []
    for _ in range(RUNS_PER_COMBO):
        optimizer = PSO(
            func=FUNCTION,
            dimensions=DIMENSIONS,
            bounds=BOUNDS,
            num_particles=params["num_particles"],
            max_iterations=params["max_iterations"],
            w=params["w"],
            c1=params["c1"],
            c2=params["c2"],
            use_constriction=False,
        )
        _, best_val = optimizer.optimize(verbose=False)
        results.append(best_val)

    return {
        "params": params,
        "best_of_runs": min(results),
        "mean_of_runs": sum(results) / len(results),
        "all_runs": results,
    }


def evaluate_constriction(params: dict) -> dict:
    """Запускает PSO с коэффициентом сжатия *RUNS_PER_COMBO* раз."""
    results = []
    for _ in range(RUNS_PER_COMBO):
        optimizer = PSO(
            func=FUNCTION,
            dimensions=DIMENSIONS,
            bounds=BOUNDS,
            num_particles=params["num_particles"],
            max_iterations=params["max_iterations"],
            c1=params["c1"],
            c2=params["c2"],
            use_constriction=True,
            kappa=params["kappa"],
        )
        _, best_val = optimizer.optimize(verbose=False)
        results.append(best_val)

    return {
        "params": params,
        "best_of_runs": min(results),
        "mean_of_runs": sum(results) / len(results),
        "all_runs": results,
    }


def run_grid(name: str, combos: list[dict], evaluate_fn) -> list[dict]:
    """Общая логика перебора сетки и вывода результатов."""
    total = len(combos)
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"  Комбинаций: {total}, запусков на каждую: {RUNS_PER_COMBO}")
    print(f"{'=' * 70}")

    all_results = []
    start = time.time()

    for idx, params in enumerate(combos, 1):
        t0 = time.time()
        result = evaluate_fn(params)
        dt = time.time() - t0
        all_results.append(result)

        parts = ", ".join(f"{k}={v}" for k, v in params.items())
        print(
            f"  [{idx}/{total}]  {parts}  →  "
            f"лучшее={result['best_of_runs']:.8f}  "
            f"среднее={result['mean_of_runs']:.8f}  "
            f"({dt:.1f} с)"
        )

    elapsed = time.time() - start

    # Сортировка: лучшая средняя → первая
    all_results.sort(key=lambda r: r["mean_of_runs"])

    print(f"\n  ТОП-10 ({name}):")
    for rank, r in enumerate(all_results[:10], 1):
        parts = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(
            f"    #{rank:>2}  {parts}  |  "
            f"лучшее={r['best_of_runs']:.8f}  "
            f"среднее={r['mean_of_runs']:.8f}"
        )

    best = all_results[0]
    print(f"\n  ЛУЧШИЕ ПАРАМЕТРЫ ({name}):")
    for k, v in best["params"].items():
        print(f"    {k:>20s} = {v}")
    print(f"    {'лучшее значение':>20s} = {best['best_of_runs']:.10f}")
    print(f"    {'среднее значение':>20s} = {best['mean_of_runs']:.10f}")
    print(f"  Время: {elapsed:.1f} с")

    return all_results


# ---------------------------------------------------------------
#  Основной блок
# ---------------------------------------------------------------

def main():
    print("=" * 70)
    print("  ПОДБОР ПАРАМЕТРОВ АЛГОРИТМА РОЯ ЧАСТИЦ (PSO)")
    print("=" * 70)

    # 1. Стандартный PSO (инерционный вес)
    std_combos = _grid_combinations(STANDARD_GRID)
    std_results = run_grid(
        "Стандартный PSO (инерционный вес)",
        std_combos,
        evaluate_standard,
    )

    # 2. PSO с коэффициентом сжатия
    con_combos = _grid_combinations(CONSTRICTION_GRID)
    con_results = run_grid(
        "PSO с коэффициентом сжатия (constriction)",
        con_combos,
        evaluate_constriction,
    )

    # Общее сравнение лучших из двух режимов
    best_std = std_results[0]
    best_con = con_results[0]

    print("\n" + "=" * 70)
    print("  ИТОГОВОЕ СРАВНЕНИЕ ЛУЧШИХ КОНФИГУРАЦИЙ")
    print("=" * 70)

    print("\n  Стандартный PSO:")
    for k, v in best_std["params"].items():
        print(f"    {k:>20s} = {v}")
    print(f"    {'среднее значение':>20s} = {best_std['mean_of_runs']:.10f}")

    print("\n  PSO (constriction):")
    for k, v in best_con["params"].items():
        print(f"    {k:>20s} = {v}")
    print(f"    {'среднее значение':>20s} = {best_con['mean_of_runs']:.10f}")

    if best_std["mean_of_runs"] <= best_con["mean_of_runs"]:
        print("\n  >>> Лучший режим: Стандартный PSO")
    else:
        print("\n  >>> Лучший режим: PSO с коэффициентом сжатия")


if __name__ == "__main__":
    main()
