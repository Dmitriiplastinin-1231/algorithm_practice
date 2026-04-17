"""
PSO Grid Search — эмпирический подбор гиперпараметров алгоритма роя частиц
для функции f(x,y) = -|sin(x)·cos(y)·exp(|1 − √(x²+y²)/π|)|.

Запуск:
    python parameter_tuning/pso_grid_search.py

Результаты сохраняются в parameter_tuning/pso_grid_search_results.csv
и выводятся в консоль в виде таблицы топ-10 комбинаций.

Примечание о коэффициенте сжатия χ:
    Алгоритм автоматически вычисляет χ из c₁ и c₂:
        φ = c₁ + c₂  (требуется φ > 4)
        χ = 2κ / |2 − φ − √(φ² − 4φ)|
    При c₁ = c₂ = 2.05 и κ = 1: φ = 4.10, χ ≈ 0.7298.
"""

import sys
import os
import math
import csv
import itertools
import time

# Добавляем корень репозитория в путь, чтобы импортировать pso
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pso import PSO

# ---------------------------------------------------------------------------
# Целевая функция (та же, что используется в gui.py)
# ---------------------------------------------------------------------------
FUNCTION = lambda x, y: -(math.fabs(
    math.sin(x) * math.cos(y) * math.exp(math.fabs(1 - (math.sqrt(x**2 + y**2) / math.pi)))
))

BOUNDS = [(-10, 10), (-10, 10)]

# Глобальный минимум функции в области [-10, 10]²:
# f ≈ -19.2085 в точках (±8.059, ±9.660)
KNOWN_GLOBAL_MIN = -19.2085

# ---------------------------------------------------------------------------
# Сетка параметров для перебора
# ---------------------------------------------------------------------------
# Для режима с коэффициентом сжатия (use_constriction=True) c₁ + c₂ должно быть > 4.
# Рассматриваем типичные диапазоны вокруг классического значения c₁=c₂=2.05.
PARAM_GRID = {
    "num_particles":  [20, 30, 40, 60],
    "max_iterations": [50, 100, 200],
    "c1":             [1.8, 2.0, 2.05, 2.2],
    "c2":             [1.8, 2.0, 2.05, 2.2],
    "kappa":          [0.5, 1.0],
}

# Количество повторных запусков на каждую комбинацию
N_REPEATS = 5

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pso_grid_search_results.csv")


def _chi_value(c1: float, c2: float, kappa: float) -> float:
    """Вычисляет χ по формуле сжатия Клерка–Кеннеди."""
    phi = c1 + c2
    if phi <= 4.0:
        return float("nan")
    return (2.0 * kappa) / abs(2.0 - phi - math.sqrt(phi ** 2 - 4.0 * phi))


def run_combination(params: dict, n_repeats: int = N_REPEATS) -> dict:
    """Запускает PSO n_repeats раз и возвращает агрегированные метрики."""
    phi = params["c1"] + params["c2"]
    if phi <= 4.0:
        # Пропускаем недопустимые комбинации (χ неопределён)
        return {
            **params,
            "chi": float("nan"),
            "avg_best": float("nan"),
            "min_best": float("nan"),
            "gap_to_opt": float("nan"),
            "avg_time_s": float("nan"),
            "valid": False,
        }

    chi = _chi_value(params["c1"], params["c2"], params["kappa"])
    best_values = []
    times = []

    for _ in range(n_repeats):
        t0 = time.perf_counter()
        optimizer = PSO(
            func=FUNCTION,
            dimensions=2,
            bounds=BOUNDS,
            num_particles=params["num_particles"],
            max_iterations=params["max_iterations"],
            use_constriction=True,
            c1=params["c1"],
            c2=params["c2"],
            kappa=params["kappa"],
        )
        _, best_val = optimizer.optimize(verbose=False)
        elapsed = time.perf_counter() - t0
        best_values.append(best_val)
        times.append(elapsed)

    avg_best = sum(best_values) / len(best_values)
    min_best = min(best_values)
    gap_to_opt = abs(avg_best - KNOWN_GLOBAL_MIN)

    return {
        **params,
        "chi": chi,
        "avg_best": avg_best,
        "min_best": min_best,
        "gap_to_opt": gap_to_opt,
        "avg_time_s": sum(times) / len(times),
        "valid": True,
    }


def main():
    keys = list(PARAM_GRID.keys())
    combinations = list(itertools.product(*[PARAM_GRID[k] for k in keys]))

    # Предварительная фильтрация: только комбинации с φ > 4
    valid_combos = [
        c for c in combinations
        if sum(dict(zip(keys, c))[k] for k in ("c1", "c2")) > 4.0
    ]

    total = len(valid_combos)
    skipped = len(combinations) - total
    print(f"Сетка параметров PSO: {total} допустимых комбинаций × {N_REPEATS} повторов = {total * N_REPEATS} запусков")
    print(f"  (пропущено {skipped} комбинаций с c₁+c₂ ≤ 4 — коэффициент сжатия неопределён)")
    print("=" * 80)

    fieldnames = keys + ["chi", "avg_best", "min_best", "gap_to_opt", "avg_time_s", "valid"]
    results = []

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, combo in enumerate(valid_combos, 1):
            params = dict(zip(keys, combo))
            chi = _chi_value(params["c1"], params["c2"], params["kappa"])
            print(f"[{i:>4d}/{total}] N={params['num_particles']:>3d}  "
                  f"T={params['max_iterations']:>4d}  "
                  f"c1={params['c1']:.2f}  c2={params['c2']:.2f}  "
                  f"κ={params['kappa']:.1f}  χ={chi:.4f}", end="  →  ", flush=True)

            row = run_combination(params)
            results.append(row)
            writer.writerow(row)
            csvfile.flush()

            if row["valid"]:
                print(f"avg_best={row['avg_best']:.6f}  gap={row['gap_to_opt']:.6f}  "
                      f"t={row['avg_time_s']:.3f}s")
            else:
                print("ПРОПУЩЕНО (φ ≤ 4)")

    # Фильтруем только допустимые результаты
    valid_results = [r for r in results if r.get("valid")]
    valid_results.sort(key=lambda r: r["gap_to_opt"])

    print("\n" + "=" * 80)
    print("ТОП-10 КОМБИНАЦИЙ (по близости к глобальному оптимуму):")
    print("=" * 80)
    header = (f"{'N':>4} {'T':>5} {'c1':>5} {'c2':>5} {'κ':>4} {'χ':>6} | "
              f"{'avg_best':>10} {'min_best':>10} {'gap':>10} {'t,s':>6}")
    print(header)
    print("-" * len(header))
    for row in valid_results[:10]:
        print(f"{row['num_particles']:>4d} "
              f"{row['max_iterations']:>5d} "
              f"{row['c1']:>5.2f} "
              f"{row['c2']:>5.2f} "
              f"{row['kappa']:>4.1f} "
              f"{row['chi']:>6.4f} | "
              f"{row['avg_best']:>10.6f} "
              f"{row['min_best']:>10.6f} "
              f"{row['gap_to_opt']:>10.6f} "
              f"{row['avg_time_s']:>6.3f}")

    print(f"\nПолные результаты сохранены в: {OUTPUT_CSV}")

    if valid_results:
        best = valid_results[0]
        print(f"\nЛучшая комбинация параметров PSO:")
        print(f"  N={best['num_particles']}, T={best['max_iterations']}, "
              f"c₁={best['c1']}, c₂={best['c2']}, κ={best['kappa']}")
        print(f"  χ = {best['chi']:.6f}")
        print(f"  Среднее лучшее значение: {best['avg_best']:.6f}  "
              f"(отклонение от оптимума: {best['gap_to_opt']:.6f})")

    # ---------------------------------------------------------------------------
    # Анализ влияния χ на качество результата
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("АНАЛИЗ ВЛИЯНИЯ χ НА КАЧЕСТВО:")
    print("=" * 80)
    buckets = {}
    for r in valid_results:
        chi_rounded = round(r["chi"], 2)
        if chi_rounded not in buckets:
            buckets[chi_rounded] = []
        buckets[chi_rounded].append(r["gap_to_opt"])
    for chi_val in sorted(buckets.keys()):
        vals = buckets[chi_val]
        avg_gap = sum(vals) / len(vals)
        print(f"  χ ≈ {chi_val:.2f}: среднее отклонение от оптимума = {avg_gap:.6f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
