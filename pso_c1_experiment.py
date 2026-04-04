"""
Эксперимент PSO: влияние когнитивного коэффициента c1.

Все параметры зафиксированы, кроме c1:
  c2               = 2.05  (социальный коэффициент)
  use_constriction = True   (χ вычисляется из φ = c1 + c2)
  num_particles    = 30
  max_iterations   = 100
  dimensions       = 2
  bounds           = [(-10, 10), (-10, 10)]
  runs_per_value   = 50

Диапазон c1: от 1.96 до 3.50 с шагом 0.1
  (требование: c1 + c2 > 4, т.е. c1 > 1.95)

Результаты: pso_c1_experiment.csv + pso_c1_experiment.png
"""

import csv
import math
import time

import numpy as np
import matplotlib.pyplot as plt

from pso import PSO

# ========================================================
#  Целевая функция (Holder Table)
# ========================================================
FUNCTION = lambda x, y: -(
    math.fabs(
        math.sin(x)
        * math.cos(y)
        * math.exp(math.fabs(1 - (((x ** 2 + y ** 2) ** 0.5) / math.pi)))
    )
)

BOUNDS = [(-10, 10), (-10, 10)]
DIMENSIONS = 2

# ========================================================
#  Фиксированные параметры PSO
# ========================================================
C2 = 2.05
NUM_PARTICLES = 30
MAX_ITERATIONS = 100
USE_CONSTRICTION = True

# ========================================================
#  Параметры эксперимента
# ========================================================
# c1 > 4 - c2 = 4 - 2.05 = 1.95, поэтому начинаем с 1.96
C1_VALUES = [round(v, 2) for v in np.arange(1.96, 3.51, 0.1)]
RUNS_PER_VALUE = 50
SUCCESS_THRESHOLD = -19.0   # глобальный минимум ≈ -19.2085

CSV_FILE = "pso_c1_experiment.csv"
PNG_FILE = "pso_c1_experiment.png"


# ========================================================
#  Вспомогательные функции
# ========================================================
def _convergence_iteration(history: list, threshold: float) -> int:
    """Номер итерации (1-based) первого пересечения порога; иначе len(history)."""
    for i, val in enumerate(history, start=1):
        if val < threshold:
            return i
    return len(history)


# ========================================================
#  Основной эксперимент
# ========================================================
def run_experiment() -> list:
    results = []

    for c1 in C1_VALUES:
        phi = c1 + C2
        # χ по формуле Clerc & Kennedy (kappa=1)
        chi = (2.0) / abs(2.0 - phi - math.sqrt(phi ** 2 - 4.0 * phi))

        best_values = []
        conv_iters = []
        times = []
        successes = 0

        for _ in range(RUNS_PER_VALUE):
            optimizer = PSO(
                func=FUNCTION,
                dimensions=DIMENSIONS,
                bounds=BOUNDS,
                num_particles=NUM_PARTICLES,
                max_iterations=MAX_ITERATIONS,
                use_constriction=USE_CONSTRICTION,
                c1=c1,
                c2=C2,
            )

            t_start = time.perf_counter()
            _, best_val = optimizer.optimize()
            t_end = time.perf_counter()

            best_values.append(best_val)
            times.append(t_end - t_start)
            conv_iters.append(_convergence_iteration(optimizer.history, SUCCESS_THRESHOLD))
            if best_val < SUCCESS_THRESHOLD:
                successes += 1

        mean_best = float(np.mean(best_values))
        best_of_best = float(np.min(best_values))
        success_rate = successes / RUNS_PER_VALUE
        mean_conv_iter = float(np.mean(conv_iters))
        mean_time_s = float(np.mean(times))

        results.append(
            {
                "c1": c1,
                "chi": round(chi, 6),
                "mean_best": mean_best,
                "best_of_best": best_of_best,
                "success_rate": success_rate,
                "mean_conv_iter": mean_conv_iter,
                "mean_time_s": mean_time_s,
            }
        )

        print(
            f"c1 = {c1:.2f}  χ = {chi:.4f}  |  среднее = {mean_best:10.6f}  |  "
            f"лучшее = {best_of_best:10.6f}  |  успех = {success_rate:.2f}  |  "
            f"сход. итер. = {mean_conv_iter:5.1f}  |  время = {mean_time_s*1000:.1f} мс"
        )

    return results


# ========================================================
#  Сохранение CSV
# ========================================================
def save_csv(results: list) -> None:
    fieldnames = ["c1", "chi", "mean_best", "best_of_best",
                  "success_rate", "mean_conv_iter", "mean_time_s"]
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nРезультаты сохранены в {CSV_FILE}")


# ========================================================
#  Визуализация
# ========================================================
def plot_results(results: list) -> None:
    c1_vals = [r["c1"] for r in results]
    mean_bests = [r["mean_best"] for r in results]
    best_of_bests = [r["best_of_best"] for r in results]
    success_rates = [r["success_rate"] for r in results]
    conv_iters = [r["mean_conv_iter"] for r in results]
    times_ms = [r["mean_time_s"] * 1000 for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Влияние когнитивного коэффициента c1 на PSO\n"
        f"(фиксированные: c2={C2}, num_particles={NUM_PARTICLES}, "
        f"max_iter={MAX_ITERATIONS}, constriction, {RUNS_PER_VALUE} запусков/точка)",
        fontsize=13,
    )

    # --- Панель 1: среднее и лучшее найденное значение ---
    ax = axes[0, 0]
    ax.plot(c1_vals, mean_bests, marker="o", linewidth=2, markersize=6,
            color="steelblue", label="Среднее лучшее")
    ax.plot(c1_vals, best_of_bests, marker="s", linewidth=2, markersize=6,
            color="darkorange", linestyle="--", label="Лучший из запусков")
    ax.axhline(y=SUCCESS_THRESHOLD, color="gray", linestyle=":", linewidth=1,
               label=f"Порог успеха ({SUCCESS_THRESHOLD})")
    ax.set_xlabel("Когнитивный коэффициент c1", fontsize=11)
    ax.set_ylabel("Значение функции (меньше = лучше)", fontsize=11)
    ax.set_title("Качество найденного решения", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Панель 2: доля успешных запусков ---
    ax = axes[0, 1]
    best_idx = int(np.argmax(success_rates))
    ax.bar([str(v) for v in c1_vals], success_rates, color="steelblue", alpha=0.8)
    ax.axvline(x=str(c1_vals[best_idx]), color="tomato", linestyle="--", linewidth=1.5,
               label=f"Лучший c1: {c1_vals[best_idx]}")
    ax.set_xlabel("Когнитивный коэффициент c1", fontsize=11)
    ax.set_ylabel("Доля успешных запусков", fontsize=11)
    ax.set_title(f"Доля успешных запусков (f < {SUCCESS_THRESHOLD})", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.tick_params(axis="x", rotation=45)

    # --- Панель 3: средняя итерация сходимости ---
    ax = axes[1, 0]
    ax.plot(c1_vals, conv_iters, marker="^", linewidth=2, markersize=6,
            color="seagreen")
    ax.set_xlabel("Когнитивный коэффициент c1", fontsize=11)
    ax.set_ylabel("Средняя итерация первого успеха", fontsize=11)
    ax.set_title(
        f"Скорость сходимости (итерация, когда f < {SUCCESS_THRESHOLD})", fontsize=12
    )
    ax.set_ylim(0, MAX_ITERATIONS + 5)
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Панель 4: среднее время одного запуска ---
    ax = axes[1, 1]
    ax.plot(c1_vals, times_ms, marker="D", linewidth=2, markersize=6,
            color="mediumpurple")
    ax.set_xlabel("Когнитивный коэффициент c1", fontsize=11)
    ax.set_ylabel("Среднее время запуска (мс)", fontsize=11)
    ax.set_title("Вычислительные затраты", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PNG_FILE, dpi=150)
    print(f"График сохранён в {PNG_FILE}")
    plt.show()


# ========================================================
#  Точка входа
# ========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  Эксперимент: влияние когнитивного коэффициента c1 на PSO")
    print("  Фиксированные параметры:")
    print(f"    c2               = {C2}")
    print(f"    num_particles    = {NUM_PARTICLES}")
    print(f"    max_iterations   = {MAX_ITERATIONS}")
    print(f"    use_constriction = {USE_CONSTRICTION}")
    print(f"    bounds           = {BOUNDS}")
    print(f"    запусков / точка = {RUNS_PER_VALUE}")
    print(f"    порог успеха     = f < {SUCCESS_THRESHOLD}")
    print(f"    диапазон c1      = {C1_VALUES[0]} – {C1_VALUES[-1]}")
    print("=" * 70)

    results = run_experiment()
    save_csv(results)
    plot_results(results)
