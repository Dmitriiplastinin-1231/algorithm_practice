import csv
import math
import time

import numpy as np
import matplotlib.pyplot as plt

from pso import PSO

# ========================================================
#  Целевая функция (та же, что в roevoy.py)
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
MAX_ITERATIONS = 100
C1 = 2.05
C2 = 2.05
USE_CONSTRICTION = True

# ========================================================
#  Параметры эксперимента
# ========================================================
SWARM_SIZES = list(range(5, 101, 5))   # 5, 10, 15, ..., 100
RUNS_PER_SIZE = 50
SUCCESS_THRESHOLD = -19.0              # глобальный минимум ≈ -19.2085

CSV_FILE = "swarm_size_experiment.csv"
PNG_FILE = "swarm_size_experiment.png"


# ========================================================
#  Вспомогательная функция: итерация первого успеха
# ========================================================
def _convergence_iteration(history: list, threshold: float) -> int:
    """Возвращает номер итерации (1-based), когда лучшее значение впервые
    опустилось ниже порога. Если этого не произошло — возвращает len(history)."""
    for i, val in enumerate(history, start=1):
        if val < threshold:
            return i
    return len(history)


# ========================================================
#  Основной эксперимент
# ========================================================
def run_experiment() -> list:
    """
    Перебирает размеры роя, для каждого выполняет RUNS_PER_SIZE независимых
    запусков PSO и собирает статистику.

    Возвращает список словарей с полями:
        swarm_size, mean_best, best_of_best, success_rate,
        mean_conv_iter, mean_time_s
    """
    results = []

    for n in SWARM_SIZES:
        best_values = []
        conv_iters = []
        times = []
        successes = 0

        for run in range(RUNS_PER_SIZE):
            optimizer = PSO(
                func=FUNCTION,
                dimensions=DIMENSIONS,
                bounds=BOUNDS,
                num_particles=n,
                max_iterations=MAX_ITERATIONS,
                use_constriction=USE_CONSTRICTION,
                c1=C1,
                c2=C2,
            )

            t_start = time.perf_counter()
            _, best_val = optimizer.optimize()
            t_end = time.perf_counter()

            best_values.append(best_val)
            times.append(t_end - t_start)

            conv_iter = _convergence_iteration(optimizer.history, SUCCESS_THRESHOLD)
            conv_iters.append(conv_iter)

            if best_val < SUCCESS_THRESHOLD:
                successes += 1

        mean_best = float(np.mean(best_values))
        best_of_best = float(np.min(best_values))
        success_rate = successes / RUNS_PER_SIZE
        mean_conv_iter = float(np.mean(conv_iters))
        mean_time_s = float(np.mean(times))

        results.append(
            {
                "swarm_size": n,
                "mean_best": mean_best,
                "best_of_best": best_of_best,
                "success_rate": success_rate,
                "mean_conv_iter": mean_conv_iter,
                "mean_time_s": mean_time_s,
            }
        )

        print(
            f"N = {n:3d}  |  среднее лучшее = {mean_best:10.6f}  |  "
            f"лучший результат = {best_of_best:10.6f}  |  "
            f"успех = {success_rate:.2f}  |  "
            f"сред. итерация сходимости = {mean_conv_iter:5.1f}  |  "
            f"время = {mean_time_s*1000:.1f} мс"
        )

    return results


# ========================================================
#  Сохранение CSV
# ========================================================
def save_csv(results: list) -> None:
    fieldnames = [
        "swarm_size",
        "mean_best",
        "best_of_best",
        "success_rate",
        "mean_conv_iter",
        "mean_time_s",
    ]
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nРезультаты сохранены в {CSV_FILE}")


# ========================================================
#  Визуализация
# ========================================================
def plot_results(results: list) -> None:
    sizes = [r["swarm_size"] for r in results]
    mean_bests = [r["mean_best"] for r in results]
    best_of_bests = [r["best_of_best"] for r in results]
    success_rates = [r["success_rate"] for r in results]
    conv_iters = [r["mean_conv_iter"] for r in results]
    times_ms = [r["mean_time_s"] * 1000 for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Влияние размера роя на сходимость PSO\n"
        f"(фиксированные параметры: max_iter={MAX_ITERATIONS}, "
        f"c1=c2={C1}, коэф. сжатия, {RUNS_PER_SIZE} запусков/точка)",
        fontsize=13,
    )

    # --- Панель 1: среднее и лучшее найденное значение ---
    ax = axes[0, 0]
    ax.plot(sizes, mean_bests, marker="o", linewidth=2, markersize=6,
            color="steelblue", label="Среднее лучшее")
    ax.plot(sizes, best_of_bests, marker="s", linewidth=2, markersize=6,
            color="darkorange", linestyle="--", label="Лучший из всех запусков")
    ax.axhline(y=SUCCESS_THRESHOLD, color="gray", linestyle=":", linewidth=1,
               label=f"Порог успеха ({SUCCESS_THRESHOLD})")
    ax.set_xlabel("Размер роя (число частиц)", fontsize=11)
    ax.set_ylabel("Значение функции (меньше = лучше)", fontsize=11)
    ax.set_title("Качество найденного решения", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Панель 2: доля успешных запусков ---
    ax = axes[0, 1]
    best_idx = np.argmax(success_rates)
    ax.bar(sizes, success_rates, width=3.5, color="steelblue", alpha=0.8)
    ax.axvline(x=sizes[best_idx], color="tomato", linestyle="--", linewidth=1.5,
               label=f"Лучший размер роя: {sizes[best_idx]}")
    ax.set_xlabel("Размер роя (число частиц)", fontsize=11)
    ax.set_ylabel("Доля успешных запусков", fontsize=11)
    ax.set_title(
        f"Доля успешных запусков (f < {SUCCESS_THRESHOLD})", fontsize=12
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # --- Панель 3: средняя итерация сходимости ---
    ax = axes[1, 0]
    ax.plot(sizes, conv_iters, marker="^", linewidth=2, markersize=6,
            color="seagreen")
    ax.set_xlabel("Размер роя (число частиц)", fontsize=11)
    ax.set_ylabel("Средняя итерация первого успеха", fontsize=11)
    ax.set_title(
        f"Скорость сходимости (итерация, когда f < {SUCCESS_THRESHOLD})",
        fontsize=12,
    )
    ax.set_ylim(0, MAX_ITERATIONS + 5)
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Панель 4: среднее время одного запуска ---
    ax = axes[1, 1]
    ax.plot(sizes, times_ms, marker="D", linewidth=2, markersize=6,
            color="mediumpurple")
    ax.set_xlabel("Размер роя (число частиц)", fontsize=11)
    ax.set_ylabel("Среднее время запуска (мс)", fontsize=11)
    ax.set_title("Вычислительные затраты", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    for ax in axes.flat:
        ax.set_xticks(sizes)
        ax.set_xticklabels(sizes, rotation=45, fontsize=8)

    plt.tight_layout()
    plt.savefig(PNG_FILE, dpi=150)
    print(f"График сохранён в {PNG_FILE}")
    plt.show()


# ========================================================
#  Точка входа
# ========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  Эксперимент: влияние размера роя (числа частиц) на PSO")
    print("  Фиксированные параметры:")
    print(f"    max_iterations   = {MAX_ITERATIONS}")
    print(f"    c1               = {C1}")
    print(f"    c2               = {C2}")
    print(f"    use_constriction = {USE_CONSTRICTION}")
    print(f"    bounds           = {BOUNDS}")
    print(f"    запусков / точка = {RUNS_PER_SIZE}")
    print(f"    порог успеха     = f < {SUCCESS_THRESHOLD}")
    print(f"    диапазон роя     = {SWARM_SIZES[0]} – {SWARM_SIZES[-1]}, шаг 5")
    print("=" * 70)

    results = run_experiment()
    save_csv(results)
    plot_results(results)
