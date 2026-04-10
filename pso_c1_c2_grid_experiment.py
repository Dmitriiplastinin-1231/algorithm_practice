"""
Эксперимент PSO: перебор всех комбинаций c1 и c2.

Программа перебирает все пары (c1, c2) из заданного диапазона
с помощью двух вложенных циклов:

    for c1 in C1_VALUES:
        for c2 in C2_VALUES:
            ...

Для каждой допустимой пары (c1 + c2 > 4 — требование коэффициента
сжатия Clerc & Kennedy) запускается краткая оптимизация PSO и
сохраняются метрики (среднее лучшее значение, доля успехов, скорость
сходимости).

Параметры диапазона задаются константами в начале файла:
  C1_START, C1_STOP, C1_STEP — диапазон когнитивного коэффициента
  C2_START, C2_STOP, C2_STEP — диапазон социального коэффициента

Результаты: pso_c1_c2_grid.csv + pso_c1_c2_grid.png
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
#  Параметры диапазона (задаются здесь)
# ========================================================
C1_START = 0.5
C1_STOP  = 3.0
C1_STEP  = 0.1

C2_START = 0.5
C2_STOP  = 3.0
C2_STEP  = 0.1

# ========================================================
#  Фиксированные параметры PSO
# ========================================================
NUM_PARTICLES    = 30
MAX_ITERATIONS   = 100
USE_CONSTRICTION = True

# ========================================================
#  Параметры эксперимента
# ========================================================
RUNS_PER_PAIR    = 20          # запусков на каждую пару (c1, c2)
SUCCESS_THRESHOLD = -19.0      # глобальный минимум ≈ -19.2085

CSV_FILE = "pso_c1_c2_grid.csv"
PNG_FILE = "pso_c1_c2_grid.png"


# ========================================================
#  Вспомогательные функции
# ========================================================
def _make_range(start: float, stop: float, step: float) -> list:
    """Генерирует список значений от start до stop включительно с шагом step."""
    n = round((stop - start) / step)
    return [round(start + i * step, 10) for i in range(n + 1)]


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
    C1_VALUES = _make_range(C1_START, C1_STOP, C1_STEP)
    C2_VALUES = _make_range(C2_START, C2_STOP, C2_STEP)

    total = sum(1 for c1 in C1_VALUES for c2 in C2_VALUES if c1 + c2 > 4.0)
    done  = 0
    results = []

    for c1 in C1_VALUES:
        for c2 in C2_VALUES:
            phi = c1 + c2

            # Пропускаем пары, не удовлетворяющие условию коэффициента сжатия
            if phi <= 4.0:
                continue

            chi = 2.0 / abs(2.0 - phi - math.sqrt(phi ** 2 - 4.0 * phi))

            best_values = []
            conv_iters  = []
            times       = []
            successes   = 0

            for _ in range(RUNS_PER_PAIR):
                optimizer = PSO(
                    func=FUNCTION,
                    dimensions=DIMENSIONS,
                    bounds=BOUNDS,
                    num_particles=NUM_PARTICLES,
                    max_iterations=MAX_ITERATIONS,
                    use_constriction=USE_CONSTRICTION,
                    c1=c1,
                    c2=c2,
                )

                t_start = time.perf_counter()
                _, best_val = optimizer.optimize()
                t_end = time.perf_counter()

                best_values.append(best_val)
                times.append(t_end - t_start)
                conv_iters.append(
                    _convergence_iteration(optimizer.history, SUCCESS_THRESHOLD)
                )
                if best_val < SUCCESS_THRESHOLD:
                    successes += 1

            mean_best      = float(np.mean(best_values))
            best_of_best   = float(np.min(best_values))
            success_rate   = successes / RUNS_PER_PAIR
            mean_conv_iter = float(np.mean(conv_iters))
            mean_time_s    = float(np.mean(times))

            results.append(
                {
                    "c1":            round(c1, 2),
                    "c2":            round(c2, 2),
                    "phi":           round(phi, 4),
                    "chi":           round(chi, 6),
                    "mean_best":     mean_best,
                    "best_of_best":  best_of_best,
                    "success_rate":  success_rate,
                    "mean_conv_iter": mean_conv_iter,
                    "mean_time_s":   mean_time_s,
                }
            )

            done += 1
            print(
                f"[{done:3d}/{total}] c1={c1:.2f}  c2={c2:.2f}  χ={chi:.4f}  |  "
                f"среднее={mean_best:10.6f}  |  успех={success_rate:.2f}  |  "
                f"сход.={mean_conv_iter:5.1f}"
            )

    return results


# ========================================================
#  Сохранение CSV
# ========================================================
def save_csv(results: list) -> None:
    fieldnames = [
        "c1", "c2", "phi", "chi",
        "mean_best", "best_of_best",
        "success_rate", "mean_conv_iter", "mean_time_s",
    ]
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nРезультаты сохранены в {CSV_FILE}")


# ========================================================
#  Визуализация (тепловые карты)
# ========================================================
def plot_results(results: list) -> None:
    c1_vals = sorted(set(r["c1"] for r in results))
    c2_vals = sorted(set(r["c2"] for r in results))

    # Строим матрицы значений; ячейки без данных — NaN
    nan = float("nan")
    idx_c1 = {v: i for i, v in enumerate(c1_vals)}
    idx_c2 = {v: i for i, v in enumerate(c2_vals)}

    def make_matrix(key: str):
        mat = np.full((len(c1_vals), len(c2_vals)), nan)
        for r in results:
            mat[idx_c1[r["c1"]], idx_c2[r["c2"]]] = r[key]
        return mat

    mat_mean    = make_matrix("mean_best")
    mat_success = make_matrix("success_rate")
    mat_conv    = make_matrix("mean_conv_iter")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Перебор всех комбинаций (c1, c2) — PSO, Holder Table\n"
        f"(num_particles={NUM_PARTICLES}, max_iter={MAX_ITERATIONS}, "
        f"{RUNS_PER_PAIR} запусков/пара, constriction)",
        fontsize=13,
    )

    c1_labels = [f"{v:.1f}" for v in c1_vals]
    c2_labels = [f"{v:.1f}" for v in c2_vals]

    # Панель 1: среднее лучшее значение
    ax = axes[0]
    im = ax.imshow(mat_mean, aspect="auto", origin="lower",
                   cmap="viridis_r",
                   extent=[-0.5, len(c2_vals) - 0.5, -0.5, len(c1_vals) - 0.5])
    plt.colorbar(im, ax=ax, label="Среднее лучшее значение")
    ax.set_xticks(range(len(c2_vals)))
    ax.set_xticklabels(c2_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(c1_vals)))
    ax.set_yticklabels(c1_labels, fontsize=7)
    ax.set_xlabel("c2", fontsize=11)
    ax.set_ylabel("c1", fontsize=11)
    ax.set_title("Среднее лучшее значение\n(чем меньше — тем лучше)", fontsize=11)

    # Панель 2: доля успешных запусков
    ax = axes[1]
    im = ax.imshow(mat_success, aspect="auto", origin="lower",
                   cmap="RdYlGn", vmin=0, vmax=1,
                   extent=[-0.5, len(c2_vals) - 0.5, -0.5, len(c1_vals) - 0.5])
    plt.colorbar(im, ax=ax, label="Доля успешных запусков")
    ax.set_xticks(range(len(c2_vals)))
    ax.set_xticklabels(c2_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(c1_vals)))
    ax.set_yticklabels(c1_labels, fontsize=7)
    ax.set_xlabel("c2", fontsize=11)
    ax.set_ylabel("c1", fontsize=11)
    ax.set_title(f"Доля успешных запусков\n(f < {SUCCESS_THRESHOLD})", fontsize=11)

    # Панель 3: средняя итерация сходимости
    ax = axes[2]
    im = ax.imshow(mat_conv, aspect="auto", origin="lower",
                   cmap="plasma_r",
                   extent=[-0.5, len(c2_vals) - 0.5, -0.5, len(c1_vals) - 0.5])
    plt.colorbar(im, ax=ax, label="Средняя итерация сходимости")
    ax.set_xticks(range(len(c2_vals)))
    ax.set_xticklabels(c2_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(c1_vals)))
    ax.set_yticklabels(c1_labels, fontsize=7)
    ax.set_xlabel("c2", fontsize=11)
    ax.set_ylabel("c1", fontsize=11)
    ax.set_title("Средняя итерация сходимости\n(чем меньше — тем быстрее)", fontsize=11)

    plt.tight_layout()
    plt.savefig(PNG_FILE, dpi=150)
    print(f"График сохранён в {PNG_FILE}")
    plt.show()


# ========================================================
#  Точка входа
# ========================================================
if __name__ == "__main__":
    C1_VALUES = _make_range(C1_START, C1_STOP, C1_STEP)
    C2_VALUES = _make_range(C2_START, C2_STOP, C2_STEP)
    valid_pairs = sum(1 for c1 in C1_VALUES for c2 in C2_VALUES if c1 + c2 > 4.0)

    print("=" * 70)
    print("  Эксперимент: перебор всех комбинаций (c1, c2) для PSO")
    print(f"  Диапазон c1: {C1_START} – {C1_STOP}, шаг {C1_STEP}")
    print(f"  Диапазон c2: {C2_START} – {C2_STOP}, шаг {C2_STEP}")
    print(f"  Допустимых пар (c1 + c2 > 4): {valid_pairs}")
    print(f"  Запусков на пару: {RUNS_PER_PAIR}")
    print(f"  num_particles    = {NUM_PARTICLES}")
    print(f"  max_iterations   = {MAX_ITERATIONS}")
    print(f"  use_constriction = {USE_CONSTRICTION}")
    print(f"  порог успеха     = f < {SUCCESS_THRESHOLD}")
    print("=" * 70)

    results = run_experiment()
    save_csv(results)
    plot_results(results)
