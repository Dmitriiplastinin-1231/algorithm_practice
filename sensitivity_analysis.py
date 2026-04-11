"""
Сравнительный анализ чувствительности параметров PSO и ГА.

PSO:
  Варьируются c1 и c2 от 0.5 до 3.0 с шагом 0.25.
  Все остальные параметры зафиксированы:
    num_particles  = 30
    max_iterations = 100
    w              = 0.7298  (инерционный вес; use_constriction=False)
    dimensions     = 2
    bounds         = [(-10, 10), (-10, 10)]
  На каждую комбинацию: RUNS_PER_COMBO независимых запусков.

ГА:
  Варьируются p_mutation (от 0.001 до 0.1) и p_crossover (от 0.5 до 0.99).
  Все остальные параметры зафиксированы:
    population_size = 500
    max_generations = 150
    elitism_count   = 20
  На каждую комбинацию: RUNS_PER_COMBO независимых запусков.

Метрики:
  success_rate    – доля запусков, в которых найдено f < SUCCESS_THRESHOLD
  mean_time_s     – среднее время одного запуска (секунды)
  mean_best       – среднее лучшее значение функции

Результаты:
  pso_sensitivity.csv  /  pso_sensitivity_success.png
                          pso_sensitivity_time.png
                          pso_sensitivity_best.png
  ga_sensitivity.csv   /  ga_sensitivity_success.png
                          ga_sensitivity_time.png
                          ga_sensitivity_best.png
"""

import csv
import math
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")          # позволяет запускать без GUI
import matplotlib.pyplot as plt

from pso import PSO
from geneticAlgorithmWithoutModific import (
    run_genetic_algorithm,
    BORDER,
)

# ============================================================
#  Общие константы
# ============================================================
RUNS_PER_COMBO = 20
SUCCESS_THRESHOLD = -19.0      # глобальный минимум функции Holder Table ≈ -19.2085

FUNCTION = lambda x, y: -(
    math.fabs(
        math.sin(x)
        * math.cos(y)
        * math.exp(math.fabs(1.0 - (((x ** 2 + y ** 2) ** 0.5) / math.pi)))
    )
)
BOUNDS = [(-10, 10), (-10, 10)]
DIMENSIONS = 2

# ============================================================
#  Параметры PSO
# ============================================================
PSO_NUM_PARTICLES = 30
PSO_MAX_ITERATIONS = 100
PSO_W = 0.7298                 # инерционный вес (use_constriction=False)

C1_VALUES = [round(v, 2) for v in np.arange(0.5, 3.01, 0.25)]
C2_VALUES = [round(v, 2) for v in np.arange(0.5, 3.01, 0.25)]

PSO_CSV = "pso_sensitivity.csv"
PSO_PNG_SUCCESS = "pso_sensitivity_success.png"
PSO_PNG_TIME    = "pso_sensitivity_time.png"
PSO_PNG_BEST    = "pso_sensitivity_best.png"

# ============================================================
#  Параметры ГА
# ============================================================
GA_POPULATION_SIZE = 500
GA_MAX_GENERATIONS = 150
GA_ELITISM_COUNT   = 20

PM_VALUES = [round(v, 3) for v in [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]]
PC_VALUES = [round(v, 2)  for v in np.arange(0.5, 1.00, 0.07)]

GA_CSV = "ga_sensitivity.csv"
GA_PNG_SUCCESS = "ga_sensitivity_success.png"
GA_PNG_TIME    = "ga_sensitivity_time.png"
GA_PNG_BEST    = "ga_sensitivity_best.png"

# Text-contrast thresholds for heatmap cell annotations
_TEXT_DARK_THRESHOLD  = 0.6   # normalised value above which white text is used
_TEXT_LIGHT_THRESHOLD = 0.3   # normalised value below which white text is used


# ============================================================
#  Эксперимент PSO
# ============================================================
def run_pso_experiment() -> list:
    """Перебирает все комбинации (c1, c2) и собирает статистику."""
    results = []
    total = len(C1_VALUES) * len(C2_VALUES)
    done = 0

    print("=" * 70)
    print(f"PSO: сетка c1 × c2 = {len(C1_VALUES)} × {len(C2_VALUES)} = {total} комбинаций")
    print(f"     {RUNS_PER_COMBO} запусков на комбинацию  →  {total * RUNS_PER_COMBO} запусков всего")
    print("=" * 70)

    for c1 in C1_VALUES:
        for c2 in C2_VALUES:
            best_values = []
            times = []
            successes = 0

            for _ in range(RUNS_PER_COMBO):
                optimizer = PSO(
                    func=FUNCTION,
                    dimensions=DIMENSIONS,
                    bounds=BOUNDS,
                    num_particles=PSO_NUM_PARTICLES,
                    max_iterations=PSO_MAX_ITERATIONS,
                    w=PSO_W,
                    c1=c1,
                    c2=c2,
                    use_constriction=False,
                )
                t0 = time.perf_counter()
                _, best_val = optimizer.optimize()
                t1 = time.perf_counter()

                best_values.append(best_val)
                times.append(t1 - t0)
                if best_val < SUCCESS_THRESHOLD:
                    successes += 1

            done += 1
            success_rate = successes / RUNS_PER_COMBO
            mean_time_s  = float(np.mean(times))
            mean_best    = float(np.mean(best_values))

            results.append({
                "c1": c1,
                "c2": c2,
                "success_rate": success_rate,
                "mean_time_s":  mean_time_s,
                "mean_best":    mean_best,
            })

            print(
                f"[{done:>4}/{total}] c1={c1:.2f} c2={c2:.2f}  |  "
                f"успех={success_rate:.2f}  среднее={mean_best:9.5f}  "
                f"время={mean_time_s*1000:.1f}мс"
            )

    return results


# ============================================================
#  Сохранение CSV (PSO)
# ============================================================
def save_pso_csv(results: list) -> None:
    fieldnames = ["c1", "c2", "success_rate", "mean_time_s", "mean_best"]
    with open(PSO_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nРезультаты PSO сохранены в {PSO_CSV}")


# ============================================================
#  Тепловые карты PSO
# ============================================================
def _build_pso_matrix(results: list, key: str) -> np.ndarray:
    """Собирает матрицу [c1_idx, c2_idx] из списка результатов."""
    matrix = np.zeros((len(C1_VALUES), len(C2_VALUES)))
    c1_map = {v: i for i, v in enumerate(C1_VALUES)}
    c2_map = {v: i for i, v in enumerate(C2_VALUES)}
    for r in results:
        matrix[c1_map[r["c1"]], c2_map[r["c2"]]] = r[key]
    return matrix


def _heatmap(matrix: np.ndarray, x_labels, y_labels,
             x_label: str, y_label: str, title: str,
             cmap: str, fmt: str, filename: str, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        matrix, origin="lower", aspect="auto", cmap=cmap,
        vmin=vmin, vmax=vmax,
        extent=[-0.5, len(x_labels) - 0.5, -0.5, len(y_labels) - 0.5],
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([fmt.format(v) for v in x_labels], rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([fmt.format(v) for v in y_labels])
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=13)

    # Подписи значений в ячейках
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = f"{val:.2f}" if abs(val) < 100 else f"{val:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if (im.norm(val) > _TEXT_DARK_THRESHOLD or im.norm(val) < _TEXT_LIGHT_THRESHOLD) else "black")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"График сохранён в {filename}")
    plt.close(fig)


def plot_pso_heatmaps(results: list) -> None:
    mat_success = _build_pso_matrix(results, "success_rate")
    mat_time    = _build_pso_matrix(results, "mean_time_s") * 1000   # → мс
    mat_best    = _build_pso_matrix(results, "mean_best")

    _heatmap(
        mat_success, C2_VALUES, C1_VALUES,
        x_label="Социальный коэффициент c2",
        y_label="Когнитивный коэффициент c1",
        title=(f"PSO: доля успешных запусков (f < {SUCCESS_THRESHOLD})\n"
               f"num_particles={PSO_NUM_PARTICLES}, max_iter={PSO_MAX_ITERATIONS}, "
               f"w={PSO_W}, {RUNS_PER_COMBO} запусков/точка"),
        cmap="RdYlGn", fmt="{:.2f}", filename=PSO_PNG_SUCCESS,
        vmin=0.0, vmax=1.0,
    )

    _heatmap(
        mat_time, C2_VALUES, C1_VALUES,
        x_label="Социальный коэффициент c2",
        y_label="Когнитивный коэффициент c1",
        title=(f"PSO: среднее время запуска (мс)\n"
               f"num_particles={PSO_NUM_PARTICLES}, max_iter={PSO_MAX_ITERATIONS}, "
               f"w={PSO_W}, {RUNS_PER_COMBO} запусков/точка"),
        cmap="YlOrRd_r", fmt="{:.2f}", filename=PSO_PNG_TIME,
    )

    _heatmap(
        mat_best, C2_VALUES, C1_VALUES,
        x_label="Социальный коэффициент c2",
        y_label="Когнитивный коэффициент c1",
        title=(f"PSO: среднее лучшее значение функции\n"
               f"num_particles={PSO_NUM_PARTICLES}, max_iter={PSO_MAX_ITERATIONS}, "
               f"w={PSO_W}, {RUNS_PER_COMBO} запусков/точка"),
        cmap="RdYlGn_r", fmt="{:.2f}", filename=PSO_PNG_BEST,
    )


# ============================================================
#  Эксперимент ГА
# ============================================================
def run_ga_experiment() -> list:
    """Перебирает все комбинации (p_mutation, p_crossover) и собирает статистику."""
    results = []
    total = len(PM_VALUES) * len(PC_VALUES)
    done = 0

    print("=" * 70)
    print(f"ГА: сетка pm × pc = {len(PM_VALUES)} × {len(PC_VALUES)} = {total} комбинаций")
    print(f"    {RUNS_PER_COMBO} запусков на комбинацию  →  {total * RUNS_PER_COMBO} запусков всего")
    print("=" * 70)

    for pm in PM_VALUES:
        for pc in PC_VALUES:
            best_values = []
            times = []
            successes = 0

            for _ in range(RUNS_PER_COMBO):
                t0 = time.perf_counter()
                result = run_genetic_algorithm(
                    population_size=GA_POPULATION_SIZE,
                    p_crossover=pc,
                    p_mutation=pm,
                    max_generations=GA_MAX_GENERATIONS,
                    elitism_count=GA_ELITISM_COUNT,
                )
                t1 = time.perf_counter()

                bv = result["best_value"]
                best_values.append(bv)
                times.append(t1 - t0)
                if bv < SUCCESS_THRESHOLD:
                    successes += 1

            done += 1
            success_rate = successes / RUNS_PER_COMBO
            mean_time_s  = float(np.mean(times))
            mean_best    = float(np.mean(best_values))

            results.append({
                "p_mutation":   pm,
                "p_crossover":  pc,
                "success_rate": success_rate,
                "mean_time_s":  mean_time_s,
                "mean_best":    mean_best,
            })

            print(
                f"[{done:>3}/{total}] pm={pm:.3f} pc={pc:.2f}  |  "
                f"успех={success_rate:.2f}  среднее={mean_best:9.5f}  "
                f"время={mean_time_s:.2f}с"
            )

    return results


# ============================================================
#  Сохранение CSV (ГА)
# ============================================================
def save_ga_csv(results: list) -> None:
    fieldnames = ["p_mutation", "p_crossover", "success_rate", "mean_time_s", "mean_best"]
    with open(GA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Результаты ГА сохранены в {GA_CSV}")


# ============================================================
#  Тепловые карты ГА
# ============================================================
def _build_ga_matrix(results: list, key: str) -> np.ndarray:
    matrix = np.zeros((len(PM_VALUES), len(PC_VALUES)))
    pm_map = {v: i for i, v in enumerate(PM_VALUES)}
    pc_map = {v: i for i, v in enumerate(PC_VALUES)}
    for r in results:
        matrix[pm_map[r["p_mutation"]], pc_map[r["p_crossover"]]] = r[key]
    return matrix


def plot_ga_heatmaps(results: list) -> None:
    mat_success = _build_ga_matrix(results, "success_rate")
    mat_time    = _build_ga_matrix(results, "mean_time_s")
    mat_best    = _build_ga_matrix(results, "mean_best")

    _heatmap(
        mat_success, PC_VALUES, PM_VALUES,
        x_label="Вероятность кроссовера (Pc)",
        y_label="Вероятность мутации (Pm)",
        title=(f"ГА: доля успешных запусков (f < {SUCCESS_THRESHOLD})\n"
               f"pop={GA_POPULATION_SIZE}, max_gen={GA_MAX_GENERATIONS}, "
               f"elitism={GA_ELITISM_COUNT}, {RUNS_PER_COMBO} запусков/точка"),
        cmap="RdYlGn", fmt="{:.2f}", filename=GA_PNG_SUCCESS,
        vmin=0.0, vmax=1.0,
    )

    _heatmap(
        mat_time, PC_VALUES, PM_VALUES,
        x_label="Вероятность кроссовера (Pc)",
        y_label="Вероятность мутации (Pm)",
        title=(f"ГА: среднее время запуска (с)\n"
               f"pop={GA_POPULATION_SIZE}, max_gen={GA_MAX_GENERATIONS}, "
               f"elitism={GA_ELITISM_COUNT}, {RUNS_PER_COMBO} запусков/точка"),
        cmap="YlOrRd_r", fmt="{:.2f}", filename=GA_PNG_TIME,
    )

    _heatmap(
        mat_best, PC_VALUES, PM_VALUES,
        x_label="Вероятность кроссовера (Pc)",
        y_label="Вероятность мутации (Pm)",
        title=(f"ГА: среднее лучшее значение функции\n"
               f"pop={GA_POPULATION_SIZE}, max_gen={GA_MAX_GENERATIONS}, "
               f"elitism={GA_ELITISM_COUNT}, {RUNS_PER_COMBO} запусков/точка"),
        cmap="RdYlGn_r", fmt="{:.2f}", filename=GA_PNG_BEST,
    )


# ============================================================
#  Точка входа
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  СРАВНИТЕЛЬНЫЙ АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ПАРАМЕТРОВ PSO и ГА")
    print("=" * 70)
    print(f"  Целевая функция : Holder Table  f(x,y)  (min ≈ −19.2085)")
    print(f"  Порог успеха    : f < {SUCCESS_THRESHOLD}")
    print(f"  Запусков/точка  : {RUNS_PER_COMBO}")
    print()

    # ------ PSO ------
    print(">>> Запуск эксперимента PSO...")
    pso_results = run_pso_experiment()
    save_pso_csv(pso_results)
    plot_pso_heatmaps(pso_results)

    # ------ ГА ------
    print("\n>>> Запуск эксперимента ГА...")
    ga_results = run_ga_experiment()
    save_ga_csv(ga_results)
    plot_ga_heatmaps(ga_results)

    print("\n" + "=" * 70)
    print("  Готово! Все результаты сохранены.")
    print(f"  PSO → {PSO_CSV}, {PSO_PNG_SUCCESS}, {PSO_PNG_TIME}, {PSO_PNG_BEST}")
    print(f"  ГА  → {GA_CSV}, {GA_PNG_SUCCESS}, {GA_PNG_TIME}, {GA_PNG_BEST}")
    print("=" * 70)
