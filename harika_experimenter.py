"""
Harika Theory Experimenter
==========================
Проверка теории Харика: зависимость вероятности успеха генетического алгоритма
от размера популяции (S-кривая).

Все параметры зафиксированы. Варьируется только размер популяции N.
Мутация и элитизм отключены полностью.

Для каждого N выполняется 100 независимых запусков ГА с разными random seed.
Успех — нахождение значения функции ниже -19 (глобальный минимум ≈ -19.2085).
"""

import random
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geneticAlgorithmWithoutModific import run_genetic_algorithm, BORDER, FUNCTION

# ========================================
#           Зафиксированные параметры
# ========================================
P_CROSSOVER = 0.9       # вероятность кроссовера
P_MUTATION = 0.0        # мутация отключена
ELITISM_COUNT = 0       # элитизм отключён
MAX_GENERATIONS = 300   # число поколений

SUCCESS_THRESHOLD = -19.0   # порог успеха: значение функции ниже -19
RUNS_PER_N = 100            # число независимых запусков для каждого N

# Исследуемые размеры популяции
POPULATION_SIZES = [50, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000, 8000]

OUTPUT_FILE = "harika_results.png"


def run_experiment():
    print("=" * 60)
    print("Эксперимент: проверка теории Харика")
    print(f"  p_crossover    = {P_CROSSOVER}")
    print(f"  p_mutation     = {P_MUTATION}  (отключена)")
    print(f"  elitism_count  = {ELITISM_COUNT}  (отключён)")
    print(f"  max_generations= {MAX_GENERATIONS}")
    print(f"  runs_per_N     = {RUNS_PER_N}")
    print(f"  success_thresh = {SUCCESS_THRESHOLD}")
    print("=" * 60)

    success_rates = []

    for n in POPULATION_SIZES:
        successes = 0
        for seed in range(RUNS_PER_N):
            random.seed(seed)
            result = run_genetic_algorithm(
                population_size=n,
                p_crossover=P_CROSSOVER,
                p_mutation=P_MUTATION,
                max_generations=MAX_GENERATIONS,
                elitism_count=ELITISM_COUNT,
                border=BORDER,
                function=FUNCTION,
            )
            if result["best_value"] < SUCCESS_THRESHOLD:
                successes += 1

        rate = successes / RUNS_PER_N
        success_rates.append(rate)
        print(f"N = {n:5d} | успехов: {successes:3d}/{RUNS_PER_N} | P(успеха) = {rate:.2f}")

    _save_plot(POPULATION_SIZES, success_rates)
    return POPULATION_SIZES, success_rates


def _save_plot(pop_sizes, success_rates):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(pop_sizes, success_rates, marker="o", linewidth=2, color="steelblue",
            markersize=7, label="Эмпирическая P(успеха)")
    ax.axhline(y=0.95, color="red", linestyle="--", linewidth=1.2, label="95% порог (α = 0.05)")
    ax.axhline(y=0.50, color="orange", linestyle="--", linewidth=1.2, label="50% порог")

    ax.set_xlabel("Размер популяции N", fontsize=13)
    ax.set_ylabel("Вероятность успеха P(N)", fontsize=13)
    ax.set_title(
        "S-кривая Харика: зависимость вероятности нахождения глобального минимума\n"
        "функции Хёльдера от размера популяции ГА\n"
        f"(мутация и элитизм отключены, p_crossover={P_CROSSOVER}, "
        f"поколений={MAX_GENERATIONS}, запусков={RUNS_PER_N})",
        fontsize=11,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.close(fig)
    print(f"\nГрафик сохранён в файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_experiment()
