import numpy as np
import matplotlib.pyplot as plt

from geneticAlgorithmWithoutModific import (
    run_genetic_algorithm,
    POPULATION_SIZE,
    P_MUTATION,
    MAX_GENERATIONS,
    ELITISM_COUNT,
    BORDER,
)

# ========================================
#           Experiment parameters
# ========================================
RUNS_PER_VALUE = 100
SUCCESS_THRESHOLD = -19          # считаем запуск успешным, если best_value < порога
PC_VALUES = np.arange(0.6, 1.01, 0.05)   # 0.60, 0.65, ..., 1.00


def run_experiment():
    success_rates = []

    for pc in PC_VALUES:
        successes = 0
        for _ in range(RUNS_PER_VALUE):
            result = run_genetic_algorithm(
                population_size=POPULATION_SIZE,
                p_crossover=float(pc),
                p_mutation=P_MUTATION,
                max_generations=MAX_GENERATIONS,
                elitism_count=ELITISM_COUNT,
            )
            if result["best_value"] < SUCCESS_THRESHOLD:
                successes += 1
        rate = successes / RUNS_PER_VALUE
        success_rates.append(rate)
        print(f"Pc = {pc:.2f}  ->  доля успешных запусков = {rate:.2f}  ({successes}/{RUNS_PER_VALUE})")

    return success_rates


def plot_results(success_rates):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(PC_VALUES, success_rates, marker="o", linewidth=2, markersize=7, color="steelblue")
    ax.axhline(y=max(success_rates), color="gray", linestyle="--", linewidth=1, label=f"макс. {max(success_rates):.2f}")

    best_pc = PC_VALUES[int(np.argmax(success_rates))]
    ax.axvline(x=best_pc, color="tomato", linestyle="--", linewidth=1, label=f"лучший Pc = {best_pc:.2f}")

    ax.set_xlabel("Вероятность кроссинговера (Pc)", fontsize=13)
    ax.set_ylabel("Доля успешных запусков", fontsize=13)
    ax.set_title(
        f"Влияние Pc на успешность ГА\n"
        f"(порог успеха: f < {SUCCESS_THRESHOLD}, {RUNS_PER_VALUE} запусков на точку)",
        fontsize=13,
    )
    ax.set_xticks(PC_VALUES)
    ax.set_xticklabels([f"{v:.2f}" for v in PC_VALUES], rotation=30)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("crossover_experiment.png", dpi=150)
    print("\nГрафик сохранён в crossover_experiment.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Эксперимент: влияние вероятности кроссинговера (Pc)")
    print(f"Фиксированные параметры:")
    print(f"  population_size = {POPULATION_SIZE}")
    print(f"  max_generations = {MAX_GENERATIONS}")
    print(f"  p_mutation      = {P_MUTATION}")
    print(f"  elitism_count   = {ELITISM_COUNT}")
    print(f"  border          = {BORDER}")
    print(f"  запусков / точка = {RUNS_PER_VALUE}")
    print(f"  порог успеха     = f < {SUCCESS_THRESHOLD}")
    print("=" * 60)

    success_rates = run_experiment()
    plot_results(success_rates)
