import numpy as np
import matplotlib.pyplot as plt

from geneticAlgorithmWithoutModific import (
    run_genetic_algorithm,
    POPULATION_SIZE,
    P_CROSSOVER,
    MAX_GENERATIONS,
    ELITISM_COUNT,
    BORDER,
)

# ========================================
#           Experiment parameters
# ========================================
RUNS_PER_VALUE = 100
SUCCESS_THRESHOLD = -19          # считаем запуск успешным, если best_value < порога
PM_VALUES = np.arange(0.0, 0.11, 0.01)   # 0.00, 0.01, ..., 0.10


def run_experiment():
    success_rates = []

    for pm in PM_VALUES:
        successes = 0
        for _ in range(RUNS_PER_VALUE):
            result = run_genetic_algorithm(
                population_size=POPULATION_SIZE,
                p_crossover=P_CROSSOVER,
                p_mutation=float(pm),
                max_generations=MAX_GENERATIONS,
                elitism_count=ELITISM_COUNT,
            )
            if result["best_value"] < SUCCESS_THRESHOLD:
                successes += 1
        rate = successes / RUNS_PER_VALUE
        success_rates.append(rate)
        print(f"Pm = {pm:.2f}  ->  доля успешных запусков = {rate:.2f}  ({successes}/{RUNS_PER_VALUE})")

    return success_rates


def plot_results(success_rates):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(PM_VALUES, success_rates, marker="o", linewidth=2, markersize=7, color="steelblue")
    ax.axhline(y=max(success_rates), color="gray", linestyle="--", linewidth=1, label=f"макс. {max(success_rates):.2f}")

    best_pm = PM_VALUES[int(np.argmax(success_rates))]
    ax.axvline(x=best_pm, color="tomato", linestyle="--", linewidth=1, label=f"лучший Pm = {best_pm:.2f}")

    ax.set_xlabel("Вероятность мутации (Pm)", fontsize=13)
    ax.set_ylabel("Доля успешных запусков", fontsize=13)
    ax.set_title(
        f"Влияние Pm на успешность ГА\n"
        f"(порог успеха: f < {SUCCESS_THRESHOLD}, {RUNS_PER_VALUE} запусков на точку)",
        fontsize=13,
    )
    ax.set_xticks(PM_VALUES)
    ax.set_xticklabels([f"{v:.2f}" for v in PM_VALUES], rotation=30)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("mutation_experiment.png", dpi=150)
    print("\nГрафик сохранён в mutation_experiment.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Эксперимент: влияние вероятности мутации (Pm)")
    print(f"Фиксированные параметры:")
    print(f"  population_size = {POPULATION_SIZE}")
    print(f"  max_generations = {MAX_GENERATIONS}")
    print(f"  p_crossover     = {P_CROSSOVER}")
    print(f"  elitism_count   = {ELITISM_COUNT}")
    print(f"  border          = {BORDER}")
    print(f"  запусков / точка = {RUNS_PER_VALUE}")
    print(f"  порог успеха     = f < {SUCCESS_THRESHOLD}")
    print("=" * 60)

    success_rates = run_experiment()
    plot_results(success_rates)
