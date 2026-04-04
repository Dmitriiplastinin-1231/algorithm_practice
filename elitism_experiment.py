import numpy as np
import matplotlib.pyplot as plt

from geneticAlgorithmWithoutModific import (
    run_genetic_algorithm,
    POPULATION_SIZE,
    P_CROSSOVER,
    P_MUTATION,
    MAX_GENERATIONS,
    BORDER,
)

# ========================================
#           Experiment parameters
# ========================================
RUNS_PER_VALUE = 100
SUCCESS_THRESHOLD = -19          # считаем запуск успешным, если best_value < порога
ELITISM_PERCENTS = np.arange(0, 21, 1)   # 0%, 1%, ..., 20%


def _elitism_count_from_percent(percent):
    """Вычисляет число элитных особей по проценту от популяции."""
    return int(round(POPULATION_SIZE * percent / 100))


def run_experiment():
    success_rates = []

    for pct in ELITISM_PERCENTS:
        elitism_count = _elitism_count_from_percent(pct)
        successes = 0
        for _ in range(RUNS_PER_VALUE):
            result = run_genetic_algorithm(
                population_size=POPULATION_SIZE,
                p_crossover=P_CROSSOVER,
                p_mutation=P_MUTATION,
                max_generations=MAX_GENERATIONS,
                elitism_count=elitism_count,
            )
            if result["best_value"] < SUCCESS_THRESHOLD:
                successes += 1
        rate = successes / RUNS_PER_VALUE
        success_rates.append(rate)
        print(
            f"Элитизм = {pct:2d}%  ({elitism_count:4d} особей)  ->  "
            f"доля успешных = {rate:.2f}  ({successes}/{RUNS_PER_VALUE})"
        )

    return success_rates


def plot_results(success_rates):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        ELITISM_PERCENTS,
        success_rates,
        marker="o",
        linewidth=2,
        markersize=7,
        color="steelblue",
    )

    max_rate = max(success_rates)
    best_pct = ELITISM_PERCENTS[int(np.argmax(success_rates))]

    ax.axhline(
        y=max_rate,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"макс. доля успеха: {max_rate:.2f}",
    )
    ax.axvline(
        x=best_pct,
        color="tomato",
        linestyle="--",
        linewidth=1,
        label=f"лучший уровень элитизма: {best_pct}%",
    )

    ax.set_xlabel("Уровень элитизма (% от популяции)", fontsize=13)
    ax.set_ylabel("Доля успешных запусков", fontsize=13)
    ax.set_title(
        f"Влияние уровня элитизма на сходимость ГА\n"
        f"(порог успеха: f < {SUCCESS_THRESHOLD}, {RUNS_PER_VALUE} запусков на точку)",
        fontsize=13,
    )
    ax.set_xticks(ELITISM_PERCENTS)
    ax.set_xticklabels([f"{v}%" for v in ELITISM_PERCENTS], rotation=45)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("elitism_experiment.png", dpi=150)
    print("\nГрафик сохранён в elitism_experiment.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Эксперимент: влияние уровня элитизма на сходимость ГА")
    print("Фиксированные параметры:")
    print(f"  population_size  = {POPULATION_SIZE}")
    print(f"  max_generations  = {MAX_GENERATIONS}")
    print(f"  p_crossover      = {P_CROSSOVER}")
    print(f"  p_mutation       = {P_MUTATION}")
    print(f"  border           = {BORDER}")
    print(f"  запусков / точка = {RUNS_PER_VALUE}")
    print(f"  порог успеха     = f < {SUCCESS_THRESHOLD}")
    print(f"  диапазон элитизма: {ELITISM_PERCENTS[0]}% – {ELITISM_PERCENTS[-1]}% (шаг 1%)")
    print("=" * 60)

    success_rates = run_experiment()
    plot_results(success_rates)
