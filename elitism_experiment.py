import csv
import os

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
RUNS_PER_VALUE = 50
SUCCESS_THRESHOLD = -19          # считаем запуск успешным, если best_value < порога
ELITISM_PERCENTS = np.arange(0, 21, 1)   # 0%, 1%, ..., 20%
CSV_PATH = "elitism_experiment_results.csv"


def _elitism_count_from_percent(percent):
    """Вычисляет число элитных особей по проценту от популяции."""
    return int(round(POPULATION_SIZE * percent / 100))


def run_experiment():
    """
    Для каждого значения процента элитизма запускает RUNS_PER_VALUE независимых
    экспериментов и собирает следующие статистики:
      - success_rate           : доля успешных запусков (best_value < SUCCESS_THRESHOLD)
      - avg_convergence_gen    : среднее поколение до сходимости (по успешным запускам)
      - best_fitness           : лучшее найденное значение функции приспособленности
      - avg_best_fitness       : среднее лучшее значение по всем запускам
      - avg_mean_fitness       : среднее «среднее поколенческое» значение (последнее поколение)

    Результаты сохраняются в CSV и возвращаются в виде списка словарей.
    """
    rows = []

    for pct in ELITISM_PERCENTS:
        elitism_count = _elitism_count_from_percent(pct)
        successes = 0
        convergence_gens = []
        best_fitnesses = []
        final_mean_fitnesses = []

        for _ in range(RUNS_PER_VALUE):
            result = run_genetic_algorithm(
                population_size=POPULATION_SIZE,
                p_crossover=P_CROSSOVER,
                p_mutation=P_MUTATION,
                max_generations=MAX_GENERATIONS,
                elitism_count=elitism_count,
                convergence_threshold=SUCCESS_THRESHOLD,
            )

            bv = result["best_value"]
            best_fitnesses.append(bv)
            final_mean_fitnesses.append(result["mean_fitness_history"][-1])

            if bv < SUCCESS_THRESHOLD:
                successes += 1
                if result["convergence_generation"] is not None:
                    convergence_gens.append(result["convergence_generation"])

        success_rate = successes / RUNS_PER_VALUE
        avg_conv_gen = np.mean(convergence_gens) if convergence_gens else float("nan")
        best_fitness = min(best_fitnesses)
        avg_best_fitness = np.mean(best_fitnesses)
        avg_mean_fitness = np.mean(final_mean_fitnesses)

        row = {
            "elitism_percent": int(pct),
            "elitism_count": elitism_count,
            "success_rate": success_rate,
            "avg_convergence_generation": avg_conv_gen,
            "best_fitness": best_fitness,
            "avg_best_fitness": avg_best_fitness,
            "avg_mean_fitness": avg_mean_fitness,
            "runs": RUNS_PER_VALUE,
            "successes": successes,
        }
        rows.append(row)

        print(
            f"Элитизм = {pct:2d}%  ({elitism_count:4d} особей) | "
            f"доля успеха = {success_rate:.2f} ({successes}/{RUNS_PER_VALUE}) | "
            f"ср. поколение сходимости = {avg_conv_gen:.1f} | "
            f"лучший f = {best_fitness:.4f} | "
            f"ср. лучший f = {avg_best_fitness:.4f}"
        )

    _save_csv(rows)
    return rows


def _save_csv(rows):
    """Сохраняет результаты эксперимента в CSV-файл."""
    fieldnames = [
        "elitism_percent",
        "elitism_count",
        "runs",
        "successes",
        "success_rate",
        "avg_convergence_generation",
        "best_fitness",
        "avg_best_fitness",
        "avg_mean_fitness",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nРезультаты сохранены в {os.path.abspath(CSV_PATH)}")


def plot_results(rows):
    """Строит четыре графика по собранным данным и сохраняет изображение."""
    percents = [r["elitism_percent"] for r in rows]
    success_rates = [r["success_rate"] for r in rows]
    avg_conv_gens = [r["avg_convergence_generation"] for r in rows]
    avg_best_fitnesses = [r["avg_best_fitness"] for r in rows]
    avg_mean_fitnesses = [r["avg_mean_fitness"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Влияние уровня элитизма на сходимость ГА\n"
        f"(популяция={POPULATION_SIZE}, поколений={MAX_GENERATIONS}, "
        f"P_cross={P_CROSSOVER}, P_mut={P_MUTATION}, "
        f"{RUNS_PER_VALUE} запусков на точку)",
        fontsize=12,
    )

    def _add_best_marker(ax, xs, ys, best_func=min):
        best_y = best_func(ys)
        best_idx = (np.argmax(ys) if best_func is max else np.argmin(ys))
        best_x = xs[int(best_idx)]
        ax.axvline(x=best_x, color="tomato", linestyle="--", linewidth=1,
                   label=f"лучший элитизм: {best_x}%")
        return best_x, best_y

    # --- 1. Доля успешных запусков ---
    ax = axes[0, 0]
    ax.plot(percents, success_rates, marker="o", linewidth=2, markersize=6, color="steelblue")
    _add_best_marker(ax, percents, success_rates, best_func=max)
    ax.set_xlabel("Уровень элитизма, %")
    ax.set_ylabel("Доля успешных запусков")
    ax.set_title("Доля успешных запусков")
    ax.set_xticks(percents)
    ax.set_xticklabels([f"{v}%" for v in percents], rotation=45)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- 2. Среднее поколение до сходимости ---
    ax = axes[0, 1]
    valid_percents = [p for p, g in zip(percents, avg_conv_gens) if not np.isnan(g)]
    valid_gens = [g for g in avg_conv_gens if not np.isnan(g)]
    if valid_gens:
        ax.plot(valid_percents, valid_gens, marker="s", linewidth=2, markersize=6, color="seagreen")
        _add_best_marker(ax, valid_percents, valid_gens, best_func=min)
    ax.set_xlabel("Уровень элитизма, %")
    ax.set_ylabel("Среднее поколение до сходимости")
    ax.set_title(f"Среднее поколение до сходимости\n(только успешные запуски, порог f < {SUCCESS_THRESHOLD})")
    ax.set_xticks(percents)
    ax.set_xticklabels([f"{v}%" for v in percents], rotation=45)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- 3. Среднее лучшее значение функции ---
    ax = axes[1, 0]
    ax.plot(percents, avg_best_fitnesses, marker="^", linewidth=2, markersize=6, color="darkorange")
    _add_best_marker(ax, percents, avg_best_fitnesses, best_func=min)
    ax.set_xlabel("Уровень элитизма, %")
    ax.set_ylabel("Среднее лучшее значение f")
    ax.set_title("Среднее лучшее значение функции приспособленности")
    ax.set_xticks(percents)
    ax.set_xticklabels([f"{v}%" for v in percents], rotation=45)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- 4. Среднее значение функции (последнее поколение) ---
    ax = axes[1, 1]
    ax.plot(percents, avg_mean_fitnesses, marker="D", linewidth=2, markersize=6, color="mediumpurple")
    _add_best_marker(ax, percents, avg_mean_fitnesses, best_func=min)
    ax.set_xlabel("Уровень элитизма, %")
    ax.set_ylabel("Среднее значение f (последнее поколение)")
    ax.set_title("Среднее значение функции приспособленности (последнее поколение)")
    ax.set_xticks(percents)
    ax.set_xticklabels([f"{v}%" for v in percents], rotation=45)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("elitism_experiment.png", dpi=150)
    print("График сохранён в elitism_experiment.png")
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

    rows = run_experiment()
    plot_results(rows)

