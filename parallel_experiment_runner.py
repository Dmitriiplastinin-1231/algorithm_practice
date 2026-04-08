"""
parallel_experiment_runner.py — Параллельный запуск экспериментов.

Запускает любой из экспериментов (elitism, mutation, crossover) используя
все доступные ядра CPU. Исходный код geneticAlgorithmWithoutModific.py
при этом НЕ изменяется — параллелизм достигается на уровне запуска.

Использование:
    python3 parallel_experiment_runner.py elitism    # исследование элитизма
    python3 parallel_experiment_runner.py mutation   # исследование мутации
    python3 parallel_experiment_runner.py crossover  # исследование скрещивания

Работает поверх существующих experiment-файлов, перехватывая
вычисление success_rate через multiprocessing.Pool без правки источника.
"""
import sys
import time
import multiprocessing as mp

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _run_point(point_args):
    """
    Запускает RUNS_PER_VALUE прогонов для одного значения параметра.
    point_args = (label, run_kwargs_list, threshold)
    """
    label, run_kwargs_list, threshold = point_args
    successes = 0
    for kwargs in run_kwargs_list:
        from geneticAlgorithmWithoutModific import run_genetic_algorithm
        result = run_genetic_algorithm(**kwargs)
        if result["best_value"] < threshold:
            successes += 1
    return label, successes, len(run_kwargs_list)


# ---------------------------------------------------------------------------
# Параллельный движок
# ---------------------------------------------------------------------------

def run_parallel(points, threshold, workers=None):
    """
    points: список (label, list_of_run_kwargs)
    threshold: порог успеха
    workers: число процессов (None = число CPU)

    Возвращает список (label, success_rate).
    """
    if workers is None:
        workers = mp.cpu_count()

    # Каждая «точка» — один элемент для пула
    tasks = [(label, kwargs_list, threshold) for label, kwargs_list in points]

    t0 = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        raw = pool.map(_run_point, tasks)
    elapsed = time.perf_counter() - t0

    results = []
    for label, successes, total in raw:
        rate = successes / total
        print(f"  {label}  ->  доля успешных = {rate:.2f}  ({successes}/{total})")
        results.append((label, rate))

    print(f"\nВремя (параллельно, {workers} ядра): {elapsed:.1f} с")
    return results


# ---------------------------------------------------------------------------
# Эксперимент: элитизм
# ---------------------------------------------------------------------------

def elitism_experiment(workers=None):
    import numpy as np
    from geneticAlgorithmWithoutModific import (
        POPULATION_SIZE, P_CROSSOVER, P_MUTATION, MAX_GENERATIONS,
    )

    RUNS_PER_VALUE   = 20          # уменьшено для демонстрации скорости
    SUCCESS_THRESHOLD = -19
    ELITISM_PERCENTS  = np.arange(0, 21, 1)

    print("=" * 60)
    print("Параллельный эксперимент: влияние элитизма на сходимость ГА")
    print(f"  запусков / точка = {RUNS_PER_VALUE}  (параллельно)")
    print("=" * 60)

    def make_kwargs(pct):
        elitism_count = int(round(POPULATION_SIZE * pct / 100))
        return {
            "population_size": POPULATION_SIZE,
            "p_crossover":     P_CROSSOVER,
            "p_mutation":      P_MUTATION,
            "max_generations": MAX_GENERATIONS,
            "elitism_count":   elitism_count,
        }

    points = [
        (f"Элитизм = {pct:2d}%  ({int(round(POPULATION_SIZE*pct/100)):4d} ос.)",
         [make_kwargs(pct)] * RUNS_PER_VALUE)
        for pct in ELITISM_PERCENTS
    ]

    return run_parallel(points, SUCCESS_THRESHOLD, workers)


# ---------------------------------------------------------------------------
# Эксперимент: мутация
# ---------------------------------------------------------------------------

def mutation_experiment(workers=None):
    import numpy as np
    from geneticAlgorithmWithoutModific import (
        POPULATION_SIZE, P_CROSSOVER, MAX_GENERATIONS, ELITISM_COUNT,
    )

    RUNS_PER_VALUE    = 20
    SUCCESS_THRESHOLD = -19
    MUTATION_VALUES   = np.arange(0.0, 1.05, 0.05)

    print("=" * 60)
    print("Параллельный эксперимент: влияние мутации на сходимость ГА")
    print(f"  запусков / точка = {RUNS_PER_VALUE}  (параллельно)")
    print("=" * 60)

    def make_kwargs(pm):
        return {
            "population_size": POPULATION_SIZE,
            "p_crossover":     P_CROSSOVER,
            "p_mutation":      float(pm),
            "max_generations": MAX_GENERATIONS,
            "elitism_count":   ELITISM_COUNT,
        }

    points = [
        (f"p_mutation = {pm:.2f}", [make_kwargs(pm)] * RUNS_PER_VALUE)
        for pm in MUTATION_VALUES
    ]

    return run_parallel(points, SUCCESS_THRESHOLD, workers)


# ---------------------------------------------------------------------------
# Эксперимент: скрещивание
# ---------------------------------------------------------------------------

def crossover_experiment(workers=None):
    import numpy as np
    from geneticAlgorithmWithoutModific import (
        POPULATION_SIZE, P_MUTATION, MAX_GENERATIONS, ELITISM_COUNT,
    )

    RUNS_PER_VALUE    = 20
    SUCCESS_THRESHOLD = -19
    CROSSOVER_VALUES  = np.arange(0.0, 1.05, 0.05)

    print("=" * 60)
    print("Параллельный эксперимент: влияние скрещивания на сходимость ГА")
    print(f"  запусков / точка = {RUNS_PER_VALUE}  (параллельно)")
    print("=" * 60)

    def make_kwargs(pc):
        return {
            "population_size": POPULATION_SIZE,
            "p_crossover":     float(pc),
            "p_mutation":      P_MUTATION,
            "max_generations": MAX_GENERATIONS,
            "elitism_count":   ELITISM_COUNT,
        }

    points = [
        (f"p_crossover = {pc:.2f}", [make_kwargs(pc)] * RUNS_PER_VALUE)
        for pc in CROSSOVER_VALUES
    ]

    return run_parallel(points, SUCCESS_THRESHOLD, workers)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "elitism":   elitism_experiment,
    "mutation":  mutation_experiment,
    "crossover": crossover_experiment,
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "elitism"
    if name not in EXPERIMENTS:
        print(f"Неизвестный эксперимент '{name}'. Доступны: {list(EXPERIMENTS)}")
        sys.exit(1)

    EXPERIMENTS[name]()
