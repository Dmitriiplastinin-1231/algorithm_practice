"""
Сравнение генетического алгоритма (ГА) и роевого алгоритма (PSO).

Программа многократно запускает оба алгоритма на одной и той же тестовой
функции (Holder Table), фиксирует ключевые метрики каждого запуска и
сохраняет сводную статистику в CSV-файл.

Метрики на один запуск
-----------------------
  - время выполнения (сек)
  - итерация / поколение первого пересечения порога успеха
  - лучшее найденное значение функции
  - флаг успешной сходимости (best < SUCCESS_THRESHOLD)

Сводные метрики (по всем запускам)
-----------------------------------
  - среднее время
  - среднее число итераций / поколений до сходимости
  - среднее лучшее значение
  - лучшее из всех запусков
  - процент успешных сходимостей
  - стандартные отклонения (время, итерации, значение)

Результаты: ga_vs_pso_comparison.csv
"""

import csv
import math
import sys
import time

from pso import PSO
from geneticAlgorithmWithoutModific import run_genetic_algorithm

# ===========================================================================
#  Тестовая функция (Holder Table, глобальный минимум ≈ −19.2085)
# ===========================================================================
FUNCTION = lambda x, y: -(
    math.fabs(
        math.sin(x)
        * math.cos(y)
        * math.exp(math.fabs(1 - (((x ** 2 + y ** 2) ** 0.5) / math.pi)))
    )
)

BOUNDS = [(-10, 10), (-10, 10)]

# ===========================================================================
#  Параметры эксперимента
# ===========================================================================
NUM_RUNS        = 30        # количество независимых запусков каждого алгоритма
SUCCESS_THRESHOLD = -19.0   # порог: считаем сходимость успешной, если best < порога

# --- Параметры ГА ---
GA_POPULATION_SIZE  = 300
GA_P_CROSSOVER      = 0.9
GA_P_MUTATION       = 0.3
GA_MAX_GENERATIONS  = 150
GA_ELITISM_COUNT    = 10

# --- Параметры PSO ---
PSO_NUM_PARTICLES  = 30
PSO_MAX_ITERATIONS = 100
PSO_C1             = 2.05
PSO_C2             = 2.05
PSO_USE_CONSTRICTION = True   # χ вычисляется автоматически

# --- Выходной файл ---
CSV_FILE = "ga_vs_pso_comparison.csv"

# ===========================================================================
#  Вспомогательные функции
# ===========================================================================

def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _convergence_step(history, threshold):
    """
    Возвращает номер шага (1-based), на котором история впервые опускается
    ниже threshold. Если порог не достигнут — возвращает len(history).
    """
    for i, val in enumerate(history, start=1):
        if val < threshold:
            return i
    return len(history)


# ===========================================================================
#  Одиночный запуск ГА
# ===========================================================================

def _run_ga_once():
    """
    Запускает ГА один раз и возвращает словарь с метриками.
    """
    t_start = time.perf_counter()
    result = run_genetic_algorithm(
        population_size=GA_POPULATION_SIZE,
        p_crossover=GA_P_CROSSOVER,
        p_mutation=GA_P_MUTATION,
        max_generations=GA_MAX_GENERATIONS,
        elitism_count=GA_ELITISM_COUNT,
        border=BOUNDS,
        function=FUNCTION,
    )
    elapsed = time.perf_counter() - t_start

    best_val = result["best_value"]
    history  = result["best_fitness_history"]
    conv_gen = _convergence_step(history, SUCCESS_THRESHOLD)
    success  = best_val < SUCCESS_THRESHOLD

    return {
        "algorithm":      "GA",
        "best_value":     best_val,
        "conv_step":      conv_gen,
        "time_s":         elapsed,
        "success":        int(success),
    }


# ===========================================================================
#  Одиночный запуск PSO
# ===========================================================================

def _run_pso_once():
    """
    Запускает PSO один раз и возвращает словарь с метриками.
    """
    optimizer = PSO(
        func=FUNCTION,
        dimensions=2,
        bounds=BOUNDS,
        num_particles=PSO_NUM_PARTICLES,
        max_iterations=PSO_MAX_ITERATIONS,
        c1=PSO_C1,
        c2=PSO_C2,
        use_constriction=PSO_USE_CONSTRICTION,
    )

    t_start = time.perf_counter()
    _, best_val = optimizer.optimize()
    elapsed = time.perf_counter() - t_start

    conv_iter = _convergence_step(optimizer.history, SUCCESS_THRESHOLD)
    success   = best_val < SUCCESS_THRESHOLD

    return {
        "algorithm":  "PSO",
        "best_value": best_val,
        "conv_step":  conv_iter,
        "time_s":     elapsed,
        "success":    int(success),
    }


# ===========================================================================
#  Основной эксперимент
# ===========================================================================

def run_experiment():
    """
    Многократно запускает ГА и PSO, собирает и сохраняет статистику.
    Возвращает (ga_rows, pso_rows) — списки словарей с данными каждого запуска.
    """
    ga_rows  = []
    pso_rows = []

    print("=" * 70)
    print(f"  Сравнение ГА и PSO  |  {NUM_RUNS} запусков каждого")
    print(f"  Функция: Holder Table  |  порог успеха: f < {SUCCESS_THRESHOLD}")
    print("=" * 70)

    # --- Запуски ГА ---
    print(f"\n[ГА] Параметры: pop={GA_POPULATION_SIZE}, поколений={GA_MAX_GENERATIONS}, "
          f"кроссовер={GA_P_CROSSOVER}, мутация={GA_P_MUTATION}, элитизм={GA_ELITISM_COUNT}")
    print(f"[ГА] Выполняю {NUM_RUNS} запусков...")

    for i in range(1, NUM_RUNS + 1):
        row = _run_ga_once()
        row["run"] = i
        ga_rows.append(row)
        _progress_dot(i, NUM_RUNS)

    print()

    # --- Запуски PSO ---
    print(f"\n[PSO] Параметры: частиц={PSO_NUM_PARTICLES}, итераций={PSO_MAX_ITERATIONS}, "
          f"c1={PSO_C1}, c2={PSO_C2}, constriction={PSO_USE_CONSTRICTION}")
    print(f"[PSO] Выполняю {NUM_RUNS} запусков...")

    for i in range(1, NUM_RUNS + 1):
        row = _run_pso_once()
        row["run"] = i
        pso_rows.append(row)
        _progress_dot(i, NUM_RUNS)

    print()

    return ga_rows, pso_rows


def _progress_dot(i, total):
    """Выводит прогресс-индикатор без переноса строки."""
    if i % max(1, total // 10) == 0 or i == total:
        pct = i * 100 // total
        sys.stdout.write(f"\r  {i}/{total} ({pct}%)")
        sys.stdout.flush()


# ===========================================================================
#  Сводная статистика
# ===========================================================================

def compute_summary(rows):
    """
    Принимает список словарей (per-run), возвращает сводный словарь.
    """
    best_vals  = [r["best_value"] for r in rows]
    conv_steps = [r["conv_step"]  for r in rows]
    times      = [r["time_s"]     for r in rows]
    successes  = [r["success"]    for r in rows]

    return {
        "algorithm":        rows[0]["algorithm"],
        "runs":             len(rows),
        "mean_best_value":  _mean(best_vals),
        "std_best_value":   _std(best_vals),
        "best_of_best":     min(best_vals),
        "mean_conv_step":   _mean(conv_steps),
        "std_conv_step":    _std(conv_steps),
        "mean_time_s":      _mean(times),
        "std_time_s":       _std(times),
        "success_rate":     sum(successes) / len(successes),
    }


# ===========================================================================
#  Сохранение в CSV
# ===========================================================================

def save_csv(ga_rows, pso_rows, ga_summary, pso_summary):
    """
    Сохраняет в CSV-файл:
      - секцию со всеми отдельными запусками (per-run)
      - секцию со сводной статистикой (summary)
    """
    all_rows = ga_rows + pso_rows

    per_run_fields = ["run", "algorithm", "best_value", "conv_step", "time_s", "success"]
    summary_fields = [
        "algorithm", "runs",
        "mean_best_value", "std_best_value", "best_of_best",
        "mean_conv_step",  "std_conv_step",
        "mean_time_s",     "std_time_s",
        "success_rate",
    ]

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # --- Заголовок эксперимента ---
        writer.writerow(["# Сравнение ГА и PSO"])
        writer.writerow(["# Функция", "Holder Table"])
        writer.writerow(["# Порог успеха", SUCCESS_THRESHOLD])
        writer.writerow(["# Параметры ГА",
                         f"pop={GA_POPULATION_SIZE}",
                         f"generations={GA_MAX_GENERATIONS}",
                         f"crossover={GA_P_CROSSOVER}",
                         f"mutation={GA_P_MUTATION}",
                         f"elitism={GA_ELITISM_COUNT}"])
        writer.writerow(["# Параметры PSO",
                         f"particles={PSO_NUM_PARTICLES}",
                         f"iterations={PSO_MAX_ITERATIONS}",
                         f"c1={PSO_C1}",
                         f"c2={PSO_C2}",
                         f"constriction={PSO_USE_CONSTRICTION}"])
        writer.writerow([])

        # --- Данные каждого запуска ---
        writer.writerow(["## Данные каждого запуска"])
        writer.writerow(per_run_fields)
        for row in all_rows:
            writer.writerow([row[k] for k in per_run_fields])

        writer.writerow([])

        # --- Сводная статистика ---
        writer.writerow(["## Сводная статистика"])
        writer.writerow(summary_fields)
        for summary in (ga_summary, pso_summary):
            writer.writerow([summary[k] for k in summary_fields])

    print(f"\nРезультаты сохранены в {CSV_FILE}")


# ===========================================================================
#  Вывод итоговой таблицы в консоль
# ===========================================================================

def print_summary(ga_summary, pso_summary):
    col = 28
    sep = "=" * 68

    print(f"\n{sep}")
    print("  ИТОГОВОЕ СРАВНЕНИЕ ГА vs PSO")
    print(sep)
    header = f"  {'Метрика':<{col}}  {'ГА':>16}  {'PSO':>16}"
    print(header)
    print("-" * 68)

    rows = [
        ("Запусков",               f"{ga_summary['runs']}",
                                   f"{pso_summary['runs']}"),
        ("Среднее лучшее значение",f"{ga_summary['mean_best_value']:>16.6f}",
                                   f"{pso_summary['mean_best_value']:>16.6f}"),
        ("Ст. откл. лучшего знач.",f"{ga_summary['std_best_value']:>16.6f}",
                                   f"{pso_summary['std_best_value']:>16.6f}"),
        ("Лучший результат",       f"{ga_summary['best_of_best']:>16.6f}",
                                   f"{pso_summary['best_of_best']:>16.6f}"),
        ("Среднее шагов до порога",f"{ga_summary['mean_conv_step']:>16.1f}",
                                   f"{pso_summary['mean_conv_step']:>16.1f}"),
        ("Ст. откл. шагов",        f"{ga_summary['std_conv_step']:>16.1f}",
                                   f"{pso_summary['std_conv_step']:>16.1f}"),
        ("Среднее время (сек)",    f"{ga_summary['mean_time_s']:>16.4f}",
                                   f"{pso_summary['mean_time_s']:>16.4f}"),
        ("Ст. откл. времени (сек)",f"{ga_summary['std_time_s']:>16.4f}",
                                   f"{pso_summary['std_time_s']:>16.4f}"),
        ("Доля успешных сходим.",  f"{ga_summary['success_rate']:>15.1%} ",
                                   f"{pso_summary['success_rate']:>15.1%} "),
    ]

    for label, ga_val, pso_val in rows:
        print(f"  {label:<{col}}  {ga_val}  {pso_val}")

    print(sep)
    print(f"  Порог успеха: f(x,y) < {SUCCESS_THRESHOLD}  "
          f"(глобальный минимум ≈ −19.2085)")
    print(sep)


# ===========================================================================
#  Точка входа
# ===========================================================================

if __name__ == "__main__":
    ga_rows, pso_rows = run_experiment()

    ga_summary  = compute_summary(ga_rows)
    pso_summary = compute_summary(pso_rows)

    save_csv(ga_rows, pso_rows, ga_summary, pso_summary)
    print_summary(ga_summary, pso_summary)
