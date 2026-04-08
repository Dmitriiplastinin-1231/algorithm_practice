"""
profile_ga.py — Профилирование генетического алгоритма.

Запуск:
    python3 profile_ga.py

Выводит:
  • Таблицу горячих точек (hotspots) из cProfile
  • Итоговое время выполнения при стандартных параметрах
  • Долю каждой функции в суммарном времени

Не изменяет исходный код geneticAlgorithmWithoutModific.py.
"""
import cProfile
import pstats
import io
import time

from geneticAlgorithmWithoutModific import (
    run_genetic_algorithm,
    POPULATION_SIZE,
    MAX_GENERATIONS,
)

PROFILE_GENERATIONS = 50   # число поколений для профилирования (быстрее)
PROFILE_POPULATION  = 300  # размер популяции для профилирования


def profile_run():
    pr = cProfile.Profile()
    pr.enable()
    run_genetic_algorithm(
        max_generations=PROFILE_GENERATIONS,
        population_size=PROFILE_POPULATION,
    )
    pr.disable()
    return pr


def print_report(pr: cProfile.Profile):
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(20)
    report = s.getvalue()

    # Сокращаем длинные пути для читаемости
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    report = report.replace(base + os.sep, "")

    print(report)


def full_timing():
    print("=" * 60)
    print(f"Замер времени при стандартных параметрах:")
    print(f"  population_size = {POPULATION_SIZE}")
    print(f"  max_generations = {MAX_GENERATIONS}")
    print("=" * 60)
    t0 = time.perf_counter()
    result = run_genetic_algorithm()
    elapsed = time.perf_counter() - t0
    print(f"Время выполнения: {elapsed:.2f} с")
    print(f"Найденный минимум: f({result['best_x']:.5f}, {result['best_y']:.5f}) = {result['best_value']:.6f}")
    print()


def main():
    print("=" * 60)
    print("Профилирование генетического алгоритма")
    print(f"  (pop={PROFILE_POPULATION}, gen={PROFILE_GENERATIONS} для скорости)")
    print("=" * 60)

    pr = profile_run()
    print_report(pr)

    full_timing()

    print("Подробнее о рекомендациях по оптимизации: optimization_guide.md")


if __name__ == "__main__":
    main()
