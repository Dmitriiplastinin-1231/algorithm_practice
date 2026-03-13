import math

from pso import PSO


# ---------------------------------------------------------------
# Тестовые функции
# ---------------------------------------------------------------



FUNCTION = lambda x, y: -(math.fabs(    math.sin(x) * math.cos(y) * math.exp(math.fabs(1 - (((x**2 + y**2)**0.5)/math.pi)))     ))



# ---------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------

def run_example(name: str, func, dimensions: int, bounds, known_min: str):
    """Запуск одного теста: сравнение классического PSO и PSO с коэф. сжатия."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  Известный минимум: {known_min}")
    print(f"{'=' * 60}")



    print(f"\n  [PSO]")
    optimizer_constr = PSO(
        func=func,
        dimensions=dimensions,
        bounds=bounds,
        num_particles=40,
        max_iterations=100,
        seed=42,
        use_constriction=True,
        c1=2.05,
        c2=2.05,
    )
    best_pos_x, best_val_x = optimizer_constr.optimize(verbose=True)
    pos_str_x = ", ".join(f"{x:.8f}" for x in best_pos_x)
    print(f"\n  Результат (коэф. сжатия, χ = {optimizer_constr.chi:.6f}):")
    print(f"    Лучшая позиция : ({pos_str_x})")
    print(f"    Лучшее значение: {best_val_x:.10f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  АЛГОРИТМ РОЯ ЧАСТИЦ (PSO) — поиск минимума функции")
    print("=" * 60)

    # 1. Функция сферы
    run_example(
        name="my_function",
        func=FUNCTION,
        dimensions=2,
        bounds=[(-10, 10), (-10, 10)],
        known_min="",
    )


    print(f"\n{'=' * 60}")
    print(f"{'=' * 60}")
