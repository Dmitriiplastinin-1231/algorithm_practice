"""
Графический интерфейс для сравнения генетического алгоритма и алгоритма роя частиц.

Запуск: python gui.py
"""

import tkinter as tk
from tkinter import ttk
import threading
import math

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib import cm


# Функция, минимум которой ищут оба алгоритма
FUNCTION = lambda x, y: -(math.fabs(
    math.sin(x) * math.cos(y) * math.exp(math.fabs(1 - (math.sqrt(x**2 + y**2) / math.pi)))
))

BOUNDS = [(-10, 10), (-10, 10)]


# ---------------------------------------------------------------------------
# Вспомогательная функция: строим сетку значений функции для контурного графика
# ---------------------------------------------------------------------------

def _build_contour_data(bounds, nx=300, ny=300):
    xs = np.linspace(bounds[0][0], bounds[0][1], nx)
    ys = np.linspace(bounds[1][0], bounds[1][1], ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(FUNCTION)(X, Y)
    return X, Y, Z


# ---------------------------------------------------------------------------
# Окно выбора алгоритма
# ---------------------------------------------------------------------------

class SelectionWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Выбор алгоритма оптимизации")
        self.root.resizable(False, False)

        # ---- Центрирование окна ----
        w, h = 440, 260
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        # ---- Виджеты ----
        tk.Label(
            root,
            text="Выберите алгоритм оптимизации",
            font=("Arial", 15, "bold"),
        ).pack(pady=30)

        btn_genetic = tk.Button(
            root,
            text="🧬  Генетический алгоритм",
            command=self._open_genetic,
            font=("Arial", 12),
            width=28,
            height=2,
            bg="#4a90d9",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
        )
        btn_genetic.pack(pady=8)

        btn_pso = tk.Button(
            root,
            text="🐝  Роевой алгоритм (PSO)",
            command=self._open_pso,
            font=("Arial", 12),
            width=28,
            height=2,
            bg="#e67e22",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
        )
        btn_pso.pack(pady=8)

    def _open_genetic(self):
        AlgorithmWindow(self.root, "genetic")

    def _open_pso(self):
        AlgorithmWindow(self.root, "pso")


# ---------------------------------------------------------------------------
# Окно с параметрами и графиком
# ---------------------------------------------------------------------------

class AlgorithmWindow:
    _LABEL_STYLE = {"font": ("Arial", 10), "anchor": "w"}
    _ENTRY_WIDTH = 10

    def __init__(self, parent: tk.Tk, algo_type: str):
        self.algo_type = algo_type
        self._running = False

        self.window = tk.Toplevel(parent)
        title = (
            "Генетический алгоритм — поиск минимума"
            if algo_type == "genetic"
            else "Роевой алгоритм (PSO) — поиск минимума"
        )
        self.window.title(title)
        self.window.geometry("1280x700")
        self.window.minsize(900, 600)

        # ---- Основные панели ----
        left_frame = tk.Frame(self.window, bg="#f4f4f4", width=280)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        left_frame.pack_propagate(False)

        separator = tk.Frame(self.window, width=2, bg="#cccccc")
        separator.pack(side=tk.LEFT, fill=tk.Y)

        right_frame = tk.Frame(self.window, bg="white")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ---- Правая панель: matplotlib ----
        self.fig = Figure(figsize=(9, 6), dpi=100, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ---- Левая панель: параметры ----
        self._build_left_panel(left_frame)

        # Нарисуем пустой (начальный) контур функции
        self._draw_initial_plot()

    # ------------------------------------------------------------------
    # Построение левой панели
    # ------------------------------------------------------------------

    def _build_left_panel(self, frame: tk.Frame):
        bg = "#f4f4f4"

        tk.Label(
            frame,
            text="Параметры алгоритма",
            font=("Arial", 12, "bold"),
            bg=bg,
            anchor="center",
        ).pack(fill=tk.X, padx=12, pady=(16, 8))

        ttk.Separator(frame, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        params_frame = tk.Frame(frame, bg=bg)
        params_frame.pack(fill=tk.X, padx=12, pady=4)

        if self.algo_type == "genetic":
            self._params = self._make_genetic_params(params_frame)
        else:
            self._params = self._make_pso_params(params_frame)

        ttk.Separator(frame, orient="horizontal").pack(fill=tk.X, padx=8, pady=8)

        # Кнопка запуска
        self._run_btn = tk.Button(
            frame,
            text="▶  Запустить",
            command=self._on_run,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            activebackground="#1e8449",
            relief=tk.FLAT,
            cursor="hand2",
            height=2,
        )
        self._run_btn.pack(fill=tk.X, padx=12, pady=6)

        # Прогресс-бар
        self._progress = ttk.Progressbar(frame, mode="indeterminate", length=200)
        self._progress.pack(fill=tk.X, padx=12, pady=4)

        # Статус
        self._status_var = tk.StringVar(value="Готов к запуску")
        tk.Label(
            frame,
            textvariable=self._status_var,
            bg=bg,
            font=("Arial", 10),
            wraplength=250,
            justify=tk.LEFT,
            fg="#555555",
        ).pack(fill=tk.X, padx=12, pady=4)

        # Результат
        ttk.Separator(frame, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)
        self._result_var = tk.StringVar(value="")
        tk.Label(
            frame,
            textvariable=self._result_var,
            bg=bg,
            font=("Arial", 10, "bold"),
            wraplength=250,
            justify=tk.LEFT,
            fg="#2c3e50",
        ).pack(fill=tk.X, padx=12, pady=6)

    def _row(self, frame, label_text, default_value):
        """Добавляет строку «метка + поле ввода» и возвращает StringVar."""
        row = tk.Frame(frame, bg=frame["bg"])
        row.pack(fill=tk.X, pady=3)
        tk.Label(row, text=label_text, bg=frame["bg"], font=("Arial", 10), anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        var = tk.StringVar(value=str(default_value))
        entry = tk.Entry(row, textvariable=var, width=self._ENTRY_WIDTH, font=("Arial", 10))
        entry.pack(side=tk.RIGHT)
        return var

    def _make_genetic_params(self, frame):
        return {
            "population_size": self._row(frame, "Размер популяции:", 1000),
            "max_generations": self._row(frame, "Поколений:", 300),
            "p_crossover":     self._row(frame, "Вер-ть скрещивания:", 0.9),
            "p_mutation":      self._row(frame, "Вер-ть мутации:", 0.3),
            "elitism_count":   self._row(frame, "Элитизм (особей):", 20),
        }

    def _make_pso_params(self, frame):
        return {
            "num_particles":   self._row(frame, "Частиц:", 40),
            "max_iterations":  self._row(frame, "Итераций:", 100),
            "c1":              self._row(frame, "c₁ (когнитивный):", 2.05),
            "c2":              self._row(frame, "c₂ (социальный):", 2.05),
        }

    # ------------------------------------------------------------------
    # Начальный (пустой) контурный график функции
    # ------------------------------------------------------------------

    def _draw_initial_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        X, Y, Z = _build_contour_data(BOUNDS)
        cp = ax.contourf(X, Y, Z, levels=40, cmap="viridis")
        self.fig.colorbar(cp, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Функция f(x, y) — контурная карта", fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Запуск алгоритма
    # ------------------------------------------------------------------

    def _on_run(self):
        if self._running:
            return
        self._running = True
        self._run_btn.config(state=tk.DISABLED)
        self._result_var.set("")
        self._status_var.set("Выполняется…")
        self._progress.start(10)

        thread = threading.Thread(target=self._run_algorithm, daemon=True)
        thread.start()

    def _run_algorithm(self):
        try:
            if self.algo_type == "genetic":
                self._run_genetic()
            else:
                self._run_pso()
        except Exception as exc:
            self.window.after(0, lambda: self._status_var.set(f"Ошибка: {exc}"))
        finally:
            self.window.after(0, self._on_done)

    def _on_done(self):
        self._progress.stop()
        self._run_btn.config(state=tk.NORMAL)
        self._running = False

    # ------------------------------------------------------------------
    # Генетический алгоритм
    # ------------------------------------------------------------------

    def _read_genetic_params(self):
        return {
            "population_size": int(self._params["population_size"].get()),
            "max_generations": int(self._params["max_generations"].get()),
            "p_crossover":     float(self._params["p_crossover"].get()),
            "p_mutation":      float(self._params["p_mutation"].get()),
            "elitism_count":   int(self._params["elitism_count"].get()),
        }

    def _run_genetic(self):
        from geneticAlgorithmWithoutModific import run_genetic_algorithm

        params = self._read_genetic_params()
        total = params["max_generations"]
        self.window.after(0, lambda: self._status_var.set(f"Поколение 0 / {total}"))

        def on_progress(gen, best, mean, bx, by):
            if gen % max(1, total // 20) == 0 or gen == total:
                self.window.after(
                    0,
                    lambda g=gen, b=best, m=mean: self._status_var.set(
                        f"Поколение {g} / {total}\nЛучшее: {b:.6f}\nСреднее: {m:.6f}"
                    ),
                )

        result = run_genetic_algorithm(
            population_size=params["population_size"],
            max_generations=params["max_generations"],
            p_crossover=params["p_crossover"],
            p_mutation=params["p_mutation"],
            elitism_count=params["elitism_count"],
            on_progress=on_progress,
        )
        self.window.after(0, lambda: self._draw_genetic_plots(result, params))

    def _draw_genetic_plots(self, result, params):
        gens = list(range(1, len(result["best_fitness_history"]) + 1))
        best_hist = result["best_fitness_history"]
        mean_hist = result["mean_fitness_history"]
        bx_hist = result["best_x_history"]
        by_hist = result["best_y_history"]

        self.fig.clear()
        self.fig.suptitle("Генетический алгоритм", fontsize=13, fontweight="bold")

        # ---- 1: Кривая сходимости ----
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax1.plot(gens, best_hist, color="#e74c3c", linewidth=1.5, label="Лучшее")
        ax1.plot(gens, mean_hist, color="#3498db", linewidth=1.0, alpha=0.7, label="Среднее")
        ax1.set_title("Кривая сходимости", fontsize=11)
        ax1.set_xlabel("Поколение")
        ax1.set_ylabel("Приспособленность (f)")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.4)

        # ---- 2: Траектория лучшего решения на фоне контура ----
        ax2 = self.fig.add_subplot(1, 2, 2)
        X, Y, Z = _build_contour_data(BOUNDS)
        cp = ax2.contourf(X, Y, Z, levels=40, cmap="viridis", alpha=0.85)
        self.fig.colorbar(cp, ax=ax2, fraction=0.046, pad=0.04)

        # Траектория через каждые несколько поколений
        step = max(1, len(bx_hist) // 50)
        tx = bx_hist[::step]
        ty = by_hist[::step]
        ax2.plot(tx, ty, color="white", linewidth=0.8, alpha=0.6, zorder=2)
        ax2.scatter(tx, ty, c=range(len(tx)), cmap="Reds", s=15, zorder=3, label="Траектория")
        # Финальная точка
        ax2.scatter(
            result["best_x"], result["best_y"],
            marker="*", s=220, color="#f39c12", edgecolors="white",
            linewidths=0.8, zorder=5, label=f"Минимум ({result['best_x']:.3f}, {result['best_y']:.3f})"
        )
        ax2.set_title("Траектория лучшего решения", fontsize=11)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.legend(fontsize=8, loc="upper right")

        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas.draw()

        self._result_var.set(
            f"✅ Готово!\n"
            f"x* = {result['best_x']:.6f}\n"
            f"y* = {result['best_y']:.6f}\n"
            f"f* = {result['best_value']:.8f}"
        )
        self._status_var.set("Завершено.")

    # ------------------------------------------------------------------
    # Роевой алгоритм (PSO)
    # ------------------------------------------------------------------

    def _read_pso_params(self):
        return {
            "num_particles":  int(self._params["num_particles"].get()),
            "max_iterations": int(self._params["max_iterations"].get()),
            "c1":             float(self._params["c1"].get()),
            "c2":             float(self._params["c2"].get()),
        }

    def _run_pso(self):
        from pso import PSO

        params = self._read_pso_params()
        total = params["max_iterations"]

        pso_verbose_step = max(1, total // 10)

        optimizer = PSO(
            func=FUNCTION,
            dimensions=2,
            bounds=BOUNDS,
            num_particles=params["num_particles"],
            max_iterations=params["max_iterations"],
            use_constriction=True,
            c1=params["c1"],
            c2=params["c2"],
        )

        # Запускаем с колбэком через monkey-patch _evaluate_swarm для статуса
        _orig_evaluate = optimizer._evaluate_swarm
        _iteration = [0]

        def _patched_evaluate():
            _orig_evaluate()
            _iteration[0] += 1
            it = _iteration[0]
            if it % max(1, total // 20) == 0 or it == total:
                bv = optimizer.global_best_value
                self.window.after(
                    0,
                    lambda i=it, v=bv: self._status_var.set(
                        f"Итерация {i} / {total}\nЛучшее: {v:.8f}"
                    ),
                )

        optimizer._evaluate_swarm = _patched_evaluate

        best_pos, best_val = optimizer.optimize(verbose=False)
        self.window.after(0, lambda: self._draw_pso_plots(optimizer, best_pos, best_val, params))

    def _draw_pso_plots(self, optimizer, best_pos, best_val, params):
        iterations = list(range(1, len(optimizer.history) + 1))
        history = optimizer.history
        positions_history = optimizer.positions_history

        self.fig.clear()
        self.fig.suptitle("Роевой алгоритм (PSO)", fontsize=13, fontweight="bold")

        X, Y, Z = _build_contour_data(BOUNDS)

        # ---- 1: Кривая сходимости ----
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax1.plot(iterations, history, color="#e67e22", linewidth=1.8, label="Лучшее значение")
        ax1.set_title("Кривая сходимости", fontsize=11)
        ax1.set_xlabel("Итерация")
        ax1.set_ylabel("f (лучшая частица)")
        ax1.grid(True, alpha=0.4)
        ax1.legend(fontsize=9)

        # Аннотация минимума
        min_val = min(history)
        min_iter = history.index(min_val)
        ax1.annotate(
            f"min = {min_val:.4f}",
            xy=(min_iter + 1, min_val),
            xytext=(min_iter + 1 + len(history) * 0.05, min_val * 0.95),
            arrowprops=dict(arrowstyle="->", color="#c0392b"),
            fontsize=8,
            color="#c0392b",
        )

        # ---- 2: Позиции частиц на фоне контура ----
        ax2 = self.fig.add_subplot(1, 2, 2)
        cp = ax2.contourf(X, Y, Z, levels=40, cmap="plasma", alpha=0.8)
        self.fig.colorbar(cp, ax=ax2, fraction=0.046, pad=0.04)

        total = len(positions_history)
        snapshots = [0, total // 4, total // 2, 3 * total // 4, total - 1]
        snapshots = sorted(set(snapshots))
        colors_snap = ["#ecf0f1", "#bdc3c7", "#95a5a6", "#7f8c8d", "#f39c12"]
        sizes_snap  = [20, 25, 30, 40, 60]

        for snap_idx, (snap, col, sz) in enumerate(zip(snapshots, colors_snap, sizes_snap)):
            if snap < total:
                positions = positions_history[snap]
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                label = f"Iter {snap + 1}" if snap_idx < len(snapshots) - 1 else f"Финал"
                ax2.scatter(xs, ys, color=col, s=sz, zorder=3 + snap_idx,
                            edgecolors="none", alpha=0.9, label=label)

        # Финальная лучшая позиция
        ax2.scatter(
            best_pos[0], best_pos[1],
            marker="*", s=260, color="#f1c40f", edgecolors="white",
            linewidths=0.8, zorder=10, label=f"Минимум ({best_pos[0]:.3f}, {best_pos[1]:.3f})"
        )
        ax2.set_title("Позиции частиц (по итерациям)", fontsize=11)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.legend(fontsize=8, loc="upper right")

        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas.draw()

        self._result_var.set(
            f"✅ Готово!\n"
            f"x* = {best_pos[0]:.6f}\n"
            f"y* = {best_pos[1]:.6f}\n"
            f"f* = {best_val:.8f}"
        )
        self._status_var.set("Завершено.")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    SelectionWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
