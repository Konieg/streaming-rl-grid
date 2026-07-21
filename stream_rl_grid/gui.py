"""Tkinter control panel for configuring, training, saving, and inspecting the agent."""

import csv
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config import (
    ALGORITHMS, REPRESENTATIONS, AgentConfig, AppConfig, EnvironmentConfig,
    PROFILES, TrainingConfig, WIND_CHOICES,
)
from .environment import ContinualWindyGridWorld
from .trainer import Trainer


Coord = Tuple[int, int]


class TrainingPanel:
    def __init__(self, root: tk.Tk, fixed_steps: int = 0):
        if int(fixed_steps) < 0:
            raise ValueError("fixed_steps must be non-negative.")
        self.root = root
        self.root.title("Streaming RL Algorithms - Continual Windy Grid")
        self.root.geometry("1450x880")
        self.base_dir = Path(__file__).resolve().parents[1]
        self.trainer: Optional[Trainer] = None
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.save_event = threading.Event()
        self.messages: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self._snapshot_lock = threading.Lock()
        self._pending_snapshot: Optional[Dict[str, Any]] = None
        self.preview_maps: Optional[List[Set[Coord]]] = None
        self.preview_context = 0
        self.selected_obstacle: Optional[Coord] = None
        self.last_snapshot: Optional[Dict[str, Any]] = None
        self.fixed_steps = int(fixed_steps)
        self.target_step: Optional[int] = None
        self._canvas_geometry = (0.0, 0.0, 1.0)
        self._grid_shape: Optional[Tuple[int, int]] = None
        self._grid_geometry: Optional[Tuple[float, float, float, int, int]] = None
        self._grid_cells: Dict[Coord, int] = {}
        self._grid_cell_fills: Dict[Coord, str] = {}
        self._policy_lines: Dict[Tuple[int, int, int], int] = {}
        self._policy_stay: Dict[Coord, int] = {}
        self._grid_overlays: Dict[str, int] = {}

        self.variables: Dict[str, tk.Variable] = {}
        self.metric_labels: Dict[str, ttk.Label] = {}
        self._build_layout()
        self._set_defaults(AppConfig())
        self.generate_preview(use_configured_coordinates=True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_messages)

    def _build_layout(self) -> None:
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        controls = ttk.Frame(outer, width=380)
        display = ttk.Frame(outer)
        outer.add(controls, weight=0)
        outer.add(display, weight=1)

        notebook = ttk.Notebook(controls)
        notebook.pack(fill=tk.BOTH, expand=True)
        env_tab, agent_tab, run_tab = ttk.Frame(notebook), ttk.Frame(notebook), ttk.Frame(notebook)
        notebook.add(env_tab, text="Environment")
        notebook.add(agent_tab, text="Agent")
        notebook.add(run_tab, text="Training")

        self._add_combo(env_tab, "Profile", "profile", PROFILES, 0)
        self._add_entry(env_tab, "Grid width", "width", 1)
        self._add_entry(env_tab, "Grid height", "height", 2)
        self._add_entry(env_tab, "Obstacle count", "obstacle_count", 3)
        self._add_entry(env_tab, "Obstacles (x,y; ...)", "obstacle_coordinates", 4)
        self._add_entry(env_tab, "Start (x,y)", "start_position", 5)
        self._add_entry(env_tab, "Goal (x,y)", "goal_position", 6)
        self._add_combo(env_tab, "Wind direction", "manual_wind_direction", WIND_CHOICES, 7)
        self._add_entry(env_tab, "Wind strength (0-1)", "w_strength", 8)
        self._add_entry(env_tab, "Context maps", "num_contexts", 9)
        self._add_entry(env_tab, "Seed", "seed", 10)
        self._add_entry(env_tab, "Goal reward", "reward_goal", 11)
        self._add_entry(env_tab, "Collision reward", "reward_collision", 12)
        self._add_entry(env_tab, "Step reward", "reward_step", 13)
        self._add_entry(env_tab, "Wind/reward period", "wind_period", 14)
        self._add_entry(env_tab, "Goal move interval", "target_move_interval", 15)
        self._add_entry(env_tab, "Context switch interval", "context_switch_interval", 16)
        preview_row = ttk.Frame(env_tab)
        preview_row.grid(row=17, column=0, columnspan=2, sticky="ew", padx=6, pady=8)
        ttk.Button(preview_row, text="Generate maps", command=self.generate_preview).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(preview_row, text="Prev map", command=lambda: self._change_preview_context(-1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(preview_row, text="Next map", command=lambda: self._change_preview_context(1)).pack(side=tk.LEFT)

        self._add_combo(agent_tab, "Algorithm", "algorithm", ALGORITHMS, 0)
        self._add_combo(agent_tab, "Value representation", "representation", REPRESENTATIONS, 1)
        self._add_entry(agent_tab, "Lambda", "lambda_", 2)
        self._add_entry(agent_tab, "Fixed epsilon", "epsilon", 3)
        self._add_entry(agent_tab, "TIDBD theta", "theta", 4)
        self._add_entry(agent_tab, "Initial effective step size", "effective_initial_step", 5)
        self._add_entry(agent_tab, "Reward-rate step", "reward_rate_step", 6)
        self._add_entry(agent_tab, "Beta minimum", "beta_min", 7)
        self._add_entry(agent_tab, "Beta maximum", "beta_max", 8)
        self._add_entry(agent_tab, "Adaptive epsilon kappa", "adaptive_epsilon_kappa", 9)
        self._add_entry(agent_tab, "Adaptive epsilon minimum", "adaptive_epsilon_min", 10)
        self._add_entry(agent_tab, "Adaptive epsilon maximum", "adaptive_epsilon_max", 11)
        self._add_entry(agent_tab, "Adaptive epsilon scale c", "adaptive_epsilon_scale", 12)
        self._add_entry(agent_tab, "Adaptive epsilon u_ref", "adaptive_epsilon_u_ref", 13)

        self._add_entry(run_tab, "Metric window", "metric_window", 0)
        self._add_entry(run_tab, "Chart points", "chart_points", 1)
        self._add_entry(run_tab, "UI update steps", "ui_update_steps", 2)
        self._add_entry(run_tab, "Auto-checkpoint steps", "auto_checkpoint_steps", 3)
        self._add_entry(run_tab, "Checkpoint folder", "checkpoint_dir", 4)
        self._add_entry(run_tab, "Log folder", "log_dir", 5)
        self._add_entry(run_tab, "Adaptation recovery ratio", "adaptation_recovery_ratio", 6)
        self._add_entry(run_tab, "Adaptation recovery window", "adaptation_recovery_window", 7)
        self._add_entry(run_tab, "Adaptation sustain steps", "adaptation_sustain_steps", 8)
        self._add_entry(run_tab, "Adaptation baseline floor", "adaptation_baseline_floor", 9)

        button_box = ttk.LabelFrame(controls, text="Controls")
        button_box.pack(fill=tk.X, pady=(8, 0))
        self.start_button = ttk.Button(button_box, text="Start new training", command=self.start_training)
        self.start_button.grid(row=0, column=0, padx=4, pady=5, sticky="ew")
        self.pause_button = ttk.Button(button_box, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.grid(row=0, column=1, padx=4, pady=5, sticky="ew")
        self.save_button = ttk.Button(button_box, text="Save", command=self.request_save, state=tk.DISABLED)
        self.save_button.grid(row=1, column=0, padx=4, pady=5, sticky="ew")
        self.stop_button = ttk.Button(button_box, text="Stop (discard)", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=1, padx=4, pady=5, sticky="ew")
        ttk.Button(button_box, text="Load checkpoint", command=self.load_training).grid(
            row=2, column=0, columnspan=2, padx=4, pady=5, sticky="ew"
        )
        self.apply_button = ttk.Button(button_box, text="Apply environment now", command=self.apply_live_environment)
        self.apply_button.grid(row=3, column=0, columnspan=2, padx=4, pady=5, sticky="ew")
        ttk.Button(button_box, text="Apply wind now", command=self.apply_live_wind).grid(
            row=4, column=0, columnspan=2, padx=4, pady=5, sticky="ew"
        )
        button_box.columnconfigure(0, weight=1)
        button_box.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.status_var, wraplength=350).pack(fill=tk.X, pady=8)
        self.layout_var = tk.StringVar(value="Environment positions: -")
        ttk.Label(controls, textvariable=self.layout_var, wraplength=350).pack(fill=tk.X, pady=(0, 8))

        top = ttk.Frame(display)
        top.pack(fill=tk.BOTH, expand=True)
        self.grid_canvas = tk.Canvas(top, bg="white", highlightthickness=1, highlightbackground="#888")
        self.grid_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.grid_canvas.bind("<Button-1>", self._on_grid_click)

        metric_frame = ttk.LabelFrame(top, text="Live metrics", width=260)
        metric_frame.pack(side=tk.RIGHT, fill=tk.Y)
        metric_names = [
            ("step", "Step"), ("average_reward", "Window avg reward"),
            ("reward_rate", "Estimated reward rate"), ("goals_per_1000_steps", "Goals / 1000"),
            ("goal_count_window", "Goals in window"),
            ("mean_steps_between_goals", "Mean inter-goal steps"),
            ("collision_rate", "Collision rate"), ("invalid_action_rate", "Invalid-action rate"),
            ("average_reward_estimation_error", "|R-bar - rolling reward|"),
            ("adaptation_delay_median", "Median adaptation delay"),
            ("abs_td_error", "Mean |TD error|"),
            ("epsilon", "Current epsilon"), ("td_error_magnitude", "Smoothed |TD error|"),
            ("alpha_mean", "Mean step size"), ("alpha_max", "Max step size"),
            ("q_parameter_count", "Value parameters"), ("representation", "Representation"),
            ("context_index", "Hidden context (log)"), ("wind_phase", "Wind phase (log)"),
            ("algorithm", "Algorithm"), ("next_action", "Next action"),
        ]
        for row, (key, label) in enumerate(metric_names):
            ttk.Label(metric_frame, text=label + ":").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            value_label = ttk.Label(metric_frame, text="-")
            value_label.grid(row=row, column=1, sticky="e", padx=5, pady=2)
            self.metric_labels[key] = value_label

        self.figure = Figure(figsize=(10, 2.8), dpi=100)
        self.reward_axis = self.figure.add_subplot(121)
        self.diagnostic_axis = self.figure.add_subplot(122)
        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=display)
        self.figure_canvas.get_tk_widget().pack(fill=tk.X, pady=(6, 0))
        self.save_curves_button = ttk.Button(
            display,
            text="Save curves with timestamp",
            command=self.save_curves,
            state=tk.DISABLED,
        )
        self.save_curves_button.pack(anchor=tk.E, padx=4, pady=(3, 0))

    def _add_entry(self, parent: ttk.Frame, label: str, key: str, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        variable = tk.StringVar()
        self.variables[key] = variable
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        parent.columnconfigure(1, weight=1)

    def _add_combo(self, parent: ttk.Frame, label: str, key: str, values, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        variable = tk.StringVar()
        self.variables[key] = variable
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=18).grid(
            row=row, column=1, sticky="ew", padx=6, pady=4
        )

    def _set_defaults(self, config: AppConfig) -> None:
        values = {}
        values.update(config.environment.__dict__)
        values.update(config.agent.__dict__)
        values.update(config.training.__dict__)
        for key, variable in self.variables.items():
            if key in values:
                variable.set("" if values[key] is None else values[key])
        maps = config.environment.context_maps or (
            [config.environment.obstacle_coordinates] if config.environment.obstacle_coordinates else []
        )
        if maps:
            self.preview_maps = [{tuple(point) for point in layout} for layout in maps]
            self.variables["obstacle_coordinates"].set(self._format_obstacles(self.preview_maps[0]))
        if config.environment.start_position is not None:
            self.variables["start_position"].set(self._format_coordinate(tuple(config.environment.start_position)))
        if config.environment.goal_position is not None:
            self.variables["goal_position"].set(self._format_coordinate(tuple(config.environment.goal_position)))

    @staticmethod
    def _parse_coordinate(text: str, label: str) -> Optional[Coord]:
        value = text.strip()
        if not value:
            return None
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2:
            raise ValueError("%s must use x,y format." % label)
        return int(parts[0]), int(parts[1])

    @classmethod
    def _parse_obstacles(cls, text: str) -> List[Coord]:
        if not text.strip():
            return []
        points = [cls._parse_coordinate(item, "Each obstacle") for item in text.split(";") if item.strip()]
        result = [point for point in points if point is not None]
        if len(set(result)) != len(result):
            raise ValueError("Obstacle coordinates must be unique.")
        return result

    @staticmethod
    def _format_coordinate(point: Coord) -> str:
        return "%d,%d" % point

    @classmethod
    def _format_obstacles(cls, points) -> str:
        return "; ".join(cls._format_coordinate(tuple(point)) for point in sorted(points))

    def _read_config(self) -> AppConfig:
        obstacles = self._parse_obstacles(self.variables["obstacle_coordinates"].get())
        obstacle_count = int(self.variables["obstacle_count"].get())
        if obstacles and len(obstacles) != obstacle_count:
            raise ValueError("Obstacle count is %d, but %d coordinates were entered." % (obstacle_count, len(obstacles)))
        profile = self.variables["profile"].get()
        num_contexts = int(self.variables["num_contexts"].get())
        expected_maps = num_contexts if profile in ("hidden_context", "combined") else 1
        env = EnvironmentConfig(
            width=int(self.variables["width"].get()), height=int(self.variables["height"].get()),
            obstacle_count=obstacle_count,
            num_contexts=num_contexts, profile=profile,
            seed=int(self.variables["seed"].get()), reward_goal=float(self.variables["reward_goal"].get()),
            reward_collision=float(self.variables["reward_collision"].get()),
            reward_step=float(self.variables["reward_step"].get()),
            w_strength=float(self.variables["w_strength"].get()),
            wind_period=int(self.variables["wind_period"].get()),
            target_move_interval=int(self.variables["target_move_interval"].get()),
            context_switch_interval=int(self.variables["context_switch_interval"].get()),
            start_position=list(self._parse_coordinate(self.variables["start_position"].get(), "Start"))
            if self.variables["start_position"].get().strip() else None,
            goal_position=list(self._parse_coordinate(self.variables["goal_position"].get(), "Goal"))
            if self.variables["goal_position"].get().strip() else None,
            manual_wind_direction=self.variables["manual_wind_direction"].get(),
            obstacle_coordinates=[list(point) for point in obstacles] if obstacles else None,
        )
        preview_matches = (
            self.preview_maps is not None and len(self.preview_maps) == expected_maps
            and all(len(layout) == env.obstacle_count for layout in self.preview_maps)
            and set(obstacles) == self.preview_maps[self.preview_context % expected_maps]
        )
        if preview_matches:
            env.context_maps = [[list(point) for point in sorted(layout)] for layout in self.preview_maps]
        elif obstacles:
            env.context_maps = [[list(point) for point in obstacles] for _ in range(expected_maps)]
        agent = AgentConfig(
            algorithm=self.variables["algorithm"].get(),
            representation=self.variables["representation"].get(),
            lambda_=float(self.variables["lambda_"].get()),
            epsilon=float(self.variables["epsilon"].get()), theta=float(self.variables["theta"].get()),
            effective_initial_step=float(self.variables["effective_initial_step"].get()),
            reward_rate_step=float(self.variables["reward_rate_step"].get()),
            beta_min=float(self.variables["beta_min"].get()), beta_max=float(self.variables["beta_max"].get()),
            use_tidbd=self.variables["algorithm"].get() in ("tidbd", "expected_sarsa_tidbd"),
            adaptive_epsilon_kappa=float(self.variables["adaptive_epsilon_kappa"].get()),
            adaptive_epsilon_min=float(self.variables["adaptive_epsilon_min"].get()),
            adaptive_epsilon_max=float(self.variables["adaptive_epsilon_max"].get()),
            adaptive_epsilon_scale=float(self.variables["adaptive_epsilon_scale"].get()),
            adaptive_epsilon_u_ref=float(self.variables["adaptive_epsilon_u_ref"].get()),
        )
        training = TrainingConfig(
            metric_window=int(self.variables["metric_window"].get()),
            chart_points=int(self.variables["chart_points"].get()),
            ui_update_steps=int(self.variables["ui_update_steps"].get()),
            auto_checkpoint_steps=int(self.variables["auto_checkpoint_steps"].get()),
            checkpoint_dir=self.variables["checkpoint_dir"].get(), log_dir=self.variables["log_dir"].get(),
            adaptation_recovery_ratio=float(self.variables["adaptation_recovery_ratio"].get()),
            adaptation_recovery_window=int(self.variables["adaptation_recovery_window"].get()),
            adaptation_sustain_steps=int(self.variables["adaptation_sustain_steps"].get()),
            adaptation_baseline_floor=float(self.variables["adaptation_baseline_floor"].get()),
        )
        config = AppConfig(env, agent, training)
        config.validate()
        return config

    def generate_preview(self, use_configured_coordinates: bool = False) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            if not use_configured_coordinates:
                self.preview_maps = None
                self.variables["obstacle_coordinates"].set("")
            config = self._read_config()
            environment = ContinualWindyGridWorld(config.environment)
            self.preview_maps = [set(layout) for layout in environment.context_maps]
            self.preview_context = 0
            self.selected_obstacle = None
            self.variables["obstacle_count"].set(len(self.preview_maps[0]))
            self.variables["obstacle_coordinates"].set(self._format_obstacles(self.preview_maps[0]))
            self.variables["start_position"].set(self._format_coordinate(environment.start_position))
            self.variables["goal_position"].set(self._format_coordinate(environment.goal))
            snapshot = {
                "agent_state": environment.agent_state, "start_position": environment.start_position,
                "goal": environment.goal,
                "obstacles": sorted(self.preview_maps[0]), "dormant_obstacle": None,
                "wind": environment.wind_vector(environment.agent_state), "wind_phase": 0,
                "context_index": 0, "events": ["preview"],
            }
            self._draw_grid(snapshot, config.environment.width, config.environment.height)
            self.status_var.set("Maps generated. Click an obstacle, then a free cell, to relocate it.")
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))

    def _change_preview_context(self, amount: int) -> None:
        if not self.preview_maps:
            return
        self.preview_context = (self.preview_context + amount) % len(self.preview_maps)
        self.variables["obstacle_coordinates"].set(
            self._format_obstacles(self.preview_maps[self.preview_context])
        )
        config = self._read_config()
        snapshot = {
            "agent_state": (-1, -1), "start_position": self._parse_coordinate(
                self.variables["start_position"].get(), "Start"
            ) or (-1, -1),
            "goal": self._parse_coordinate(self.variables["goal_position"].get(), "Goal") or (-1, -1),
            "obstacles": sorted(self.preview_maps[self.preview_context]), "dormant_obstacle": None,
            "wind": (0, 0), "wind_phase": 0, "context_index": self.preview_context,
            "events": ["map preview"],
        }
        self.selected_obstacle = None
        self._draw_grid(snapshot, config.environment.width, config.environment.height)

    def _on_grid_click(self, event: tk.Event) -> None:
        if self.worker and self.worker.is_alive() and not self.pause_event.is_set() or not self.preview_maps:
            return
        ox, oy, cell = self._canvas_geometry
        if cell <= 0:
            return
        x, y = int((event.x - ox) // cell), int((event.y - oy) // cell)
        config = self._read_config()
        if not (0 <= x < config.environment.width and 0 <= y < config.environment.height):
            return
        point = (x, y)
        layout = self.preview_maps[self.preview_context]
        if self.selected_obstacle is None:
            if point in layout:
                self.selected_obstacle = point
                self.status_var.set("Obstacle selected; click a free destination cell.")
        else:
            if point not in layout:
                old = self.selected_obstacle
                layout.remove(old)
                layout.add(point)
                if not self._preview_maps_valid(config.environment):
                    layout.remove(point)
                    layout.add(old)
                    messagebox.showwarning("Disconnected map", "That move would disconnect the legal cells.")
                self.selected_obstacle = None
                self.variables["obstacle_coordinates"].set(self._format_obstacles(layout))
        snapshot = {
            "agent_state": tuple(self.last_snapshot.get("agent_state", (-1, -1))) if self.last_snapshot else (-1, -1),
            "start_position": self._parse_coordinate(self.variables["start_position"].get(), "Start") or (-1, -1),
            "goal": self._parse_coordinate(self.variables["goal_position"].get(), "Goal") or (-1, -1),
            "obstacles": sorted(layout),
            "dormant_obstacle": None, "wind": (0, 0), "wind_phase": 0,
            "context_index": self.preview_context, "events": ["map edited"],
        }
        self._draw_grid(snapshot, config.environment.width, config.environment.height)

    def apply_live_environment(self) -> None:
        try:
            config = self._read_config()
            obstacles = self._parse_obstacles(self.variables["obstacle_coordinates"].get())
            if len(obstacles) != config.environment.obstacle_count:
                raise ValueError("Enter exactly obstacle_count obstacle coordinates before applying.")
            start = self._parse_coordinate(self.variables["start_position"].get(), "Start")
            goal = self._parse_coordinate(self.variables["goal_position"].get(), "Goal")
            if start is None or goal is None:
                raise ValueError("Start and goal coordinates are required.")
            if self.trainer is None or not self.worker or not self.worker.is_alive():
                self.preview_maps = [set(obstacles) for _ in range(
                    config.environment.num_contexts if config.environment.profile in ("hidden_context", "combined") else 1
                )]
                environment = ContinualWindyGridWorld(config.environment)
                snapshot = {
                    "agent_state": environment.agent_state, "start_position": environment.start_position,
                    "goal": environment.goal, "obstacles": sorted(environment.active_obstacles),
                    "dormant_obstacle": None, "wind": environment.wind_vector(environment.agent_state),
                    "wind_phase": environment.wind_phase, "context_index": 0,
                    "manual_wind_direction": environment.config.manual_wind_direction,
                    "events": ["environment preview updated"],
                }
                self._draw_grid(snapshot, environment.width, environment.height)
                self.status_var.set("Environment preview updated; Start will use these values.")
                return
            if (config.environment.width, config.environment.height) != (
                self.trainer.config.environment.width, self.trainer.config.environment.height
            ):
                raise ValueError("Grid width/height cannot change during training; Stop and start a new run.")
            if not self.pause_event.is_set():
                raise ValueError("Pause training before applying map, start, goal, or profile changes.")
            snapshot = self.trainer.apply_environment_configuration(
                obstacles, start, goal, config.environment.manual_wind_direction, config.environment
            )
            self.preview_maps = [set(layout) for layout in self.trainer.environment.context_maps]
            self.preview_context = self.trainer.environment.context_index
            self._render_snapshot(snapshot)
            self.status_var.set("Environment changes applied at step %d." % self.trainer.step_count)
        except Exception as exc:
            messagebox.showerror("Cannot apply environment", str(exc))

    def apply_live_wind(self) -> None:
        try:
            direction = self.variables["manual_wind_direction"].get()
            strength = float(self.variables["w_strength"].get())
            if not 0.0 <= strength <= 1.0:
                raise ValueError("Wind strength must lie in [0, 1].")
            if self.trainer is None or not self.worker or not self.worker.is_alive():
                self.status_var.set("Wind selection will be used when training starts.")
                return
            snapshot = self.trainer.apply_wind(direction, strength)
            self._render_snapshot(snapshot)
            self.status_var.set("Wind changed immediately at step %d." % self.trainer.step_count)
        except Exception as exc:
            messagebox.showerror("Cannot apply wind", str(exc))

    def _preview_maps_valid(self, env_config: EnvironmentConfig) -> bool:
        try:
            copy_config = EnvironmentConfig(**env_config.__dict__)
            copy_config.context_maps = [[list(point) for point in sorted(layout)] for layout in self.preview_maps or []]
            ContinualWindyGridWorld(copy_config)
            return True
        except ValueError:
            return False

    def start_training(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            self.trainer = Trainer(self._read_config(), base_dir=self.base_dir)
        except Exception as exc:
            messagebox.showerror("Cannot start", str(exc))
            return
        self._launch_worker()

    def _launch_worker(self) -> None:
        self.stop_event.clear()
        self.pause_event.clear()
        self.save_event.clear()
        self._take_pending_snapshot()
        self.target_step = (
            self.trainer.step_count + self.fixed_steps
            if self.trainer is not None and self.fixed_steps > 0
            else None
        )
        self.worker = threading.Thread(target=self._training_loop, name="stream-rl-training", daemon=True)
        self.worker.start()
        self.start_button.configure(state=tk.DISABLED)
        self.pause_button.configure(state=tk.NORMAL, text="Pause")
        self.save_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set(
            "Training to step %d" % self.target_step if self.target_step is not None else "Training"
        )
        if self.trainer is not None:
            self._render_snapshot(self.trainer.snapshot())

    def _training_loop(self) -> None:
        assert self.trainer is not None
        try:
            last_snapshot_time = 0.0
            while not self.stop_event.is_set():
                if self.target_step is not None and self.trainer.step_count >= self.target_step:
                    self._publish_snapshot(self.trainer.snapshot())
                    self.messages.put(("completed", self.target_step))
                    return
                if self.save_event.is_set():
                    path = self.trainer.save()
                    self.save_event.clear()
                    self.messages.put(("saved", path))
                if self.pause_event.is_set():
                    time.sleep(0.05)
                    continue
                batch = self.trainer.config.training.ui_update_steps
                if self.target_step is not None:
                    batch = min(batch, self.target_step - self.trainer.step_count)
                self.trainer.run_steps(
                    batch,
                    stop_event=self.stop_event,
                    with_snapshot=False,
                )
                now = time.monotonic()
                if now - last_snapshot_time >= 0.05:
                    self._publish_snapshot(self.trainer.snapshot())
                    last_snapshot_time = now
                # Give Tk's main thread a scheduling opportunity even for very fast algorithms.
                time.sleep(0.001)
            self._publish_snapshot(self.trainer.snapshot())
            self.messages.put(("stopped", None))
        except Exception as exc:
            try:
                path = self.trainer.save()
            except Exception:
                path = None
            self.messages.put(("error", (exc, path)))

    def _publish_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Publish one GUI frame, replacing any older frame not rendered yet."""
        with self._snapshot_lock:
            self._pending_snapshot = snapshot

    def _take_pending_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._snapshot_lock:
            snapshot = self._pending_snapshot
            self._pending_snapshot = None
            return snapshot

    def toggle_pause(self) -> None:
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_button.configure(text="Pause")
            self.status_var.set("Training")
        else:
            self.pause_event.set()
            self.pause_button.configure(text="Resume")
            if self.trainer is not None:
                snapshot = self.trainer.snapshot()
                self._take_pending_snapshot()
                self._render_snapshot(snapshot)
                self.preview_maps = [set(layout) for layout in self.trainer.environment.context_maps]
                self.preview_context = self.trainer.environment.context_index
                current_layout = self.trainer.environment.context_maps[self.trainer.environment.context_index]
                self.variables["obstacle_count"].set(len(current_layout))
                self.variables["obstacle_coordinates"].set(self._format_obstacles(current_layout))
                self.variables["start_position"].set(self._format_coordinate(snapshot["start_position"]))
                self.variables["goal_position"].set(self._format_coordinate(snapshot["goal"]))
                self.variables["manual_wind_direction"].set(snapshot["manual_wind_direction"])
            self.status_var.set("Paused. Edit the environment, then click Apply environment now.")

    def request_save(self) -> None:
        self.save_event.set()
        self.status_var.set("Checkpoint requested...")

    def save_curves(self) -> Optional[Path]:
        """Save live plots plus offline comparison metrics with one timestamp."""
        if self.trainer is None or self.last_snapshot is None:
            messagebox.showinfo("No curves", "Start or load training before saving curves.")
            return None
        saved_at = datetime.now().astimezone()
        timestamp = saved_at.strftime("%Y%m%d-%H%M%S-%f")
        folder = (
            self.base_dir
            / self.trainer.config.training.log_dir
            / self.trainer.run_id
            / "figures"
        )
        path = folder / ("learning_curves_%s.png" % timestamp)
        timestamp_label = "Saved at %s" % saved_at.isoformat(timespec="seconds")
        annotation = None
        try:
            folder.mkdir(parents=True, exist_ok=True)
            annotation = self.figure.text(
                0.995, 0.01, timestamp_label, ha="right", va="bottom", fontsize=7, color="#555"
            )
            self.figure.savefig(
                path,
                dpi=150,
                bbox_inches="tight",
                metadata={"Title": "Continual RL learning curves", "Creation Time": timestamp_label},
            )
            self._save_performance_exports(folder, timestamp, timestamp_label, self.last_snapshot)
            self.status_var.set("Curves and performance metrics saved: %s" % folder)
        except Exception as exc:
            messagebox.showerror("Cannot save curves", str(exc))
            return None
        finally:
            if annotation is not None:
                annotation.remove()
        return folder

    def _save_performance_exports(
        self, folder: Path, timestamp: str, timestamp_label: str, snapshot: Dict[str, Any]
    ) -> None:
        curves = snapshot["curves"]
        events = list(snapshot.get("adaptation_events", []))
        steps = curves["steps"]
        figure = Figure(figsize=(12, 9), dpi=100)
        axes = figure.subplots(3, 2)
        plots = [
            ("average_reward", "Rolling average reward", "reward"),
            ("goal_count_window", "Goals reached in last W steps", "count"),
            ("mean_inter_goal_time", "Mean inter-goal time", "steps"),
            ("invalid_action_rate", "Invalid-action rate", "rate"),
            ("average_reward_estimation_error", "Average-reward estimation error", "absolute error"),
        ]
        for axis, (key, title, ylabel) in zip(axes.flat[:5], plots):
            axis.plot(steps, curves.get(key, []), label=title)
            axis.set_title(title)
            axis.set_xlabel("stream step")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
        for record in events:
            axes[0, 0].axvline(float(record["event_step"]), color="#d05a4e", alpha=0.25)

        event_axis = axes[2, 1]
        styles = {
            "recovered": ("o", "#2b8c5a"),
            "censored": ("x", "#d07b32"),
            "pending": ("^", "#5078c8"),
            "unavailable": ("s", "#777777"),
        }
        for status, (marker, color) in styles.items():
            selected = [record for record in events if record["status"] == status]
            if not selected:
                continue
            event_axis.scatter(
                [record["event_step"] for record in selected],
                [record["delay"] if record.get("delay") is not None else 0.0 for record in selected],
                marker=marker,
                color=color,
                label=status,
            )
        event_axis.set_title("Adaptation delay by environment event")
        event_axis.set_xlabel("event step")
        event_axis.set_ylabel("delay (steps; 0 means unresolved)")
        event_axis.grid(alpha=0.25)
        if events:
            event_axis.legend(fontsize=8)
        figure.tight_layout(rect=(0, 0.025, 1, 1))
        figure.text(0.995, 0.005, timestamp_label, ha="right", va="bottom", fontsize=7, color="#555")
        figure.savefig(
            folder / ("performance_metrics_%s.png" % timestamp),
            dpi=150,
            bbox_inches="tight",
            metadata={"Title": "Continual RL performance metrics", "Creation Time": timestamp_label},
        )

        curve_keys = [key for key in curves if key != "steps"]
        with (folder / ("performance_metrics_%s.csv" % timestamp)).open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["step"] + curve_keys)
            writer.writeheader()
            for index, step in enumerate(steps):
                row = {"step": step}
                row.update(
                    {
                        key: curves[key][index] if index < len(curves[key]) else float("nan")
                        for key in curve_keys
                    }
                )
                writer.writerow(row)

        event_fields = [
            "event_step", "events", "status", "baseline_reward",
            "recovery_threshold", "end_step", "delay",
        ]
        with (folder / ("adaptation_events_%s.csv" % timestamp)).open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=event_fields)
            writer.writeheader()
            for record in events:
                row = {key: record.get(key) for key in event_fields}
                row["events"] = "|".join(record.get("events", []))
                writer.writerow(row)

        excluded = {"curves", "policy_probabilities", "adaptation_events"}
        summary = {key: value for key, value in snapshot.items() if key not in excluded}
        summary.update(
            {
                "saved_at": timestamp_label.removeprefix("Saved at "),
                "run_id": self.trainer.run_id,
                "metric_window": self.trainer.config.training.metric_window,
                "adaptation_events": events,
            }
        )
        with (folder / ("performance_summary_%s.json" % timestamp)).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(self._json_safe(summary), handle, ensure_ascii=False, indent=2)

    @classmethod
    def _json_safe(cls, value):
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return None
        return value

    def stop_training(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping without saving after the current streaming update...")

    def load_training(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Training active", "Stop the current training before loading a checkpoint.")
            return
        path = filedialog.askopenfilename(
            title="Load exact-continuation checkpoint", initialdir=str(self.base_dir / "checkpoints"),
            filetypes=(("Pickle checkpoint", "*.pkl"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            self.trainer = Trainer.from_checkpoint(path, base_dir=self.base_dir)
            self._set_defaults(self.trainer.config)
            self.preview_maps = [set(layout) for layout in self.trainer.environment.context_maps]
            snapshot = self.trainer.snapshot()
            self._render_snapshot(snapshot)
            self.status_var.set("Checkpoint loaded at step %d. Starting exact continuation." % self.trainer.step_count)
            self._launch_worker()
        except Exception as exc:
            messagebox.showerror("Cannot load checkpoint", str(exc))

    def _poll_messages(self) -> None:
        latest_snapshot = self._take_pending_snapshot()
        completed_step = None
        try:
            # Process a bounded number of control messages so this callback always
            # returns to Tk's event loop even if a very fast algorithm is running.
            for _ in range(32):
                kind, payload = self.messages.get_nowait()
                if kind == "snapshot":
                    latest_snapshot = payload
                elif kind == "saved":
                    self.status_var.set("Saved: %s" % payload)
                elif kind == "stopped":
                    self.status_var.set("Stopped. Current training was not saved.")
                    self._set_idle_controls()
                elif kind == "completed":
                    completed_step = int(payload)
                    self._set_idle_controls()
                elif kind == "error":
                    exc, path = payload
                    self._set_idle_controls()
                    messagebox.showerror("Training paused by safety check", "%s\nDiagnostic checkpoint: %s" % (exc, path))
        except queue.Empty:
            pass
        if latest_snapshot is not None:
            self._render_snapshot(latest_snapshot)
        if completed_step is not None:
            folder = self.save_curves()
            if folder is not None:
                self.status_var.set(
                    "Completed fixed run at step %d. Results auto-saved: %s"
                    % (completed_step, folder)
                )
            else:
                self.status_var.set(
                    "Completed fixed run at step %d, but automatic result export failed."
                    % completed_step
                )
        self.root.after(100, self._poll_messages)

    def _set_idle_controls(self) -> None:
        self.start_button.configure(state=tk.NORMAL)
        self.pause_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.DISABLED)

    def _render_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.last_snapshot = snapshot
        assert self.trainer is not None
        self._draw_grid(snapshot, self.trainer.config.environment.width, self.trainer.config.environment.height)
        self.layout_var.set(
            "Start %s | Agent %s | Goal %s | Obstacles %s | Wind %s (p=%.3g)" % (
                tuple(snapshot.get("start_position", (-1, -1))), tuple(snapshot.get("agent_state", (-1, -1))),
                tuple(snapshot.get("goal", (-1, -1))), snapshot.get("obstacles", []),
                snapshot.get("manual_wind_direction", "auto"), snapshot.get("w_strength", 0.0),
            )
        )
        for key, label in self.metric_labels.items():
            value = snapshot.get(key, "-")
            if isinstance(value, float):
                label.configure(text="%.6g" % value)
            else:
                label.configure(text=str(value))
        curves = snapshot["curves"]
        self.reward_axis.clear()
        self.reward_axis.plot(curves["steps"], curves["average_reward"], label="window reward")
        self.reward_axis.plot(curves["steps"], curves["reward_rate"], label="R-bar", alpha=0.8)
        self.reward_axis.set_title("Average reward")
        self.reward_axis.set_xlabel("stream step")
        self.reward_axis.grid(alpha=0.25)
        self.reward_axis.legend(fontsize=8)
        self.diagnostic_axis.clear()
        self.diagnostic_axis.plot(curves["steps"], curves["abs_td_error"], label="mean |delta|")
        self.diagnostic_axis.plot(curves["steps"], curves["alpha_mean"], label="mean alpha")
        self.diagnostic_axis.plot(curves["steps"], curves.get("epsilon", []), label="epsilon")
        magnitude_curve = curves.get("td_error_magnitude", [])
        if any(value == value for value in magnitude_curve):
            self.diagnostic_axis.plot(curves["steps"], magnitude_curve, label="smoothed |delta|")
        self.diagnostic_axis.set_title("Adaptation diagnostics")
        self.diagnostic_axis.set_xlabel("stream step")
        self.diagnostic_axis.grid(alpha=0.25)
        self.diagnostic_axis.legend(fontsize=8)
        self.figure_canvas.draw_idle()
        self.save_curves_button.configure(state=tk.NORMAL)

    def _draw_grid(self, snapshot: Dict[str, Any], width: int, height: int) -> None:
        canvas = self.grid_canvas
        canvas.update_idletasks()
        available_w = max(300, canvas.winfo_width())
        available_h = max(300, canvas.winfo_height())
        cell = max(8.0, min((available_w - 40) / width, (available_h - 60) / height))
        ox = (available_w - cell * width) / 2.0
        oy = (available_h - cell * height) / 2.0
        self._canvas_geometry = (ox, oy, cell)
        self._ensure_grid_items(width, height)
        geometry = (ox, oy, cell, int(width), int(height))
        geometry_changed = geometry != self._grid_geometry
        self._grid_geometry = geometry
        obstacles = {tuple(p) for p in snapshot.get("obstacles", [])}
        dormant = snapshot.get("dormant_obstacle")
        dormant = None if dormant is None else tuple(dormant)
        for y in range(height):
            for x in range(width):
                point = (x, y)
                fill = "#d9a441" if point in obstacles else "#f7f7f7"
                if point == dormant:
                    fill = "#f4dfad"
                if point == self.selected_obstacle and self.preview_context == snapshot.get("context_index", 0):
                    fill = "#d65ad1"
                x0, y0 = ox + x * cell, oy + y * cell
                item = self._grid_cells[point]
                if geometry_changed:
                    canvas.coords(item, x0, y0, x0 + cell, y0 + cell)
                if self._grid_cell_fills.get(point) != fill:
                    canvas.itemconfigure(item, fill=fill)
                    self._grid_cell_fills[point] = fill

        policies = snapshot.get("policy_probabilities")
        policy_color = "#86d7a1"
        directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
        for y in range(height):
            for x in range(width):
                point = (x, y)
                probabilities = None if not policies or point in obstacles else policies[y][x]
                cx, cy = ox + (x + 0.5) * cell, oy + (y + 0.5) * cell
                for action, (dx, dy) in enumerate(directions):
                    item = self._policy_lines[(x, y, action)]
                    probability = 0.0 if probabilities is None else max(
                        0.0, min(1.0, float(probabilities[action]))
                    )
                    length = 0.42 * cell * probability
                    if length > 0.35:
                        canvas.coords(item, cx, cy, cx + dx * length, cy + dy * length)
                        canvas.itemconfigure(
                            item, state=tk.NORMAL, fill=policy_color,
                            width=max(1, int(cell * 0.028)),
                        )
                    else:
                        canvas.itemconfigure(item, state=tk.HIDDEN)

                stay_item = self._policy_stay[point]
                if probabilities is None:
                    canvas.itemconfigure(stay_item, state=tk.HIDDEN)
                    continue
                stay_probability = max(0.0, min(1.0, float(probabilities[4])))
                radius = 0.11 * cell * stay_probability
                if stay_probability <= 1e-12 or radius < 1.25:
                    dot_radius = max(1.0, min(2.0, cell * 0.025))
                    canvas.coords(
                        stay_item, cx - dot_radius, cy - dot_radius,
                        cx + dot_radius, cy + dot_radius,
                    )
                    canvas.itemconfigure(
                        stay_item, state=tk.NORMAL, fill=policy_color,
                        outline=policy_color, width=1,
                    )
                else:
                    canvas.coords(stay_item, cx - radius, cy - radius, cx + radius, cy + radius)
                    canvas.itemconfigure(
                        stay_item, state=tk.NORMAL, fill="", outline=policy_color,
                        width=max(1, int(cell * 0.025)),
                    )

        start = tuple(snapshot.get("start_position", (-1, -1)))
        start_box = self._grid_overlays["start_box"]
        start_label = self._grid_overlays["start_label"]
        if 0 <= start[0] < width and 0 <= start[1] < height:
            x0, y0 = ox + start[0] * cell, oy + start[1] * cell
            canvas.coords(
                start_box, x0 + 0.08 * cell, y0 + 0.08 * cell,
                x0 + 0.92 * cell, y0 + 0.92 * cell,
            )
            canvas.itemconfigure(
                start_box, state=tk.NORMAL, outline="#28a060",
                width=max(2, int(cell * 0.04)),
            )
            canvas.coords(start_label, x0 + 0.14 * cell, y0 + 0.12 * cell)
            canvas.itemconfigure(start_label, state=tk.NORMAL, text="S", fill="#1b7947")
        else:
            canvas.itemconfigure(start_box, state=tk.HIDDEN)
            canvas.itemconfigure(start_label, state=tk.HIDDEN)

        goal = tuple(snapshot.get("goal", (-1, -1)))
        agent = tuple(snapshot.get("agent_state", (-1, -1)))
        goal_item = self._grid_overlays["goal"]
        if 0 <= goal[0] < width and 0 <= goal[1] < height:
            x0, y0 = ox + goal[0] * cell, oy + goal[1] * cell
            canvas.coords(
                goal_item, x0 + 0.18 * cell, y0 + 0.18 * cell,
                x0 + 0.82 * cell, y0 + 0.82 * cell,
            )
            canvas.itemconfigure(goal_item, state=tk.NORMAL, fill="#32b5d2", outline="")
        else:
            canvas.itemconfigure(goal_item, state=tk.HIDDEN)

        agent_item = self._grid_overlays["agent"]
        if 0 <= agent[0] < width and 0 <= agent[1] < height:
            x0, y0 = ox + agent[0] * cell, oy + agent[1] * cell
            canvas.coords(agent_item, x0 + cell / 2, y0 + cell / 2)
            canvas.itemconfigure(
                agent_item, state=tk.NORMAL, text="A", fill="#2446d8",
                font=("Segoe UI", max(10, int(cell * 0.42)), "bold"),
            )
        else:
            canvas.itemconfigure(agent_item, state=tk.HIDDEN)

        wind = snapshot.get("wind", (0, 0))
        status_item = self._grid_overlays["status"]
        canvas.coords(status_item, ox, max(12, oy - 22))
        canvas.itemconfigure(
            status_item, state=tk.NORMAL, anchor="w",
            text="map %s | wind %s | events: %s" % (
                snapshot.get("context_index", 0), wind, ", ".join(snapshot.get("events", [])) or "-"
            ), fill="#333",
        )

    def _ensure_grid_items(self, width: int, height: int) -> None:
        """Create persistent canvas items once; rebuild only when grid shape changes."""
        shape = (int(width), int(height))
        if self._grid_shape == shape:
            return
        canvas = self.grid_canvas
        canvas.delete("grid-layer")
        self._grid_shape = shape
        self._grid_geometry = None
        self._grid_cells.clear()
        self._grid_cell_fills.clear()
        self._policy_lines.clear()
        self._policy_stay.clear()
        self._grid_overlays.clear()
        for y in range(height):
            for x in range(width):
                point = (x, y)
                self._grid_cells[point] = canvas.create_rectangle(
                    0, 0, 0, 0, outline="#9a9a9a", tags=("grid-layer", "grid-cell")
                )
                for action in range(4):
                    self._policy_lines[(x, y, action)] = canvas.create_line(
                        0, 0, 0, 0, state=tk.HIDDEN, capstyle=tk.ROUND,
                        tags=("grid-layer", "policy"),
                    )
                self._policy_stay[point] = canvas.create_oval(
                    0, 0, 0, 0, state=tk.HIDDEN, tags=("grid-layer", "policy")
                )
        self._grid_overlays = {
            "start_box": canvas.create_rectangle(0, 0, 0, 0, state=tk.HIDDEN, tags=("grid-layer",)),
            "start_label": canvas.create_text(0, 0, state=tk.HIDDEN, anchor="nw", tags=("grid-layer",)),
            "goal": canvas.create_oval(0, 0, 0, 0, state=tk.HIDDEN, tags=("grid-layer",)),
            "agent": canvas.create_text(0, 0, state=tk.HIDDEN, tags=("grid-layer",)),
            "status": canvas.create_text(0, 0, state=tk.HIDDEN, anchor="w", tags=("grid-layer",)),
        }

    def _draw_grid_legacy(self, snapshot: Dict[str, Any], width: int, height: int) -> None:
        self._draw_grid(snapshot, width, height)

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Stop training", "Stop the current training without saving and close?"):
                return
            self.stop_event.set()
            self.worker.join(timeout=5.0)
        self.root.destroy()


def main(fixed_steps: int = 0) -> None:
    root = tk.Tk()
    TrainingPanel(root, fixed_steps=fixed_steps)
    root.mainloop()


if __name__ == "__main__":
    main()
