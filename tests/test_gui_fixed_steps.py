import queue
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from run_gui import build_parser
from stream_rl_grid.gui import TrainingPanel


class FakeTrainer:
    def __init__(self, batch_size=7):
        self.step_count = 0
        self.config = SimpleNamespace(training=SimpleNamespace(ui_update_steps=batch_size))

    def run_steps(self, count, stop_event=None, with_snapshot=False):
        self.step_count += int(count)
        return {}

    def snapshot(self):
        return {"step": self.step_count}


class GuiFixedStepTests(unittest.TestCase):
    def test_run_gui_parser_accepts_fixed_steps(self):
        self.assertEqual(build_parser().parse_args(["--steps", "123"]).steps, 123)

    def test_training_loop_stops_exactly_at_target_without_batch_overshoot(self):
        panel = TrainingPanel.__new__(TrainingPanel)
        panel.trainer = FakeTrainer(batch_size=7)
        panel.target_step = 20
        panel.stop_event = threading.Event()
        panel.pause_event = threading.Event()
        panel.save_event = threading.Event()
        panel.messages = queue.Queue()
        panel._snapshot_lock = threading.Lock()
        panel._pending_snapshot = None

        panel._training_loop()

        self.assertEqual(panel.trainer.step_count, 20)
        messages = []
        while not panel.messages.empty():
            messages.append(panel.messages.get_nowait())
        self.assertIn(("completed", 20), messages)
        self.assertEqual(panel._take_pending_snapshot(), {"step": 20})

    def test_completion_renders_final_snapshot_before_automatic_export(self):
        panel = TrainingPanel.__new__(TrainingPanel)
        panel._snapshot_lock = threading.Lock()
        panel._pending_snapshot = {"step": 20}
        panel.messages = queue.Queue()
        panel.messages.put(("completed", 20))
        panel.last_snapshot = None
        panel.status_var = SimpleNamespace(value="", set=lambda value: setattr(panel.status_var, "value", value))
        panel.root = SimpleNamespace(after=lambda delay, callback: None)
        panel._set_idle_controls = lambda: None
        rendered = []
        exported = []

        def render(snapshot):
            rendered.append(snapshot)
            panel.last_snapshot = snapshot

        def export():
            exported.append(panel.last_snapshot)
            return Path("exports")

        panel._render_snapshot = render
        panel.save_curves = export

        panel._poll_messages()

        self.assertEqual(rendered, [{"step": 20}])
        self.assertEqual(exported, [{"step": 20}])
        self.assertIn("Results auto-saved", panel.status_var.value)


if __name__ == "__main__":
    unittest.main()
