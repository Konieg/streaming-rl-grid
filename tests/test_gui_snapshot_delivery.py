import threading
import unittest

from stream_rl_grid.gui import TrainingPanel


class GuiSnapshotDeliveryTests(unittest.TestCase):
    def test_only_latest_unrendered_snapshot_is_retained(self):
        panel = TrainingPanel.__new__(TrainingPanel)
        panel._snapshot_lock = threading.Lock()
        panel._pending_snapshot = None

        for step in range(10_000):
            panel._publish_snapshot({"step": step})

        self.assertEqual(panel._take_pending_snapshot(), {"step": 9_999})
        self.assertIsNone(panel._take_pending_snapshot())


if __name__ == "__main__":
    unittest.main()
