import unittest

from stream_rl_grid.gui import TrainingPanel


class FakeCanvas:
    def __init__(self):
        self.next_id = 1
        self.created = []
        self.deleted = []
        self.coordinates = {}
        self.configuration = {}
        self.coordinate_updates = []
        self.configuration_updates = []

    def update_idletasks(self):
        pass

    def winfo_width(self):
        return 600

    def winfo_height(self):
        return 500

    def _create(self, kind, coordinates, options):
        item = self.next_id
        self.next_id += 1
        self.created.append((item, kind))
        self.coordinates[item] = tuple(coordinates)
        self.configuration[item] = dict(options)
        return item

    def create_rectangle(self, *coordinates, **options):
        return self._create("rectangle", coordinates, options)

    def create_line(self, *coordinates, **options):
        return self._create("line", coordinates, options)

    def create_oval(self, *coordinates, **options):
        return self._create("oval", coordinates, options)

    def create_text(self, *coordinates, **options):
        return self._create("text", coordinates, options)

    def delete(self, tag):
        self.deleted.append(tag)

    def coords(self, item, *coordinates):
        self.coordinate_updates.append(item)
        self.coordinates[item] = tuple(coordinates)

    def itemconfigure(self, item, **options):
        self.configuration_updates.append(item)
        self.configuration[item].update(options)


class IncrementalGridDrawingTests(unittest.TestCase):
    def panel(self):
        panel = TrainingPanel.__new__(TrainingPanel)
        panel.grid_canvas = FakeCanvas()
        panel._canvas_geometry = (0.0, 0.0, 1.0)
        panel._grid_shape = None
        panel._grid_geometry = None
        panel._grid_cells = {}
        panel._grid_cell_fills = {}
        panel._policy_lines = {}
        panel._policy_stay = {}
        panel._grid_overlays = {}
        panel.selected_obstacle = None
        panel.preview_context = 0
        return panel

    @staticmethod
    def snapshot(agent_state):
        probabilities = [0.02, 0.92, 0.02, 0.02, 0.02]
        return {
            "obstacles": [],
            "dormant_obstacle": None,
            "context_index": 0,
            "policy_probabilities": [
                [list(probabilities), list(probabilities)],
                [list(probabilities), list(probabilities)],
            ],
            "start_position": (0, 0),
            "goal": (1, 1),
            "agent_state": agent_state,
            "wind": (0, 0),
            "events": [],
        }

    def test_same_grid_updates_items_without_recreating_canvas_objects(self):
        panel = self.panel()
        panel._draw_grid(self.snapshot((0, 0)), 2, 2)
        create_count = len(panel.grid_canvas.created)
        delete_count = len(panel.grid_canvas.deleted)
        agent_item = panel._grid_overlays["agent"]
        first_agent_coordinates = panel.grid_canvas.coordinates[agent_item]
        coordinate_update_count = len(panel.grid_canvas.coordinate_updates)
        configuration_update_count = len(panel.grid_canvas.configuration_updates)

        panel._draw_grid(self.snapshot((1, 0)), 2, 2)

        self.assertEqual(len(panel.grid_canvas.created), create_count)
        self.assertEqual(len(panel.grid_canvas.deleted), delete_count)
        self.assertNotEqual(panel.grid_canvas.coordinates[agent_item], first_agent_coordinates)
        cell_items = set(panel._grid_cells.values())
        self.assertTrue(
            cell_items.isdisjoint(panel.grid_canvas.coordinate_updates[coordinate_update_count:])
        )
        self.assertTrue(
            cell_items.isdisjoint(panel.grid_canvas.configuration_updates[configuration_update_count:])
        )


if __name__ == "__main__":
    unittest.main()
