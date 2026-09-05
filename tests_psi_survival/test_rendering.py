from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

from psi_survival.config import ExperimentConfig
from psi_survival.engine import run_simulation
from psi_survival.render import render_static_plots, render_video
from psi_survival.storage import save_result


class RenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ExperimentConfig(world_seed=31, agent_count=1, trace_level="full")
        cls.temporary = tempfile.TemporaryDirectory(prefix="psi-render-test-")
        cls.root = Path(cls.temporary.name)
        config = replace(base, playback=replace(base.playback, ticks=5, fps=20, heatmap_bins=5))
        cls.result_path = save_result(run_simulation(config), cls.root / "recorded.json.gz")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_display_smoothing_does_not_mutate_recorded_data(self):
        before = hashlib.sha256(self.result_path.read_bytes()).hexdigest()
        render_static_plots(self.result_path, self.root / "ema-01", display_ema_lambda=0.1)
        render_static_plots(self.result_path, self.root / "ema-05", display_ema_lambda=0.5)
        after = hashlib.sha256(self.result_path.read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_video_contains_one_h264_sample_per_post_tick_state(self):
        video = render_video(self.result_path, self.root / "short.mp4")
        self.assertTrue(video.exists())
        self.assertGreater(video.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
