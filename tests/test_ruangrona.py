import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from RuangRona import FilterBank, GestureLatch, PipelineConfig, PortalProcessor, TipSmoother, VideoRecorder
from studio import PhotoBooth, PresetStore, Studio, save_png


def processor():
    detector = SimpleNamespace(process=lambda _: SimpleNamespace(multi_hand_landmarks=[]))
    mp = SimpleNamespace(solutions=SimpleNamespace(
        hands=SimpleNamespace(Hands=lambda **_: detector), drawing_utils=None))
    with patch.dict('sys.modules', {'mediapipe': mp}):
        return PortalProcessor(PipelineConfig())


class GestureTests(unittest.TestCase):
    def test_hold_and_tracking_loss_do_not_repeat(self):
        latch = GestureLatch(0.15)
        self.assertTrue(latch.update(True, False, 0))
        self.assertFalse(latch.update(True, False, 10))
        self.assertFalse(latch.update(False, False, 11))
        self.assertFalse(latch.update(True, False, 12))
        latch.update(False, True, 13)
        self.assertTrue(latch.update(True, False, 14))

    def test_cooldown_does_not_trigger_later_while_held(self):
        latch = GestureLatch(1.2)
        self.assertTrue(latch.update(True, False, 0))
        latch.update(False, True, 0.1)
        self.assertFalse(latch.update(True, False, 0.2))
        self.assertFalse(latch.update(True, False, 2))
        latch.update(False, True, 3)
        self.assertTrue(latch.update(True, False, 4))

    def test_two_fists_take_priority_and_require_release(self):
        p = processor()
        # All fingertips near wrist, but palm landmark defines hand size.
        lm = [SimpleNamespace(x=0.4, y=0.5) for _ in range(21)]
        lm[9] = SimpleNamespace(x=0.4, y=0.3)
        hand = SimpleNamespace(landmark=lm)
        p.detector.process = lambda _: SimpleNamespace(multi_hand_landmarks=[hand, hand])
        frame = np.zeros((540, 960, 3), np.uint8)
        with patch('RuangRona.time.monotonic', side_effect=[0, 3, 6]):
            p.process_frame(frame)
            self.assertTrue(p.dual_portal)
            p.process_frame(frame)
            self.assertTrue(p.dual_portal)
            self.assertEqual(p.active_filter_idx, 0)
            p.detector.process = lambda _: SimpleNamespace(multi_hand_landmarks=[])
            p.process_frame(frame)
            self.assertTrue(p.fists.held)

    def test_gestures_are_frozen_during_photobooth(self):
        p = processor()
        lm = [SimpleNamespace(x=0.4, y=0.5) for _ in range(21)]
        lm[9] = SimpleNamespace(x=0.4, y=0.3)
        hand = SimpleNamespace(landmark=lm)
        p.detector.process = lambda _: SimpleNamespace(multi_hand_landmarks=[hand, hand])
        p.process_frame(np.zeros((540, 960, 3), np.uint8), gestures_enabled=False)
        self.assertFalse(p.dual_portal)
        self.assertEqual(p.active_filter_idx, 0)
        self.assertTrue(p.pinch.held)
        self.assertTrue(p.fists.held)


class SmoothingTests(unittest.TestCase):
    def test_reordered_hands_keep_their_own_history(self):
        smooth = TipSmoother(0.4)
        a, b = np.array([100, 100]), np.array([700, 100])
        smooth.update([(a, [[100, 100]] * 5), (b, [[700, 100]] * 5)])
        result = smooth.update([(b, [[710, 100]] * 5), (a, [[110, 100]] * 5)])
        self.assertEqual(result[0][0], [704, 100])
        self.assertEqual(result[1][0], [104, 100])

    def test_loss_resets_history(self):
        smooth = TipSmoother(0.4)
        wrist = np.array([100, 100])
        smooth.update([(wrist, [[100, 100]] * 5)])
        smooth.update([])
        self.assertEqual(smooth.update([(wrist, [[140, 100]] * 5)])[0][0], [140, 100])


class RenderTests(unittest.TestCase):
    def test_all_filters_preserve_shape_dtype_and_input(self):
        p = processor()
        for name, function in p.filters.items():
            with self.subTest(filter=name):
                source = np.random.default_rng(7).integers(0, 256, (37, 51, 3), dtype=np.uint8)
                original = source.copy()
                output = function(source)
                self.assertEqual(output.shape, source.shape)
                self.assertEqual(output.dtype, np.uint8)
                np.testing.assert_array_equal(source, original)
        self.assertEqual(len(p.filters), 14)

    def test_offscreen_portal_does_not_change_unrelated_pixels(self):
        p = processor()
        frame = np.full((100, 100, 3), 100, np.uint8)
        result = p.render_portal(frame, [(-20, -20), (40, -20), (40, 40), (-20, 40)], 'invert')
        np.testing.assert_array_equal(result[10, 10], [155] * 3)
        np.testing.assert_array_equal(result[70, 70], [100] * 3)
        p.render_portal(frame, [(200, 200), (240, 200), (240, 240)], 'invert')

    def test_preview_does_not_modify_saved_frame(self):
        p = processor()
        clean = p.process_frame(np.zeros((540, 960, 3), np.uint8))
        preview = clean.copy()
        p.draw_hud(preview, VideoRecorder(p.cfg))
        self.assertFalse(np.array_equal(preview, clean))
        self.assertFalse(clean.any())

    def test_filter_strength_blends_with_original(self):
        p = processor()
        frame = np.full((80, 80, 3), 100, np.uint8)
        points = [(10, 10), (70, 10), (70, 70), (10, 70)]
        p.cfg.filter_strength = 0.0
        result = p.render_portal(frame.copy(), points, 'invert')
        np.testing.assert_array_equal(result[30, 30], [100, 100, 100])
        p.cfg.filter_strength = 0.5
        result = p.render_portal(frame.copy(), points, 'invert')
        np.testing.assert_allclose(result[30, 30], [128, 128, 128], atol=1)


class RecordingTests(unittest.TestCase):
    def test_real_encoder_duration_and_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(PipelineConfig(output_dir=Path(directory), recording_fps=30))
            frame = np.full((48, 64, 3), 120, np.uint8)
            recorder.start(frame, 10)
            recorder.write(frame, 10.05)
            recorder.write(frame, 10.5)
            path = recorder.stop(11)
            self.assertFalse(recorder.active)
            capture = cv2.VideoCapture(str(path))
            try:
                self.assertTrue(capture.isOpened())
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 30)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 31)
                success, decoded = capture.read()
                self.assertTrue(success)
                self.assertEqual(decoded.shape, frame.shape)
            finally:
                capture.release()
            recorder.stop()  # Repeated cleanup is safe.

    def test_encoder_failure_does_not_leave_recording_active(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(PipelineConfig(output_dir=Path(directory)))
            with patch('RuangRona.cv2.VideoWriter') as writer:
                writer.return_value.isOpened.return_value = False
                with self.assertRaises(OSError):
                    recorder.start(np.zeros((48, 64, 3), np.uint8), 0)
                writer.return_value.release.assert_called_once()
            self.assertFalse(recorder.active)


class PhotoBoothTests(unittest.TestCase):
    def test_countdown_captures_once_per_deadline_and_builds_strip(self):
        booth = PhotoBooth()
        frame = np.full((90, 160, 3), 80, np.uint8)
        self.assertTrue(booth.start(10, 2, 'Sage', 'Color story'))
        self.assertFalse(booth.tick(frame, 11.99))
        self.assertEqual(len(booth.frames), 0)
        self.assertFalse(booth.tick(frame, 12))
        self.assertEqual(len(booth.frames), 1)
        # A late UI frame still captures only one image.
        self.assertFalse(booth.tick(frame, 18))
        self.assertEqual(len(booth.frames), 2)
        self.assertFalse(booth.tick(frame, 20))
        self.assertTrue(booth.tick(frame, 22))
        self.assertFalse(booth.active)
        strip = booth.compose()
        self.assertEqual(strip.shape, (1796, 720, 3))

    def test_cancel_removes_partial_session(self):
        booth = PhotoBooth()
        booth.start(0, 2, 'Cream', 'Test')
        booth.tick(np.zeros((20, 20, 3), np.uint8), 2)
        booth.cancel()
        self.assertFalse(booth.active)
        self.assertEqual(booth.frames, [])


class PresetTests(unittest.TestCase):
    def preset(self):
        return {'filter': 'sepia', 'strength': .8, 'smoothing': .4, 'pinch': .35,
                'dual': False, 'theme': 'Cream', 'caption': 'My colors', 'delay': 3}

    def test_round_trip_keeps_other_slots_and_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PresetStore(Path(directory) / 'presets.json')
            filters = ['sepia', 'film']
            store.save('1', self.preset(), filters)
            second = self.preset()
            second['filter'] = 'film'
            store.save('2', second, filters)
            result = store.read(filters)
            self.assertEqual(set(result['slots']), {'1', '2'})
            self.assertEqual(result['selected'], '2')

    def test_invalid_existing_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'presets.json'
            path.write_text('{broken', encoding='utf-8')
            store = PresetStore(path)
            with self.assertRaises(ValueError):
                store.save('1', self.preset(), ['sepia'])
            self.assertEqual(path.read_text(encoding='utf-8'), '{broken')

    def test_png_atomic_writer_produces_readable_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'capture.png'
            source = np.full((31, 43, 3), 117, np.uint8)
            self.assertEqual(save_png(path, source), path)
            decoded = cv2.imread(str(path))
            np.testing.assert_array_equal(decoded, source)

    def test_values_changed_in_studio_can_be_saved_and_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            p = processor()
            p.cfg.output_dir = Path(directory) / 'captures'
            store = PresetStore(Path(directory) / 'presets.json')
            studio = Studio(p, VideoRecorder(p.cfg), SimpleNamespace(index=0), store)
            frame = np.zeros((540, 960, 3), np.uint8)
            try:
                studio.action('strength:-1', frame, 0)
                studio.action('smooth:1', frame, 0)
                studio.action('pinch:1', frame, 0)
                studio.action('save', frame, 0)
                saved = store.read(p.filter_keys)['slots']['1']
                self.assertEqual(saved['strength'], .9)
                self.assertEqual(saved['smoothing'], .5)
                self.assertEqual(saved['pinch'], .4)
            finally:
                studio.close()


if __name__ == '__main__':
    unittest.main()
