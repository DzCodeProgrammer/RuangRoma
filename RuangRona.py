"""
RuangRona - studio filter interaktif dengan gerakan tangan
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
import time
from typing import Dict, List, Tuple, Callable

import cv2
import numpy as np


@dataclass
class PipelineConfig:
    cam_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    pinch_ratio: float = 0.35
    pinch_release_ratio: float = 0.50
    smoothing_alpha: float = 0.4
    filter_cooldown_sec: float = 0.15
    mode_cooldown_sec: float = 1.2
    fist_ratio: float = 1.15
    output_dir: Path = Path(__file__).resolve().parent / "captures"
    recording_fps: float = 30.0
    filter_strength: float = 1.0


class FilterBank:
    @staticmethod
    def dual_tone(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        out = np.zeros_like(roi)
        out[mask == 255] = (10, 140, 255)
        out[mask == 0] = (180, 30, 220)
        return out

    @staticmethod
    def thermal(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    @staticmethod
    def sketch(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def pixelate(roi: np.ndarray, block_size: int = 14) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def glitch(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        b, g, r = cv2.split(roi)
        shift = random.randint(4, 12)
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        out = cv2.merge([b, g, r])
        for _ in range(2):
            y = random.randint(0, h - 1)
            out[y : y + 1, :] = np.random.randint(0, 255, (1, w, 3), dtype=np.uint8)
        return out

    @staticmethod
    def invert(roi: np.ndarray) -> np.ndarray:
        return 255 - roi

    @staticmethod
    def red_channel(roi: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(roi)
        zeros = np.zeros_like(b)
        return cv2.merge([zeros, zeros, r])

    @staticmethod
    def edge(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        colored = cv2.applyColorMap(edges, cv2.COLORMAP_SUMMER)
        return cv2.bitwise_and(colored, colored, mask=edges)

    @staticmethod
    def blur(roi: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(roi, (25, 25), 0)

    @staticmethod
    def cartoon(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(roi, 9, 250, 250)
        return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    @staticmethod
    def rainbow_wave(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        t = time.time() * 5.0
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        pattern = np.sin((x_coords + y_coords) * 0.05 + t) * 127 + 128
        rainbow = cv2.applyColorMap(pattern.astype(np.uint8), cv2.COLORMAP_HSV)
        return cv2.addWeighted(roi, 0.3, rainbow, 0.7, 0)

    @staticmethod
    def sepia(roi: np.ndarray) -> np.ndarray:
        matrix = np.array([[0.131, 0.534, 0.272],
                           [0.168, 0.686, 0.349],
                           [0.189, 0.769, 0.393]], dtype=np.float32)
        return np.clip(cv2.transform(roi.astype(np.float32), matrix), 0, 255).astype(np.uint8)

    @staticmethod
    def film(roi: np.ndarray) -> np.ndarray:
        warm = FilterBank.sepia(roi)
        faded = cv2.addWeighted(roi, 0.65, warm, 0.35, 10)
        grain = np.random.normal(0, 5, (*roi.shape[:2], 1))
        return np.clip(faded.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    @staticmethod
    def vhs(roi: np.ndarray) -> np.ndarray:
        out = roi.copy()
        out[:, :, 2] = np.roll(out[:, :, 2], 3, axis=1)
        out[::3] = (out[::3].astype(np.float32) * 0.75).astype(np.uint8)
        return out


class GestureLatch:
    """Trigger on press, re-arm only on an observed release (not tracking loss)."""

    def __init__(self, cooldown: float):
        self.held = False
        self.last_trigger = float("-inf")
        self.cooldown = cooldown

    def update(self, pressed: bool, released: bool, now: float) -> bool:
        if released:
            self.held = False
        if pressed and not self.held:
            self.held = True
            if now - self.last_trigger >= self.cooldown:
                self.last_trigger = now
                return True
        return False


class TipSmoother:
    """Match each hand to its previous wrist so detector ordering can change."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.previous = []

    def update(self, hands):
        available = list(self.previous)
        current = []
        for wrist, tips in hands:
            tips = np.asarray(tips, dtype=np.float32)
            if available:
                distances = [np.linalg.norm(wrist - old[0]) for old in available]
                match = int(np.argmin(distances))
                if distances[match] < 160:
                    _, old_tips = available.pop(match)
                    tips = self.alpha * tips + (1 - self.alpha) * old_tips
            current.append((wrist, tips))
        self.previous = current
        return [np.rint(tips).astype(int).tolist() for _, tips in current]


class VideoRecorder:
    """Resample frames to a fixed output rate to preserve wall-clock duration."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.writer = None
        self.path = None
        self.started = 0.0
        self.frames = 0
        self.last_frame = None

    @property
    def active(self):
        return self.writer is not None

    def start(self, frame, now):
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cfg.output_dir / datetime.now().strftime("ruangrona_%Y%m%d_%H%M%S_%f.avi")
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"),
                                 self.cfg.recording_fps, (w, h))
        if not writer.isOpened():
            writer.release()
            raise OSError("Encoder MJPG tidak tersedia; rekaman gagal dimulai.")
        self.writer = writer
        self.started = now
        self.frames = 0
        self.last_frame = frame.copy()
        self.write(frame, now)

    def write(self, frame, now):
        if not self.active:
            return
        target = int(max(0, now - self.started) * self.cfg.recording_fps) + 1
        while self.frames < target:
            self.writer.write(self.last_frame if self.frames < target - 1 else frame)
            self.frames += 1
        self.last_frame = frame.copy()

    def stop(self, now=None):
        if self.active:
            try:
                if now is not None:
                    self.write(self.last_frame, now)
            finally:
                self.writer.release()
                self.writer = None
                self.last_frame = None
        return self.path


class GeometryUtils:
    @staticmethod
    def euclidean_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def is_fist_closed(landmarks, w: int, h: int, threshold: float) -> bool:
        wrist = np.array([landmarks[0].x * w, landmarks[0].y * h])
        tips = [8, 12, 16, 20]
        distances = [np.linalg.norm(np.array([landmarks[t].x * w, landmarks[t].y * h]) - wrist) for t in tips]
        palm = max(1.0, np.linalg.norm(np.array([landmarks[9].x * w, landmarks[9].y * h]) - wrist))
        return float(np.mean(distances)) / palm < threshold

    @staticmethod
    def is_hand_rotated(thumb: Tuple[int, int], index: Tuple[int, int]) -> bool:
        dx, dy = index[0] - thumb[0], index[1] - thumb[1]
        return (dy > 25) or (abs(dx) > abs(dy) * 1.1)

    @staticmethod
    def sort_quad_clean(pts: List[Tuple[int, int]]) -> np.ndarray:
        arr = np.array(pts, dtype=np.float32)
        x_sorted = arr[np.argsort(arr[:, 0]), :]
        leftmost = x_sorted[:2, :][np.argsort(x_sorted[:2, 1]), :]
        rightmost = x_sorted[2:, :][np.argsort(x_sorted[2:, 1]), :]
        return np.array([leftmost[0], rightmost[0], rightmost[1], leftmost[1]], dtype=np.int32)

    @staticmethod
    def sort_quad_bowtie(pts: List[Tuple[int, int]]) -> np.ndarray:
        arr = np.array(pts, dtype=np.float32)
        x_sorted = arr[np.argsort(arr[:, 0]), :]
        leftmost = x_sorted[:2, :][np.argsort(x_sorted[:2, 1]), :]
        rightmost = x_sorted[2:, :][np.argsort(x_sorted[2:, 1]), :]
        return np.array([leftmost[0], rightmost[1], rightmost[0], leftmost[1]], dtype=np.int32)


class PortalProcessor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.filters: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "sepia": FilterBank.sepia,
            "film": FilterBank.film,
            "vhs": FilterBank.vhs,
            "dual-tone": FilterBank.dual_tone,
            "thermal": FilterBank.thermal,
            "sketch": FilterBank.sketch,
            "pixelate": FilterBank.pixelate,
            "glitch": FilterBank.glitch,
            "invert": FilterBank.invert,
            "red-channel": FilterBank.red_channel,
            "edge": FilterBank.edge,
            "blur": FilterBank.blur,
            "cartoon": FilterBank.cartoon,
            "rainbow-wave": FilterBank.rainbow_wave,
        }
        self.filter_keys = list(self.filters.keys())
        self.active_filter_idx = 0
        self.dual_portal = False

        self.pinch = GestureLatch(cfg.filter_cooldown_sec)
        self.fists = GestureLatch(cfg.mode_cooldown_sec)
        self.smoother = TipSmoother(cfg.smoothing_alpha)
        self.show_landmarks = False
        self.show_hud = True
        self.hand_count = 0
        self.is_bowtie = False
        self.fps = 0.0
        self.last_frame_time = None
        self.notice = ""
        self.notice_until = 0.0

        import mediapipe as mp

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        )

    @property
    def current_filter_name(self) -> str:
        return self.filter_keys[self.active_filter_idx]

    @property
    def secondary_filter_name(self) -> str:
        return self.filter_keys[(self.active_filter_idx + 1) % len(self.filter_keys)]

    def cycle_filter(self, step: int = 1) -> None:
        self.active_filter_idx = (self.active_filter_idx + step) % len(self.filter_keys)

    def render_portal(self, frame: np.ndarray, pts: List[Tuple[int, int]], filter_key: str) -> np.ndarray:
        poly = np.array(pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        right, bottom = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
        x, y = max(0, x), max(0, y)
        w, h = right - x, bottom - y

        if w <= 10 or h <= 10:
            return frame

        roi = frame[y : y + h, x : x + w].copy()
        processed_roi = self.filters[filter_key](roi)
        if self.cfg.filter_strength < 1.0:
            processed_roi = cv2.addWeighted(
                roi, 1.0 - self.cfg.filter_strength,
                processed_roi, self.cfg.filter_strength, 0,
            )

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly - [x, y]], 255)
        mask_3c = cv2.merge([mask, mask, mask])

        bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3c))
        fg = cv2.bitwise_and(processed_roi, mask_3c)
        frame[y : y + h, x : x + w] = cv2.add(bg, fg)

        cv2.polylines(frame, [poly], isClosed=True, color=(130, 225, 245), thickness=2, lineType=cv2.LINE_AA)
        return frame

    def process_frame(self, frame: np.ndarray, gestures_enabled: bool = True) -> np.ndarray:
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.detector.process(rgb)
        now = time.monotonic()
        if self.last_frame_time is not None:
            instant_fps = 1.0 / max(now - self.last_frame_time, 1e-6)
            self.fps = instant_fps if not self.fps else self.fps * 0.9 + instant_fps * 0.1
        self.last_frame_time = now

        hands = []
        pinch_ratios = []
        fist_count = 0
        is_bowtie = False

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                if self.show_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)

                lm = hand_lm.landmark
                tips = [(int(lm[i].x * self.cfg.frame_width), int(lm[i].y * self.cfg.frame_height)) for i in [4, 8, 12, 16, 20]]
                wrist = np.array([lm[0].x * self.cfg.frame_width, lm[0].y * self.cfg.frame_height])
                hands.append((wrist, tips))
                palm = max(1.0, np.linalg.norm(np.array([
                    lm[9].x * self.cfg.frame_width, lm[9].y * self.cfg.frame_height]) - wrist))
                pinch_ratios.append(GeometryUtils.euclidean_dist(tips[0], tips[4]) / palm)
                if GeometryUtils.is_fist_closed(lm, self.cfg.frame_width, self.cfg.frame_height, self.cfg.fist_ratio):
                    fist_count += 1

        all_hand_tips = self.smoother.update(hands)
        self.hand_count = len(hands)
        # Two fists take priority over a pinch that may occur while closing fingers.
        dual_fist = fist_count == 2
        if gestures_enabled:
            if self.fists.update(dual_fist, len(hands) == 2 and fist_count == 0, now):
                self.dual_portal = not self.dual_portal
            if dual_fist:
                self.pinch.held = True
            elif self.pinch.update(
                bool(pinch_ratios) and min(pinch_ratios) < self.cfg.pinch_ratio,
                bool(pinch_ratios) and min(pinch_ratios) > self.cfg.pinch_release_ratio,
                now,
            ):
                self.cycle_filter(1)
        else:
            # A held gesture must be released after the photobooth session.
            self.pinch.held = True
            self.fists.held = True

        if all_hand_tips:
            if self.dual_portal:
                if len(all_hand_tips) == 2:
                    t1, t2 = all_hand_tips[0], all_hand_tips[1]
                    frame = self.render_portal(frame, [t1[0], t1[1], t1[2], t2[2], t2[1], t2[0]], self.current_filter_name)
                    frame = self.render_portal(frame, [t1[2], t1[3], t1[4], t2[4], t2[3], t2[2]], self.secondary_filter_name)
                elif len(all_hand_tips) == 1:
                    frame = self.render_portal(frame, all_hand_tips[0], self.current_filter_name)
            else:
                if len(all_hand_tips) == 2:
                    corners = [all_hand_tips[0][0], all_hand_tips[0][1], all_hand_tips[1][0], all_hand_tips[1][1]]
                    if GeometryUtils.is_hand_rotated(corners[0], corners[1]) or GeometryUtils.is_hand_rotated(corners[2], corners[3]):
                        quad = GeometryUtils.sort_quad_bowtie(corners)
                        is_bowtie = True
                    else:
                        quad = GeometryUtils.sort_quad_clean(corners)
                    frame = self.render_portal(frame, quad, self.current_filter_name)
                elif len(all_hand_tips) == 1:
                    t = all_hand_tips[0]
                    frame = self.render_portal(frame, [t[0], t[1], t[2], t[4]], self.current_filter_name)

        self.is_bowtie = is_bowtie
        return frame

    def notify(self, message: str):
        self.notice = message
        self.notice_until = time.monotonic() + 4.0
        print(message)

    def draw_hud(self, frame: np.ndarray, recorder: VideoRecorder) -> None:
        now = time.monotonic()
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.show_hud:
            cv2.rectangle(frame, (0, 0), (w, 85), (25, 23, 22), -1)
            cv2.rectangle(frame, (0, h - 62), (w, h), (25, 23, 22), -1)
            cv2.putText(frame, "RUANGRONA", (18, 29), font, 0.7, (130, 225, 245), 2, cv2.LINE_AA)
            mode = "DUAL PORTAL" if self.dual_portal else ("BOWTIE" if self.is_bowtie else "PORTAL")
            filters = self.current_filter_name.upper()
            if self.dual_portal:
                filters += " + " + self.secondary_filter_name.upper()
            cv2.putText(frame, f"{filters}  |  {mode}", (18, 62), font, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{self.fps:.0f} FPS  |  {self.hand_count} tangan", (w - 215, 28), font, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, "N/P Filter   C Mode   S Foto   R Rekam   H Panel   L Landmark   Q Keluar", (18, h - 37), font, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
            hint = "Tunjukkan tangan untuk membuka portal" if not self.hand_count else "Pinch jempol-kelingking: filter | Dua kepalan: mode | Lepas untuk mengulang"
            cv2.putText(frame, hint, (18, h - 14), font, 0.43, (205, 215, 220), 1, cv2.LINE_AA)
        # Recording and save/error feedback stay visible even with the panel hidden.
        if recorder.active:
            seconds = int(now - recorder.started)
            cv2.rectangle(frame, (w - 215, 42), (w - 12, 77), (25, 23, 22), -1)
            cv2.circle(frame, (w - 198, 59), 5, (80, 80, 255), -1)
            cv2.putText(frame, f"REC {seconds // 60:02d}:{seconds % 60:02d}", (w - 183, 66), font, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        if self.notice and now < self.notice_until:
            cv2.rectangle(frame, (12, 96), (w - 12, 135), (25, 23, 22), -1)
            cv2.putText(frame, self.notice, (24, 122), font, 0.48, (245, 245, 245), 1, cv2.LINE_AA)


def main() -> None:
    from studio import CameraSource, Studio

    cfg = PipelineConfig()
    recorder = VideoRecorder(cfg)
    processor = None
    source = CameraSource(cfg)
    studio = None
    try:
        source.install(source.prepare(cfg.cam_index), cfg.cam_index)
        processor = PortalProcessor(cfg)
        studio = Studio(processor, recorder, source)
        window = "RuangRona Studio"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 800)
        cv2.setMouseCallback(window, studio.queue)

        while True:
            ret, frame = source.cap.read()
            if not ret:
                studio.notify("Kamera berhenti mengirim gambar. Aplikasi ditutup.", True)
                break

            now = time.monotonic()
            gestures_enabled = not studio.booth.active and not studio.editing
            clean_frame = processor.process_frame(frame, gestures_enabled=gestures_enabled)
            recorder.write(clean_frame, now)
            studio.update(clean_frame, now)
            cv2.imshow(window, studio.draw(clean_frame, now))

            key = cv2.waitKey(1)
            if studio.key(key, clean_frame, now):
                break
            while studio.actions:
                studio.action(studio.actions.pop(0), clean_frame, now)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except (OSError, cv2.error) as exc:
        print(f"[ERROR] Kamera tidak tersedia: {exc}")
    finally:
        try:
            recorder.stop(time.monotonic())
        finally:
            if studio is not None:
                studio.close()
            source.close()
            if processor is not None:
                processor.detector.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
