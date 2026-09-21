"""RuangRona studio controls, persistent looks, and non-blocking photobooth."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
THEMES = {
    'Cream': ((234, 243, 250), (43, 50, 60)),
    'Midnight': ((32, 29, 28), (208, 231, 248)),
    'Sage': ((207, 224, 214), (49, 70, 54)),
}
BG = (21, 23, 27)
SURFACE = (32, 35, 40)
FIELD = (44, 48, 54)
TEXT = (237, 240, 245)
MUTED = (169, 178, 189)
ACCENT = (144, 221, 244)
INK = (30, 38, 44)


def text(image, label, xy, scale=0.5, color=TEXT, thickness=1):
    cv2.putText(image, str(label), xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def fit_image(source, width, height):
    h, w = source.shape[:2]
    scale = min(width / w, height / h)
    return cv2.resize(source, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def paste_fit(canvas, source, rect):
    x, y, w, h = rect
    fitted = fit_image(source, w, h)
    fh, fw = fitted.shape[:2]
    canvas[y + (h - fh) // 2:y + (h - fh) // 2 + fh,
           x + (w - fw) // 2:x + (w - fw) // 2 + fw] = fitted


def atomic_bytes(path, data):
    """Write next to the destination, then atomically replace it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.ruangrona-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_png(path, frame):
    ok, encoded = cv2.imencode('.png', frame)
    if not ok:
        raise OSError('PNG encoder gagal')
    atomic_bytes(path, encoded.tobytes())
    return Path(path)


class PresetStore:
    def __init__(self, path=ROOT / 'presets.json'):
        self.path = Path(path)

    @staticmethod
    def validate(value, filter_names):
        if not isinstance(value, dict):
            raise ValueError('Format preset tidak valid')
        expected = {'filter', 'strength', 'smoothing', 'pinch', 'dual', 'theme', 'caption', 'delay'}
        if set(value) != expected:
            raise ValueError('Isi preset tidak lengkap')
        if not isinstance(value['filter'], str) or value['filter'] not in filter_names:
            raise ValueError('Filter preset tidak tersedia')
        for key, lo, hi in [('strength', 0, 1), ('smoothing', 0.1, 1), ('pinch', 0.15, 0.65)]:
            number = value[key]
            if type(number) not in (int, float) or not math.isfinite(number) or not lo <= number <= hi:
                raise ValueError(f'Nilai {key} di luar batas')
        if type(value['dual']) is not bool:
            raise ValueError('Mode preset tidak valid')
        if not isinstance(value['theme'], str) or value['theme'] not in THEMES:
            raise ValueError('Tema preset tidak tersedia')
        caption = value['caption']
        if not isinstance(caption, str) or len(caption) > 28 or any(not 32 <= ord(c) <= 126 for c in caption):
            raise ValueError('Caption harus maksimal 28 karakter ASCII')
        if type(value['delay']) is not int or value['delay'] not in (2, 3, 5):
            raise ValueError('Timer preset tidak valid')
        return dict(value)

    def read(self, filter_names):
        if not self.path.exists():
            return {'version': 1, 'selected': '1', 'slots': {}}
        raw = json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(raw, dict) or type(raw.get('version')) is not int or raw['version'] != 1:
            raise ValueError('Versi preset tidak didukung')
        if raw.get('selected') not in ('1', '2', '3') or not isinstance(raw.get('slots'), dict):
            raise ValueError('Daftar preset tidak valid')
        slots = {}
        for slot, value in raw['slots'].items():
            if slot not in ('1', '2', '3'):
                raise ValueError('Slot preset tidak valid')
            slots[slot] = self.validate(value, filter_names)
        return {'version': 1, 'selected': raw['selected'], 'slots': slots}

    def save(self, slot, value, filter_names):
        if slot not in ('1', '2', '3'):
            raise ValueError('Slot preset tidak valid')
        value = self.validate(value, filter_names)
        # Never silently discard other slots when an existing file is malformed.
        data = self.read(filter_names)
        data['slots'][slot] = value
        data['selected'] = slot
        atomic_bytes(self.path, json.dumps(data, indent=2, allow_nan=False).encode('utf-8'))


class PhotoBooth:
    def __init__(self):
        self.active = False
        self.frames = []
        self.deadline = 0.0
        self.delay = 3
        self.theme = 'Cream'
        self.caption = 'A little moment, a lot of color'
        self.stamp = ''
        self.flash_until = 0.0

    def start(self, now, delay, theme, caption):
        if self.active:
            return False
        self.frames = []
        self.active = True
        self.delay, self.theme, self.caption = delay, theme, caption
        self.stamp = datetime.now().strftime('%d.%m.%Y')
        self.deadline = now + delay
        self.flash_until = 0.0
        return True

    def cancel(self):
        self.active = False
        self.frames = []
        self.flash_until = 0.0

    def tick(self, frame, now):
        if not self.active or now < self.deadline:
            return False
        self.frames.append(frame.copy())
        self.flash_until = now + 0.16
        if len(self.frames) == 4:
            self.active = False
            return True
        # A slow frame never causes several identical photos to be captured at once.
        self.deadline = now + self.delay
        return False

    def compose(self):
        if len(self.frames) != 4:
            raise ValueError('Photobooth membutuhkan empat foto')
        background, foreground = THEMES[self.theme]
        width, photo_w, photo_h = 720, 648, 365
        strip = np.full((1796, width, 3), background, np.uint8)
        text(strip, 'RUANGRONA', (36, 55), 0.8, foreground, 2)
        text(strip, 'FOUR FRAMES / ONE LITTLE STORY', (36, 85), 0.40, foreground)
        for i, frame in enumerate(self.frames):
            y = 112 + i * (photo_h + 16)
            paste_fit(strip, frame, (36, y, photo_w, photo_h))
            cv2.rectangle(strip, (44, y + 8), (78, y + 34), background, -1)
            text(strip, f'{i + 1:02}', (50, y + 27), 0.43, foreground)
        text(strip, self.caption, (36, 1689), 0.57, foreground)
        text(strip, self.stamp + '   /   MADE IN RUANGRONA', (36, 1733), 0.42, foreground)
        return strip


class CameraSource:
    """Open a replacement before releasing the working camera."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.cap = None
        self.index = cfg.cam_index

    def prepare(self, index):
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                raise OSError(f'Kamera {index} tidak tersedia')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise OSError(f'Kamera {index} tidak mengirim gambar')
            return cap
        except BaseException:
            cap.release()
            raise

    def install(self, cap, index):
        old = self.cap
        self.cap, self.index = cap, index
        self.cfg.cam_index = index
        if old is not None:
            old.release()

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class Studio:
    """Immediate-mode OpenCV UI. Callback only queues actions; main loop owns state."""
    def __init__(self, processor, recorder, source, store=None):
        self.p = processor
        self.recorder = recorder
        self.source = source
        self.store = store or PresetStore()
        self.booth = PhotoBooth()
        self.theme, self.caption, self.delay = 'Cream', 'Make room for color', 3
        self.slot = '1'
        self.camera_choice = source.index
        self.tab = 'studio'
        self.focus = 0
        self.buttons = []
        self.actions = []
        self.message = 'Siap. Pilih filter, lalu buat momenmu.'
        self.message_until = 0.0
        self.error = False
        self.editing = False
        self.draft = ''
        self.last_strip = None
        self.review = False
        self.saved_path = None
        self.io_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ruangrona-save')
        self.camera_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ruangrona-camera')
        self.save_future = None
        self.camera_future = None
        self.pending_camera = None
        self.closed = False
        self.restore()

    @property
    def busy(self):
        return self.booth.active or self.save_future is not None or self.camera_future is not None

    def notify(self, message, error=False):
        self.message, self.error = message, error
        self.message_until = time.monotonic() + 5
        print(message)

    def snapshot(self):
        return {'filter': self.p.current_filter_name, 'strength': self.p.cfg.filter_strength,
                'smoothing': self.p.cfg.smoothing_alpha, 'pinch': self.p.cfg.pinch_ratio,
                'dual': self.p.dual_portal, 'theme': self.theme, 'caption': self.caption, 'delay': self.delay}

    def apply(self, value):
        value = self.store.validate(value, self.p.filter_keys)
        self.p.active_filter_idx = self.p.filter_keys.index(value['filter'])
        self.p.cfg.filter_strength = value['strength']
        self.p.cfg.smoothing_alpha = value['smoothing']
        self.p.smoother.alpha = value['smoothing']
        self.p.cfg.pinch_ratio = value['pinch']
        self.p.cfg.pinch_release_ratio = value['pinch'] + 0.15
        self.p.dual_portal = value['dual']
        self.theme, self.caption, self.delay = value['theme'], value['caption'], value['delay']

    def restore(self):
        try:
            data = self.store.read(self.p.filter_keys)
            self.slot = data['selected']
            if self.slot in data['slots']:
                self.apply(data['slots'][self.slot])
                self.notify(f'Preset {self.slot} dipulihkan.')
        except (OSError, ValueError, TypeError) as exc:
            self.notify('Preset tidak terbaca. Gunakan default; periksa presets.json.', True)
            print(exc)

    def queue(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.editing:
            for i, (rect, action, enabled) in enumerate(self.buttons):
                rx, ry, rw, rh = rect
                if rx <= x < rx + rw and ry <= y < ry + rh:
                    self.focus = i
                    if enabled:
                        self.actions.append(action)
                    break

    def submit_png(self, frame, prefix):
        if self.save_future is not None:
            self.notify('Tunggu penyimpanan sebelumnya selesai.')
            return
        path = self.p.cfg.output_dir / datetime.now().strftime(prefix + '_%Y%m%d_%H%M%S_%f.png')
        self.save_future = self.io_pool.submit(save_png, path, frame.copy())
        self.notify('Menyimpan hasil...')

    def poll(self):
        if self.save_future is not None and self.save_future.done():
            task, self.save_future = self.save_future, None
            try:
                self.saved_path = task.result()
                self.notify('Tersimpan: ' + self.saved_path.name)
            except (OSError, cv2.error) as exc:
                self.notify('Gagal menyimpan. Periksa izin/ruang folder captures.', True)
                print(exc)
        if self.camera_future is not None and self.camera_future.done():
            task, self.camera_future = self.camera_future, None
            try:
                self.source.install(task.result(), self.pending_camera)
                self.p.smoother.previous = []
                self.p.last_frame_time = None
                self.notify(f'Kamera {self.source.index} aktif.')
            except (OSError, cv2.error) as exc:
                self.camera_choice = self.source.index
                self.notify('Kamera baru gagal. Kamera sebelumnya tetap dipakai.', True)
                print(exc)

    def update(self, clean, now):
        self.poll()
        if self.booth.tick(clean, now):
            self.last_strip = self.booth.compose()
            self.review = True
            self.submit_png(self.last_strip, 'ruangrona_strip')

    def action(self, action, clean, now):
        if action.startswith('tab:'):
            self.tab = action.split(':')[1]
            self.focus = 0
            return
        if action == 'cancel':
            if self.booth.active:
                self.booth.cancel()
                self.notify('Sesi photobooth dibatalkan.')
            return
        if action == 'review':
            if not self.booth.active and self.last_strip is not None:
                self.review = not self.review
            return
        if action == 'hud':
            self.p.show_hud = not self.p.show_hud
            return
        if self.booth.active:
            self.notify('Sesi berjalan. X untuk batal sebelum mengubah pengaturan.')
            return
        if action == 'photo':
            self.submit_png(clean, 'ruangrona')
        elif action == 'booth':
            if self.busy or self.recorder.active:
                self.notify('Selesaikan rekaman/penyimpanan sebelum photobooth.')
                return
            self.review = False
            self.booth.start(now, self.delay, self.theme, self.caption)
            self.notify('Empat foto dimulai. X untuk batal.')
        elif action == 'record':
            if self.camera_future is not None:
                self.notify('Tunggu perpindahan kamera selesai.')
                return
            try:
                if self.recorder.active:
                    self.recorder.stop(now)
                    self.notify('Video tersimpan di captures/.')
                else:
                    self.review = False
                    self.recorder.start(clean, now)
                    self.notify('Merekam. R untuk selesai.')
            except (OSError, cv2.error) as exc:
                self.recorder.stop()
                self.notify('Rekaman gagal. Periksa folder atau encoder MJPG.', True)
                print(exc)
        elif action == 'camera:apply':
            if self.recorder.active or self.busy:
                self.notify('Selesaikan sesi aktif sebelum mengganti kamera.')
            elif self.camera_choice != self.source.index:
                self.pending_camera = self.camera_choice
                self.camera_future = self.camera_pool.submit(self.source.prepare, self.pending_camera)
                self.notify('Menghubungkan kamera...')
        elif action.startswith('camera:'):
            self.camera_choice = max(0, min(9, self.camera_choice + int(action.split(':')[1])))
        elif action.startswith('filter:'):
            self.p.cycle_filter(int(action.split(':')[1]))
        elif action.startswith('strength:'):
            self.p.cfg.filter_strength = float(round(np.clip(
                self.p.cfg.filter_strength + int(action.split(':')[1]) * .1, 0, 1), 2))
        elif action.startswith('smooth:'):
            self.p.cfg.smoothing_alpha = float(round(np.clip(
                self.p.cfg.smoothing_alpha + int(action.split(':')[1]) * .1, .1, 1), 2))
            self.p.smoother.alpha = self.p.cfg.smoothing_alpha
        elif action.startswith('pinch:'):
            self.p.cfg.pinch_ratio = float(round(np.clip(
                self.p.cfg.pinch_ratio + int(action.split(':')[1]) * .05, .15, .65), 2))
            self.p.cfg.pinch_release_ratio = self.p.cfg.pinch_ratio + .15
        elif action == 'mode':
            self.p.dual_portal = not self.p.dual_portal
        elif action == 'landmarks':
            self.p.show_landmarks = not self.p.show_landmarks
        elif action == 'theme':
            themes = list(THEMES)
            self.theme = themes[(themes.index(self.theme) + 1) % len(themes)]
        elif action == 'delay':
            delays = [2, 3, 5]
            self.delay = delays[(delays.index(self.delay) + 1) % len(delays)]
        elif action == 'caption':
            self.editing, self.draft = True, self.caption
        elif action == 'slot':
            self.slot = str(int(self.slot) % 3 + 1)
        elif action in ('save', 'load'):
            try:
                if action == 'save':
                    self.store.save(self.slot, self.snapshot(), self.p.filter_keys)
                    self.notify(f'Preset {self.slot} disimpan. Dipulihkan saat aplikasi dibuka.')
                else:
                    data = self.store.read(self.p.filter_keys)
                    if self.slot not in data['slots']:
                        self.notify(f'Preset {self.slot} masih kosong. Simpan tampilanmu dahulu.')
                        return
                    self.apply(data['slots'][self.slot])
                    self.notify(f'Preset {self.slot} dimuat.')
            except (OSError, ValueError, TypeError) as exc:
                self.notify('Preset gagal. Periksa izin atau isi presets.json.', True)
                print(exc)
        elif action == 'retry' and self.last_strip is not None:
            self.submit_png(self.last_strip, 'ruangrona_strip')

    def key(self, key, clean, now):
        if key == -1:
            return False
        key &= 0xFF
        if self.editing:
            if key == 27:
                self.editing = False
            elif key in (10, 13):
                self.caption, self.editing = self.draft.strip(), False
            elif key in (8, 127):
                self.draft = self.draft[:-1]
            elif 32 <= key <= 126 and len(self.draft) < 28:
                self.draft += chr(key)
            return False
        if key == 27 and self.booth.active:
            self.action('cancel', clean, now)
            return False
        if key in (ord('q'), 27):
            return True
        if key == 9 and self.buttons:
            for step in range(1, len(self.buttons) + 1):
                index = (self.focus + step) % len(self.buttons)
                if self.buttons[index][2]:
                    self.focus = index
                    break
        elif key in (10, 13, 32) and self.buttons:
            _, action, enabled = self.buttons[self.focus % len(self.buttons)]
            if enabled:
                self.action(action, clean, now)
        else:
            shortcuts = {'n': 'filter:1', 'p': 'filter:-1', 'c': 'mode', 's': 'photo', 'r': 'record',
                         'b': 'booth', 'x': 'cancel', 'v': 'review', 'h': 'hud', 'l': 'landmarks',
                         '1': 'tab:studio', '2': 'tab:booth', 'k': 'save', 'o': 'load'}
            if chr(key) in shortcuts:
                self.action(shortcuts[chr(key)], clean, now)
        return False

    def button(self, canvas, label, rect, action, enabled=True, primary=False):
        index = len(self.buttons)
        self.buttons.append((rect, action, enabled))
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), ACCENT if primary and enabled else FIELD, -1)
        if index == self.focus:
            cv2.rectangle(canvas, (x - 2, y - 2), (x + w + 2, y + h + 2), ACCENT, 1)
        color = INK if primary and enabled else TEXT if enabled else (112, 120, 130)
        scale = .47
        while cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > w - 16 and scale > .25:
            scale -= .02
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        text(canvas, label, (x + (w - tw) // 2, y + h // 2 + 5), scale, color)

    def stepper(self, canvas, title, value, y, prefix):
        text(canvas, title, (968, y), .43, MUTED)
        self.button(canvas, '-', (968, y + 12, 38, 36), prefix + ':-1', not self.booth.active)
        text(canvas, value, (1018, y + 36), .49)
        self.button(canvas, '+', (1202, y + 12, 38, 36), prefix + ':1', not self.booth.active)

    def draw(self, clean, now):
        canvas = np.full((800, 1280, 3), BG, np.uint8)
        self.buttons = []
        text(canvas, 'RUANGRONA', (24, 40), .85, ACCENT, 2)
        text(canvas, 'YOUR LITTLE COLOR STUDIO', (25, 65), .36, MUTED)
        text(canvas, f'{self.p.fps:.0f} FPS   /   {self.p.hand_count} TANGAN', (716, 42), .43, MUTED)
        cv2.rectangle(canvas, (24, 92), (920, 596), SURFACE, -1)
        preview = clean.copy()
        if self.review and self.last_strip is not None:
            paste_fit(canvas, self.last_strip, (24, 92, 896, 504))
            text(canvas, 'HASIL PHOTOBOOTH', (44, 121), .44, ACCENT)
            text(canvas, 'V untuk kembali ke kamera', (44, 145), .4, MUTED)
        else:
            paste_fit(canvas, preview, (24, 92, 896, 504))
            if self.p.show_hud:
                cv2.rectangle(canvas, (36, 104), (440, 138), BG, -1)
                mode = 'DUAL' if self.p.dual_portal else 'PORTAL'
                text(canvas, f'{self.p.current_filter_name.upper()}  /  {mode}  /  {self.p.cfg.filter_strength:.0%}', (48, 127), .44)
            if self.booth.active:
                remaining = max(1, math.ceil(self.booth.deadline - now))
                cv2.rectangle(canvas, (354, 266), (586, 411), BG, -1)
                text(canvas, str(remaining), (443, 347), 2.4, ACCENT, 3)
                text(canvas, f'FOTO {len(self.booth.frames) + 1} DARI 4', (401, 385), .5)
            if now < self.booth.flash_until:
                cv2.rectangle(canvas, (26, 94), (918, 594), TEXT, 5)
        if self.recorder.active:
            elapsed = int(now - self.recorder.started)
            cv2.rectangle(canvas, (738, 104), (906, 138), BG, -1)
            text(canvas, f'REC  {elapsed // 60:02}:{elapsed % 60:02}', (755, 127), .5, (142, 151, 255))
        for i in range(4):
            x = 24 + i * 124
            cv2.rectangle(canvas, (x, 612), (x + 112, 675), SURFACE, -1)
            if i < len(self.booth.frames):
                paste_fit(canvas, self.booth.frames[i], (x, 612, 112, 63))
            else:
                text(canvas, f'0{i + 1}', (x + 43, 651), .53, MUTED)
        self.button(canvas, 'Kamera / hasil [V]', (545, 622, 180, 42), 'review', self.last_strip is not None and not self.booth.active)
        self.button(canvas, 'Simpan strip lagi', (740, 622, 180, 42), 'retry', self.last_strip is not None and not self.busy)
        self.button(canvas, 'Foto [S]', (24, 699, 170, 48), 'photo', not self.booth.active and self.save_future is None)
        self.button(canvas, 'Stop rekam [R]' if self.recorder.active else 'Rekam [R]', (208, 699, 184, 48), 'record', not self.booth.active and self.camera_future is None)
        self.button(canvas, 'Batal sesi [X]' if self.booth.active else 'Photobooth [B]', (406, 699, 254, 48), 'cancel' if self.booth.active else 'booth', self.booth.active or (not self.busy and not self.recorder.active), True)
        text(canvas, 'Tab: pilih kontrol', (692, 717), .41, MUTED)
        text(canvas, 'Enter: aktifkan  /  Q: keluar', (692, 740), .39, MUTED)
        cv2.rectangle(canvas, (944, 92), (1256, 747), SURFACE, -1)
        self.button(canvas, 'Studio [1]', (960, 108, 138, 40), 'tab:studio', primary=self.tab == 'studio')
        self.button(canvas, 'Photobooth [2]', (1106, 108, 134, 40), 'tab:booth', primary=self.tab == 'booth')
        if self.tab == 'studio':
            self.stepper(canvas, 'KAMERA', str(self.camera_choice), 179, 'camera')
            self.button(canvas, 'Hubungkan', (1060, 191, 128, 36), 'camera:apply', not self.busy and not self.recorder.active and self.camera_choice != self.source.index)
            self.stepper(canvas, 'FILTER', self.p.current_filter_name, 257, 'filter')
            self.stepper(canvas, 'KEKUATAN EFEK', f'{self.p.cfg.filter_strength:.0%}', 335, 'strength')
            self.stepper(canvas, 'RESPONS GERAKAN', f'{self.p.cfg.smoothing_alpha:.0%}', 413, 'smooth')
            self.stepper(canvas, 'SENSITIVITAS PINCH', f'{self.p.cfg.pinch_ratio:.2f}', 491, 'pinch')
            self.button(canvas, 'Dual Portal' if self.p.dual_portal else 'Portal tunggal', (968, 556, 166, 38), 'mode', not self.booth.active)
            self.button(canvas, 'Titik: ON' if self.p.show_landmarks else 'Titik: OFF', (1142, 556, 98, 38), 'landmarks', not self.booth.active)
        else:
            text(canvas, 'EMPAT FOTO. SATU CERITA.', (968, 184), .47, ACCENT)
            text(canvas, 'Jeda antar foto, tanpa terburu-buru.', (968, 210), .38, MUTED)
            text(canvas, 'TEMA STRIP', (968, 251), .43, MUTED)
            self.button(canvas, self.theme + '  >', (968, 265, 272, 40), 'theme', not self.booth.active)
            text(canvas, 'HITUNG MUNDUR', (968, 339), .43, MUTED)
            self.button(canvas, f'{self.delay} detik per foto  >', (968, 353, 272, 40), 'delay', not self.booth.active)
            text(canvas, 'CAPTION / KLIK UNTUK EDIT', (968, 427), .43, MUTED)
            self.button(canvas, self.caption or 'Tambah caption', (968, 441, 272, 40), 'caption', not self.booth.active)
            text(canvas, 'Foto & strip bebas panel informasi.', (968, 521), .39, MUTED)
            text(canvas, 'X / Esc membatalkan sesi aktif.', (968, 546), .39, MUTED)
            text(canvas, 'Hasil PNG disimpan di captures/.', (968, 571), .39, MUTED)
        cv2.line(canvas, (968, 613), (1240, 613), FIELD, 1)
        self.button(canvas, f'Preset {self.slot} / 3  >', (968, 628, 272, 36), 'slot', not self.booth.active)
        self.button(canvas, 'Simpan [K]', (968, 678, 130, 42), 'save', not self.booth.active)
        self.button(canvas, 'Muat [O]', (1110, 678, 130, 42), 'load', not self.booth.active)
        status = self.message if now < self.message_until or self.error else 'Pinch: ganti filter  /  Dua kepalan: ganti mode  /  Lepas untuk mengulang'
        text(canvas, status, (24, 780), .44, (145, 159, 255) if self.error else MUTED)
        if self.editing:
            canvas = cv2.addWeighted(canvas, .28, np.zeros_like(canvas), .72, 0)
            cv2.rectangle(canvas, (310, 274), (970, 508), SURFACE, -1)
            text(canvas, 'TULIS CAPTION-MU', (342, 321), .7, ACCENT, 2)
            text(canvas, 'Maksimal 28 karakter (huruf Latin, angka, tanda baca).', (342, 351), .44, MUTED)
            cv2.rectangle(canvas, (342, 373), (938, 429), FIELD, -1)
            text(canvas, self.draft + '|', (357, 409), .64)
            text(canvas, 'Enter: simpan    Esc: batal    Backspace: hapus', (342, 470), .46, MUTED)
        self.focus %= max(1, len(self.buttons))
        return canvas

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.booth.cancel()
        self.io_pool.shutdown(wait=True)
        self.camera_pool.shutdown(wait=True)
        # A replacement that finishes during shutdown still owns a camera handle.
        if self.camera_future is not None:
            try:
                self.camera_future.result().release()
            except (OSError, cv2.error):
                pass
            self.camera_future = None
        self.poll()
