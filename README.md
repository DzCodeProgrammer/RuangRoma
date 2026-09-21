# RuangRona

**Ruang untuk bermain dengan warna.** RuangRona adalah studio kamera interaktif berbasis Python, OpenCV, dan MediaPipe. Gerakan tangan membentuk portal dan mengubah warna di dalamnya secara real time.

## Fitur

- Studio satu jendela dengan preview besar dan kontrol yang bisa diklik.
- 14 filter: Sepia, Film, VHS, Dual-tone, Thermal, Sketch, Pixelate, Glitch, Invert, Red-channel, Edge, Blur, Cartoon, dan Rainbow-wave.
- Gesture sekali picu dan portal yang diperhalus agar tidak mudah bergetar.
- Pengaturan kamera, kekuatan filter, respons gerakan, sensitivitas pinch, mode portal, dan landmark.
- Photobooth empat foto dengan hitung mundur 2, 3, atau 5 detik, tiga tema strip, caption, preview, dan simpan ulang.
- Tiga slot preset yang menyimpan tampilan pilihan dan memulihkan slot terakhir saat aplikasi dibuka.
- Foto PNG dan video AVI/MJPG. Hasil disimpan di `captures/` tanpa panel kontrol.
- Operasi simpan dan penggantian kamera berjalan di latar belakang agar preview tetap responsif.

## Instalasi

Gunakan Python 3.11 dan virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python RuangRona.py
```

Di Windows, aktifkan virtual environment dengan `.venv\Scripts\activate`.
Berikan izin kamera kepada Terminal atau Python. Paket `opencv-contrib-python` sudah menyediakan modul `cv2`; jangan memasang `opencv-python` secara terpisah di environment yang sama.

## Cara menggunakan

Panel **Studio** mengatur kamera dan tampilan portal. Gunakan tombol `−`/`+`, lalu klik **Hubungkan** setelah memilih indeks kamera. Kamera lama tetap aktif jika kamera baru tidak dapat dibuka.

Panel **Photobooth** mengatur tema, jeda, dan caption. Klik **Photobooth** untuk memulai empat foto. Gunakan `X` atau `Esc` untuk membatalkan sesi. Strip disimpan otomatis; tombol **Simpan strip lagi** membuat salinan baru.

Preset menyimpan filter, kekuatan efek, respons gerakan, sensitivitas pinch, mode portal, tema photobooth, caption, dan timer. Pilih slot 1–3, lalu klik **Simpan**. Slot terakhir yang disimpan akan dimuat saat aplikasi dibuka. File lokal `presets.json` tidak dimasukkan ke Git.

### Keyboard dan gesture

| Kontrol | Fungsi |
|---|---|
| Pinch jempol + kelingking | Filter berikutnya; lepaskan sebelum mengulang |
| Dua tangan mengepal | Ganti mode portal; buka tangan sebelum mengulang |
| `1` / `2` | Buka panel Studio / Photobooth |
| `N` / `P` | Filter berikutnya / sebelumnya |
| `C` | Portal tunggal / Dual Portal |
| `S` | Simpan foto |
| `R` | Mulai / hentikan rekaman |
| `B` | Mulai photobooth |
| `X` | Batalkan photobooth |
| `V` | Kamera / preview hasil strip |
| `K` / `O` | Simpan / muat preset |
| `H` | Tampilkan / sembunyikan info pada preview |
| `L` | Tampilkan / sembunyikan landmark |
| `Tab`, `Enter` | Pindah fokus dan aktifkan kontrol |
| `Q` / `Esc` | Keluar; `Esc` membatalkan photobooth lebih dahulu |

Gesture dikunci selama photobooth dan saat caption diedit supaya filter tidak berubah tanpa sengaja. Menutup aplikasi akan menyelesaikan penulisan video dan foto yang sedang berlangsung.

## Hasil

- Foto: `captures/ruangrona_*.png`
- Strip: `captures/ruangrona_strip_*.png`
- Video: `captures/ruangrona_*.avi`

Video tidak memiliki audio. Garis portal dan landmark yang diaktifkan masuk ke hasil; panel kontrol, hitung mundur, notifikasi, dan indikator rekaman hanya tampil pada layar.

## Pengujian

```bash
python -m unittest discover -s tests -v
```

Pengujian mencakup gesture, smoothing, kekuatan filter, clipping portal, encoder video, hitung mundur photobooth, komposisi strip, penyimpanan PNG atomik, dan validasi preset. Akses kamera serta kenyamanan gesture tetap perlu diperiksa langsung pada perangkat.

## Lisensi

Copyright modifikasi © 2026 DzCodeProgrammer. Proyek tersedia di bawah [Lisensi MIT](LICENSE).
