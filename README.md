# Pengukuran Dimensi Objek Real-Time (OpenCV)

## Identitas Kelompok 8

Proyek UTS Machine Vision.

| No | Nama                          | NIM          |
|----|-------------------------------|--------------|
| 1  | Ahmad Bagus Idkholus Surur    | 23050874166  |
| 2  | Khairie Arrantisy             | 23050874182  |
| 3  | Muhammad Ilham Gymnastiar     | 23050874169  |
| 4  | Wahyu Rizki Agung Kurniawan   | 23050874189  |

Repository ini dikembangkan untuk deteksi dan pengukuran dimensi objek kotak (persegi / persegi panjang) secara real-time via webcam menggunakan OpenCV murni (tanpa dataset / tanpa YOLO).

## Tabel Kalibrasi Jarak vs Piksel (1 cm Nyata)

Tabel konversi piksel ke 1 cm nyata pada kamera webcam standar (resolusi 640×480, FOV ~60°):

| Jarak ke Kamera | Ukuran 1 cm di Layar | Nilai Skala (`--scale`) | Rekomendasi Kondisi |
|:---:|:---:|:---:|:---|
| **10 cm** | **55 piksel** | `55.0` | Sangat dekat (benda kecil / kartu rapat ke lensa) |
| **15 cm** | **37 piksel** | `37.0` | Jarak dekat |
| **20 cm** | **28 piksel** | `28.0` | Jarak sedang |
| **25 cm** | **22 piksel** | `22.0` | Jarak genggam tangan nyaman (default) |
| **30 cm** | **18 piksel** | `18.0` | Jarak meja kerja normal |
| **40 cm** | **14 piksel** | `14.0` | Jarak jauh |
| **50 cm** | **11 piksel** | `11.0` | Jarak sangat jauh (benda berukuran besar) |

## Rumus Perhitungan

$$\text{Panjang (cm)} = \frac{\text{Panjang (piksel)}}{\text{Skala (px/cm)}}$$

$$\text{Lebar (cm)} = \frac{\text{Lebar (piksel)}}{\text{Skala (px/cm)}}$$

## Menjalankan Program

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Menjalankan dengan skala default (20.0 px/cm)
python object_size.py

# Menjalankan pada jarak 10 cm dari kamera (skala 55 px/cm)
python object_size.py --scale 55.0
```

### Kontrol Keyboard Interaktif

| Tombol | Fungsi |
|:---:|:---|
| **`+`** atau **`=`** | Menaikkan nilai skala (+1 px/cm) secara live |
| **`-`** atau **`_`** | Menurunkan nilai skala (-1 px/cm) secara live |
| **`q`** atau **`Esc`** | Menutup kamera dan keluar dari program |

## Fitur Deteksi

- **Convex Hull**: Merapikan tepi luar objek yang dipegang sehingga jari tangan tidak merusak bentuk kontur kotak.
- **Supresi Kontur Dalam**: Mengeliminasi gambar/tulisan/logo di dalam benda agar hanya sisi terluar benda yang terukur.
- **Pure Code (Ponytail Style)**: Kode ringkas, cepat, dan efisien tanpa dependensi berlebih.

## Tes Unit

```powershell
python -m unittest discover -v
```

Daftar anggota dan kredit lengkap: lihat [AUTHORS.md](AUTHORS.md).