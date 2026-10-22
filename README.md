# Pengukuran Dimensi Objek Real-Time (OpenCV)

## Identitas Kelompok 8

Proyek UTS Machine Vision.

| No | Nama                          | NIM          |
|----|-------------------------------|--------------|
| 1  | Ahmad Bagus Idkholus Surur    | 23050874166  |
| 2  | Khairie Arrantisy             | 23050874182  |
| 3  | Muhammad Ilham Gymnastiar     | 23050874169  |
| 4  | Wahyu Rizki Agung Kurniawan   | 23050874189  |

Repository ini dibuat untuk mengukur dimensi (panjang dan lebar) objek kotak (persegi / persegi panjang) secara real-time via webcam dengan OpenCV murni tanpa dataset atau machine learning.

## Tabel Kalibrasi Jarak vs Piksel (1 cm Nyata)

Tabel konversi piksel ke 1 cm nyata pada webcam (resolusi 640×480, FOV ~60°):

| Jarak ke Kamera | Ukuran 1 cm di Layar | Nilai Skala (`--scale`) | Rekomendasi Kondisi |
|:---:|:---:|:---:|:---|
| **10 cm** | **55 piksel** | `55.0` | Jarak sangat dekat (kartu/benda rapat ke lensa) |
| **15 cm** | **37 piksel** | `37.0` | Jarak dekat |
| **20 cm** | **28 piksel** | `28.0` | Jarak sedang |
| **25 cm** | **22 piksel** | `22.0` | Jarak genggam tangan nyaman (default) |
| **30 cm** | **18 piksel** | `18.0` | Jarak meja kerja normal |
| **40 cm** | **14 piksel** | `14.0` | Jarak jauh |
| **50 cm** | **11 piksel** | `11.0` | Jarak sangat jauh (benda besar) |

## Rumus Perhitungan

Sisi terpanjang selalu didefinisikan sebagai **Panjang**, dan sisi terpendek sebagai **Lebar**:

$$\text{Panjang}_{\text{px}} = \max(\text{sisi}_1, \text{sisi}_2)$$

$$\text{Lebar}_{\text{px}} = \min(\text{sisi}_1, \text{sisi}_2)$$

$$\text{Panjang (cm)} = \frac{\text{Panjang}_{\text{px}}}{\text{Skala (px/cm)}}$$

$$\text{Lebar (cm)} = \frac{\text{Lebar}_{\text{px}}}{\text{Skala (px/cm)}}$$

Dengan rumus ini, nilai **Panjang selalu $\ge$ Lebar** secara matematis dan tidak akan tertukar.

## Cara Menjalankan

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Menjalankan kamera langsung
python object_size.py

# Kalibrasi langsung dengan benda berukuran diketahui (misal kartu 8.5 cm)
python object_size.py --ref 8.5
```

### Kontrol Keyboard Interaktif

| Tombol | Fungsi |
|:---:|:---|
| **`k`** | **Kalibrasi 1-Klik**: Menyesuaikan skala otomatis dengan panjang benda acuan nyata |
| **`+`** atau **`=`** | Menambah nilai skala (+0.5 px/cm) secara langsung |
| **`-`** atau **`_`** | Mengurangi nilai skala (-0.5 px/cm) secara langsung |
| **`q`** atau **`Esc`** | Menutup kamera dan keluar dari program |

## Fitur Deteksi

- **Kontur Tepi Luar Sahaja**: Menggunakan `RETR_EXTERNAL` dan morfologi penutup celah sehingga hanya tepi terluar benda yang dideteksi; gambar/tulisan di dalam benda diabaikan.
- **Validasi Sudut 90°**: Mengevaluasi kosinus keempat sudut poligon ($|\cos\theta| \le 0.35$) untuk memastikan hanya benda kotak nyata yang terdeteksi (tangan dan background acak tereliminasi).
- **Style Ponytail (Kode Tunggal)**: Kode tunggal yang ringkas, bersih, tanpa komentar berlebih di dalam kode.

## Tes Unit

```powershell
python -m unittest discover -v
```

Daftar anggota dan kredit lengkap: lihat [AUTHORS.md](AUTHORS.md).