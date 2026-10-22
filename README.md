# Pengukuran Dimensi Objek

## Identitas Kelompok 8

Proyek UTS Machine Vision.

| No | Nama                          | NIM          |
|----|-------------------------------|--------------|
| 1  | Ahmad Bagus Idkholus Surur    | 23050874166  |
| 2  | Khairie Arrantisy             | 23050874182  |
| 3  | Muhammad Ilham Gymnastiar     | 23050874169  |
| 4  | Wahyu Rizki Agung Kurniawan   | 23050874189  |

Repository ini dikembangkan dari proyek sumber yang tercantum di bagian kredit.

Mengukur panjang dan lebar objek kotak (persegi / persegi panjang) secara real-time via kamera/webcam dengan rumus konversi piksel ke cm.

## Alur Deteksi (Tahapan OpenCV)

```text
Webcam Frame -> Grayscale -> Gaussian Blur -> Canny Edge -> Morfologi (Dilasi/Erosi) -> Find Contours -> Filter Bentuk Kotak -> Hitung Panjang/Lebar Piksel -> Rumus Konversi ke cm
```

## Rumus Konversi

$$Panjang (cm) = \frac{Panjang (piksel)}{Skala (piksel/cm)}$$
$$Lebar (cm) = \frac{Lebar (piksel)}{Skala (piksel/cm)}$$

- Default skala: `20.0 px/cm` (cocok untuk jarak wajar webcam).
- Tekan **`+`** atau **`-`** saat aplikasi berjalan untuk menaikkan/menurunkan skala secara live.
- Atau masukkan parameter opsional: `python object_size.py --scale 25.0`

## Menjalankan

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python object_size.py
```

### Tombol Interaktif

- **`+`** : Tambah skala (+1 px/cm)
- **`-`** : Kurangi skala (-1 px/cm)
- **`q` / `Esc`** : Keluar dari program

## Tes

```powershell
python -m unittest discover -v
```

Daftar anggota dan kredit lengkap: lihat [AUTHORS.md](AUTHORS.md).