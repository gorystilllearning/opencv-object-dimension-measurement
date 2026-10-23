# Pengukuran Dimensi Objek Real-Time (OpenCV)

## Identitas Kelompok 8

Proyek UTS Machine Vision.

| No | Nama                          | NIM          |
|----|-------------------------------|--------------|
| 1  | Ahmad Bagus Idkholus Surur    | 23050874166  |
| 2  | Khairie Arrantisy             | 23050874182  |
| 3  | Muhammad Ilham Gymnastiar     | 23050874169  |
| 4  | Wahyu Rizki Agung Kurniawan   | 23050874189  |

Mengukur panjang dan lebar objek kotak secara real-time melalui webcam.
Skala pengukuran menggunakan piksel per cm.

## Perkiraan Jarak dan Piksel

Tabel ini ilustrasi teoretis untuk resolusi 640×480 dan FOV horizontal sekitar
60°, **bukan hasil kalibrasi webcam MacBook atau kamera tertentu**. Program
tidak memaksa resolusi/FOV tersebut. Gunakan benda referensi untuk menentukan
skala sebenarnya; jangan mengandalkan tabel ini sebagai ukuran akurat.

| Jarak ke Kamera | Ukuran 1 cm di Layar | Nilai Skala (`--scale`) | Rekomendasi Kondisi |
|:---:|:---:|:---:|:---|
| **10 cm** | **55 piksel** | `55.0` | Jarak sangat dekat (kartu/benda rapat ke lensa) |
| **15 cm** | **37 piksel** | `37.0` | Jarak dekat |
| **20 cm** | **28 piksel** | `28.0` | Jarak sedang |
| **25 cm** | **22 piksel** | `22.0` | Perkiraan pada asumsi di atas |
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
Koordinat sudut tetap berupa pecahan saat dihitung; pembulatan hanya untuk
menggambar dan menampilkan hasil, bukan untuk menentukan skalanya.

## Cara Menjalankan

Mac/Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python object_size.py --ref 8.5
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Menjalankan kamera langsung
python object_size.py

# Ukuran referensi harus sesuai penggaris; contoh sisi terpanjang 8.5 cm
python object_size.py --ref 8.5
```

`--scale` bernilai awal 20 px/cm dan ditandai **skala awal/manual** di layar,
bukan otomatis hasil kalibrasi. Untuk hasil yang bisa dipertanggungjawabkan:

1. Ukur sisi terpanjang benda referensi dengan penggaris dan masukkan lewat
   `--ref` (alias `--reference-length`; opsi lama `--width` tetap diterima).
2. Letakkan referensi di bidang pengukuran. **Tombol k memilih kotak terbesar
   yang terdeteksi**, bukan objek paling kiri dan bukan hasil pilihan mouse.
   Pastikan benda lain lebih kecil atau singkirkan dulu saat kalibrasi.
3. Tekan k, lalu letakkan benda lain pada bidang yang sama. Skala tidak berubah
   otomatis ketika referensi keluar frame.
4. Kamera, jarak, zoom, resolusi, dan bidang harus tetap. Kalibrasi ulang setelah
   salah satunya berubah. Pengaturan +/- mengubah skala secara manual.

Di Mac, izinkan aplikasi Terminal/VS Code melalui System Settings → Privacy &
Security → Camera. Gunakan `--camera 1` jika perlu mencoba kamera lain.

### Kontrol Keyboard Interaktif

| Tombol | Fungsi |
|:---:|:---|
| **`k`** | Kalibrasi dari sisi terpanjang kotak terbesar yang sedang terdeteksi |
| **`+`** atau **`=`** | Menambah nilai skala (+0.5 px/cm) secara langsung |
| **`-`** atau **`_`** | Mengurangi nilai skala (-0.5 px/cm) secara langsung |
| **`q`** atau **`Esc`** | Menutup kamera dan keluar dari program |

Pengurangan skala diabaikan jika akan menghasilkan nilai nol atau negatif.
Jika k ditekan tanpa kotak terdeteksi, layar menampilkan pesan gagal dan skala
sebelumnya tetap digunakan.

## Fitur Deteksi

- Otsu memisahkan area terang/gelap; polaritas dibalik jika mayoritas mask putih.
  Cocok untuk background polos yang kontras; tidak menjamin hasil pada background ramai.
- `RETR_EXTERNAL` mengambil contour luar dan closing membantu menutup celah.
- Filter sudut memakai $|\cos\theta| \le 0.35$, yaitu toleransi sekitar 90°,
  bukan jaminan hanya benda kotak nyata yang diterima. Bayangan/background masih
  bisa menghasilkan contour menyerupai kotak.
- Angka satu desimal bukan jaminan akurasi 0.1 cm. Uji beberapa benda dengan
  penggaris; perspektif, distorsi lensa, dan blur dapat membuat hasil meleset.

## Susunan Kode

Semua logic program tetap di `object_size.py`, dipisahkan menjadi fungsi:
`parse_args()`, `is_rectangular()`, `detect_boxes()`, `box_sides()`,
`calibrate_scale()`, `draw_frame()`, dan `main()`.
Parameter deteksi dikumpulkan di bagian atas file. Dependensi cukup OpenCV
dan NumPy; tidak ada salinan script/debug lama yang dikembalikan.

## Tes Unit

```powershell
python -m unittest discover -v
```

Tes mencakup ukuran numerik, kalibrasi tombol k, skala tetap, pengaturan +/-,
validasi input, Otsu pada background terang/gelap, rotasi, filter bentuk, serta
pelepasan kamera saat error. Kamera dan GUI disimulasikan pada tes; akurasi
webcam fisik belum diverifikasi oleh asisten.

Daftar anggota dan kredit lengkap: lihat [AUTHORS.md](AUTHORS.md).
