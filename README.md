# Pengukuran Dimensi Objek

Mengukur panjang dan lebar objek pada foto dengan referensi berukuran diketahui.
Input `--width` dan hasil pengukuran menggunakan **cm**.

Kode awal oleh Pawan Bansal ([hackbansu](https://github.com/hackbansu)) melalui
[Practical-CV/Measuring-Size-of-Objects-with-OpenCV](https://github.com/Practical-CV/Measuring-Size-of-Objects-with-OpenCV),
berdasarkan [tutorial Adrian Rosebrock / PyImageSearch](https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/).
Alur contour dan perhitungan pixels-per-metric tetap mengikuti kode sumber.

## Menjalankan

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python object_size.py --image images/example_01.png --width 2.4257
```

`--width` adalah lebar objek referensi **paling kiri**, dalam cm, bukan lebar
benda yang ingin diukur. Pada contoh, referensinya koin US quarter:
0.955 inci × 2.54 = 2.4257 cm. Untuk foto sendiri, masukkan ukuran referensi
yang benar. Jangan memasukkan nilai inci lagi atau mengalikan hasil dengan 2.54.

Tekan tombol apa saja pada jendela gambar untuk melihat objek berikutnya.
`object_size_mine.py` menampilkan tahap edge dan morfologi sebelum pengukuran;
parameter dan satuannya sama.

## Kondisi foto

- Referensi dan benda berada pada bidang yang sama, tidak saling menutupi.
- Kamera menghadap hampir tegak lurus ke bidang objek.
- Background kontras agar tepi benda terdeteksi.
- Ukuran dihitung dari kotak contour, bukan rekonstruksi 3D. Perspektif,
  distorsi lensa, dan segmentasi dapat membuat hasil meleset.

## Tes

```sh
python -m unittest discover -v
```

Tes memakai gambar sintetis tanpa membuka jendela. Akurasi foto nyata belum
dikalibrasi. GIF bawaan repository masih menunjukkan versi asli dalam inci.
Repository sumber tidak mencantumkan lisensi; periksa izin sebelum distribusi ulang.
