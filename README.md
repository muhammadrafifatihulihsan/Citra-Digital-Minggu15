# Pengolahan Citra Digital - Praktikum & Assignment Minggu 15
### Topik: Penerapan Pengolahan Citra Digital & Computer Vision di Berbagai Bidang Industri (Advanced Digital Image Processing & Computer Vision Applications)

Repository ini berisi kumpulan kode program untuk mata kuliah Pengolahan Citra Digital (PCD) pada Minggu ke-15. Fokus utama pada minggu ini adalah eksplorasi dan implementasi teknologi pengolahan citra digital dan computer vision tingkat lanjut dalam memecahkan masalah riil di berbagai sektor industri. Proyek minggu ini terbagi dalam empat modul praktikum utama: deteksi objek berbasis Deep Learning (YOLOv3) untuk computer vision umum, analisis kesehatan vegetasi (NDVI) pada data geospasial penginderaan jauh (Remote Sensing), segmentasi dan deteksi anomali jaringan pada citra MRI/X-Ray (Medical Imaging), serta pelacakan dan penghitungan pejalan kaki (People Counting & Tracking) pada sistem keamanan pintar (Surveillance Systems).

---

## 👤 Informasi Mahasiswa
* **Nama:** Muhammad Rafi Fatihul Ihsan
* **NIM:** 24343016
* **Sesi/Kelas:** 202523430039

---

## 📂 Struktur Project
```text
program/
├── src/
│   ├── config/                 # File konfigurasi dan model YOLOv3
│   │   ├── coco.names          # Label kelas dataset COCO (80 objek standar)
│   │   ├── yolov3.cfg          # File konfigurasi arsitektur jaringan YOLOv3
│   │   ├── yolov3.weights      # Bobot (weights) terlatih YOLOv3 (~248MB - Diabaikan Git)
│   │   └── LOKASI FILE YOLO    # File penanda direktori konfigurasi
│   ├── models/                 # File model tersimpan (jika ada)
│   └── praktikum/              # Modul Praktikum Pengolahan Citra & Vision
│       ├── praktikum1.ipynb    # Latihan 1: Deteksi Objek Berbasis Deep Learning - YOLO
│       ├── praktikum2.ipynb    # Latihan 2: Analisis Vegetasi Berbasis NDVI (Remote Sensing)
│       ├── praktikum3.ipynb    # Latihan 3: Deteksi Anomali pada Citra Medis (Medical Imaging)
│       └── praktikum4.ipynb    # Latihan 4: People Counting & Tracking (Surveillance Systems)
├── requirements.txt            # Daftar dependensi Python (OpenCV, PyTorch, Matplotlib, dll)
├── .gitignore                  # File konfigurasi penyaringan repositori Git
└── README.md                   # Dokumentasi lengkap proyek
```

---

## 🚀 Fitur & Modul Praktikum

### 1. Eksperimen Deteksi Objek Real-time dengan YOLOv3 (`praktikum1.ipynb`)
Mendemonstrasikan implementasi algoritma pendeteksi objek satu tahap (*one-stage object detector*) YOLOv3 berbasis modul OpenCV DNN (`readNetFromDarknet`).

* **Konfigurasi:** Memanfaatkan file konfigurasi arsitektur Darknet `yolov3.cfg` dan model terlatih `yolov3.weights` (~248MB) dengan file label COCO `coco.names` yang menampung 80 kategori kelas objek universal.
* **Pipeline Pemrosesan:**
  * Membaca citra input (contoh citra klasik: `dog.jpg`).
  * Melakukan ekstraksi fitur dan pre-processing citra menjadi representasi tensor `blob` menggunakan fungsi OpenCV `blobFromImage` (melakukan penskalaan piksel dengan faktor $1/255.0$, pengubahan ukuran citra menjadi $416 \times 416$ piksel, dan restorasi warna BGR ke RGB).
  * Forward pass melalui lapisan output jaringan Darknet untuk memperoleh koordinat bounding box, ID kelas, dan tingkat confidence.
  * Menerapkan penapisan *Non-Maximum Suppression* (NMS) dengan ambang batas (*threshold*) confidence 0.5 dan NMS threshold 0.4 guna mengeliminasi bounding box ganda/tumpang tindih pada objek yang sama.
* **Output Evaluasi:** Berhasil mengidentifikasi **3 objek** di dalam citra sampel secara akurat (`dog`, `bicycle`, dan `truck`) lengkap dengan bounding box dan persentase probabilitas.

### 2. Analisis Kesehatan Vegetasi & Klasifikasi Tutupan Lahan - NDVI (`praktikum2.ipynb`)
Mengimplementasikan teknologi pengolahan citra geospasial pada data satelit penginderaan jauh (*Remote Sensing*).

* **Metodologi:** Menggunakan parameter indeks spektral NDVI (*Normalized Difference Vegetation Index*) untuk membedakan vegetasi hijau sehat dari jenis tutupan lahan lainnya dengan memanfaatkan perbedaan reflektansi panjang gelombang Red (merah) dan Near-Infrared (NIR / inframerah dekat).
* **Formulasi Matematika Indeks:**
  $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
* **Hasil Analisis & Statistik:**
  * Rata-rata Nilai NDVI Citra: **-0.001** dengan jangkauan nilai spektral penuh antara **-1.000 hingga 1.000**.
  * Distribusi Tutupan Lahan (*Land Cover Classification*):
    * **Water (Air):** 96,367 piksel (40.2%) - NDVI sangat rendah/negatif.
    * **Urban/Soil (Lahan Terbuka/Perkotaan):** 66,092 piksel (27.5%) - NDVI rendah berkisar mendekati nol.
    * **Sparse Veg (Vegetasi Jarang/Stres):** 14,412 piksel (6.0%) - NDVI sedang.
    * **Dense Veg (Vegetasi Lebat/Sehat):** 63,129 piksel (26.3%) - NDVI tinggi mendekati 1.
  * Penilaian Kesehatan Tanaman:
    * Rata-rata NDVI Vegetasi Sehat (*Healthy Vegetation*): **0.891** (menunjukkan kerapatan klorofil yang sangat baik).
    * Rata-rata NDVI Vegetasi Stres (*Stressed Vegetation*): **0.301** (menunjukkan adanya indikasi kekeringan atau degradasi zat hijau daun).

### 3. Deteksi Anomali Jaringan & Segmentasi Citra Medis (`praktikum3.ipynb`)
Mengaplikasikan teknik penapisan dan segmentasi citra medis (*Medical Imaging*) untuk mengenali area anomali (seperti sel tumor, jaringan abnormal, dll.).

* **Pipeline Algoritma:** Pre-processing (filtering noise, histogram equalization), thresholding adaptif (segmentasi biner), operasi morfologi matematika (erosion dan dilation) untuk memisahkan noise kecil di luar kluster, konturisasi, dan ekstraksi area anomali.
* **Hasil Evaluasi Kinerja Klasifikasi (*Medical Imaging Analysis Summary*):**
  
  | Modality | Sensitivity | Specificity | Accuracy | AUC |
  |---|---|---|---|---|
  | **X-Ray** | 0.850 | 0.920 | 0.890 | 0.940 |
  | **MRI** | 0.910 | 0.880 | 0.900 | 0.950 |
  | **Retina** | 0.780 | 0.850 | 0.820 | 0.890 |

* **Kesimpulan Medis:** Citra MRI menunjukkan kinerja deteksi anomali tertinggi dengan akurasi **90.0%** dan tingkat sensitivitas **91.0%**, disusul oleh citra X-Ray sebesar **89.0%**, sementara citra retina memperoleh akurasi **82.0%** dikarenakan kompleksitas pembuluh darah halus yang tinggi.

### 4. Pelacakan Objek Pintar & Penghitungan Orang - Surveillance Systems (`praktikum4.ipynb`)
Membangun kecerdasan buatan pada sistem keamanan pintar (*Smart Surveillance Systems*) untuk melakukan pemantauan kerumunan (*people monitoring*).

* **Algoritma:** Deteksi tubuh manusia (*person detection*) dikombinasikan dengan metode pelacakan centroid (*centroid tracking*) berbasis perhitungan jarak Euclidean untuk memelihara identitas objek (*object tracking ID*) yang dinamis antarframenya.
* **Hasil Evaluasi Kinerja Penghitungan (*Surveillance System Performance Analysis*):**
  * Akurasi Penghitungan (*Counting Accuracy*): **1.000 (100.0%)**
  * Rata-rata Kesalahan Absolut (*Mean Absolute Error*): **0.00 orang per frame** (sangat presisi pada lingkungan pengujian terkontrol).
  * Total Orang Terdeteksi (*Estimated Counts*): **12 orang**
  * Total Orang Sebenarnya (*True Counts*): **12 orang**
* **Simulasi Pelacakan Sederhana (*Simple Tracking Simulation*):**
  * Frame 0 s.d 2: 0 orang terdeteksi
  * Frame 3 s.d 5: 1 orang terdeteksi (memulai pelacakan dengan ID tunggal)
  * Frame 6 s.d 8: 2 orang terdeteksi (terdapat objek baru memasuki bidang pantau)
  * Frame 9: 3 orang terdeteksi (deteksi kerumunan skala kecil).

---

## 📝 Analisis Kritis Hasil Eksperimen

### A. Analisis Trade-off antar Sektor Aplikasi
* **Kecepatan vs. Presisi Komputasi:** Deteksi objek real-time menggunakan YOLOv3 (`praktikum1`) memerlukan daya komputasi GPU yang tinggi karena ukuran file weights yang besar (~248MB) dan proses ekstraksi ribuan *anchor boxes*, namun memberikan presisi spasial multi-kelas yang tinggi. Sebaliknya, modul pelacakan centroid sederhana (`praktikum4`) berjalan dengan konsumsi daya yang sangat minim (*ultra-low computational footprint*) karena hanya menghitung geometri centroid objek yang terdeteksi, menjadikannya ideal untuk perangkat Internet of Things (IoT) atau *Embedded Surveillance System*.
* **Analisis Spektral vs. Geometris:** Di sektor *Remote Sensing* (`praktikum2`), citra tidak hanya diolah secara visual geometris, melainkan secara radiometris spektral. Penggunaan indeks NDVI membuktikan bahwa informasi di luar spektrum kasat mata manusia (yaitu *Near-Infrared*) sangat krusial dalam mengenali kondisi fisik objek bumi secara objektif, yang tidak dapat diidentifikasi oleh kamera RGB standar.

### B. Kombinasi Operasi Morfologi pada Deteksi Citra Medis
* Keberhasilan deteksi tumor/anomali medis (`praktikum3`) sangat bergantung pada *pre-processing* dan *post-processing* citra. Penggunaan *thresholding* biner saja tidak cukup karena menyisakan noise berupa bintik-bintik kecil akibat variasi intensitas jaringan tubuh yang sehat.
* Penerapan operasi morfologi **Opening** (Erosi diikuti Dilasi) terbukti sangat efektif untuk mengeliminasi noise terisolasi tanpa mengubah geometri tumor utama, sedangkan operasi **Closing** (Dilasi diikuti Erosi) berhasil merapatkan celah-celah kecil di dalam jaringan anomali sehingga luas area abnormal dapat dihitung secara presisi.

### C. Tantangan Tracking di Sistem Pengawasan Riil
* Pada praktikum 4, meskipun pelacakan centroid mendapatkan akurasi **100%** pada simulasi terkontrol, pada skenario dunia nyata algoritma pelacakan centroid ini akan menghadapi kendala **occlusion** (objek saling menutupi) dan **ID switching** ketika pejalan kaki bergerak sangat berhimpitan.
* **Rekomendasi:** Untuk sistem komersial, disarankan mengintegrasikan fitur deskriptor visual (*such as deep features*) seperti pada algoritma DeepSORT agar pelacakan tetap konsisten meskipun objek sempat terhalang objek lain.

---

## 🛠️ Cara Menjalankan Program

### 1. Persiapan Lingkungan & Instalasi Dependensi
Pastikan Anda menggunakan Python 3.9 atau lebih baru.
```bash
pip install -r requirements.txt
```
> [!WARNING]
> Untuk modul YOLOv3, file weights (`yolov3.weights` sebesar ~248MB) akan diunduh secara otomatis dari repository darknet resmi PJ Reddie saat pertama kali Anda menjalankan notebook `praktikum1.ipynb`. Pastikan koneksi internet Anda stabil.

### 2. Membuka Notebook Praktikum
Untuk melihat dan menjalankan masing-masing modul eksperimen:
* **Modul YOLOv3 Object Detection:**
  ```bash
  jupyter notebook src/praktikum/praktikum1.ipynb
  ```
* **Modul Remote Sensing (NDVI):**
  ```bash
  jupyter notebook src/praktikum/praktikum2.ipynb
  ```
* **Modul Medical Imaging (Deteksi Anomali):**
  ```bash
  jupyter notebook src/praktikum/praktikum3.ipynb
  ```
* **Modul Surveillance (People Counting & Tracking):**
  ```bash
  jupyter notebook src/praktikum/praktikum4.ipynb
  ```

---

## 🏁 Ringkasan Rekomendasi Aplikasi PCD & Computer Vision
* **Computer Vision (YOLOv3)** adalah solusi terbaik untuk otomasi industri umum seperti deteksi kerusakan barang, robotika, dan *self-driving cars*.
* **Remote Sensing (NDVI)** sangat direkomendasikan untuk sektor pertanian presisi (*precision agriculture*) dan pemantauan kehutanan global guna mendeteksi kesehatan lahan secara makro.
* **Medical Imaging** membutuhkan tingkat sensitivitas yang sangat tinggi (seperti modalitas MRI dengan Sensitivity **91%**) untuk meminimalkan tingkat *false negative* (pasien sakit terdiagnosa sehat).
* **Smart Surveillance (Tracking & Counting)** merupakan pilar utama pengembangan *Smart City* untuk manajemen transportasi publik, keamanan lingkungan, dan analisis pola pergerakan masyarakat.
# Citra-Digital-Minggu15
