# 🚀 Predictive Churn Guard: Enterprise Customer Retention Intelligence

Sistem analisis prediktif berbasis Kecerdasan Buatan (AI) yang dirancang khusus untuk mengidentifikasi resiko kehilangan pelanggan (*customer churn*) secara dini. Aplikasi ini membantu jajaran manajemen dan tim *Customer Success* untuk mengambil tindakan retensi preventif secara tepat sasaran, mengoptimalkan Biaya Akuisisi Pelanggan (CAC), serta meningkatkan *Customer Lifetime Value* (LTV).

Aplikasi ini ditenagai oleh algoritma komputasi **Random Forest Classifier** tingkat lanjut yang telah dioptimasi melalui fase *Hyperparameter Tuning* intensif untuk memastikan stabilitas prediksi yang tinggi pada skala data operasional global.

---

## 🎯 Value Proposition & Fitur Utama Dashboard

1. **Enterprise Data Ingestion Form**:
   * Fasilitas entri data karakteristik operasional dan finansial pelanggan secara *real-time*.
   * Seluruh metrik finansial menggunakan standarisasi mata uang universal **US Dollar (USD)** untuk mendukung analisis performa multi-nasional.
   * Dilengkapi dengan instruksi batas nilai logis bawaan guna memastikan validitas data sebelum diproses oleh model inti.

2. **Dynamic Geographical Routing**:
   * Fitur dropdown dinamis pintar (*conditional binding*). Menu **Kota Domisili (City)** akan secara otomatis melakukan pemfilteran wilayah secara adaptif hanya berdasarkan **Negara Asal (Country)** yang dipilih secara sah (USA, UK, Germany, India, Bangladesh).

3. **Live Probability Charting & Metrics**:
   * Visualisasi interaktif instan (*real-time bar chart*) yang membedah perbandingan persentase probabilitas hasil keputusan akurat model antara status `Loyal` dan `Churn`.

4. **Data Pipeline Transparency (Audit Trail)**:
   * Menyediakan fitur enkapsulasi transparansi proses (*expander audittrail*). Fitur ini membongkar proses transformasi data mentah di balik layar (*Under the Hood*) dari bentuk teks/angka acak menjadi matriks biner terstandardisasi (`StandardScaler` & `OneHotEncoder`) 51 dimensi vektor sebelum dieksekusi oleh mesin kalkulasi biner.

---

## 🛠️ Arsitektur Teknologi & Dependensi

Proyek ini dibangun menggunakan arsitektur *clean-code pipeline* berbasis Python guna menjamin skalabilitas saat diintegrasikan dengan infrastruktur IT perusahaan yang sudah ada:

* **Core Engine**: Python 3.12+
* **Data Preprocessing & Scaler**: Scikit-Learn (Imputer, Standardizer, OneHotEncoder)
* **Data Analytics Layer**: Pandas & NumPy
* **Application Delivery Framework**: Streamlit Cloud Engine
* **Model Serialization**: Secure Pickle Binary Encapsulation

---

## 📂 Struktur Direktori Repositori

```text
├── app.py                  # Core script logika interface, css dashboard, dan pipeline data
├── model_churn_final.pkl   # Berkas biner enkapsulasi (Model AI, Scaler, Imputer, dan Encoder)
├── requirements.txt        # Manifest daftar pustaka (library) pihak ketiga untuk replikasi sistem
└── README.md               # Dokumentasi teknis operasional sistem
```

---

## 💻 Panduan Arsitektur Infrastruktur & Operasional

Sistem analitis ini dirancang fleksibel untuk dioperasikan melalui dua metode infrastruktur penempatan (deployment) berikut:

### Opsi A: Pengoperasian Server Lokal (Local Development & Staging)
Untuk mereplikasi dan menjalankan lingkungan sistem ini pada perangkat komputer lokal atau server internal perusahaan, eksekusi instruksi berikut pada terminal:

1. **Isolasi Environment**  
   Buat ruang kerja virtual terisolasi guna menghindari konflik versi library global komputer:
   ```bash
   python -m venv .venv
```

2. **Aktivasi Ruang Kerja Virtual**  
   * **Windows (Command Prompt/PowerShell)**: 
     ```bash
.venv\Scripts\activate
```
   * **Mac/Linux**: 
     ```bash
source .venv/bin/activate
```

3. **Instalasi Pustaka Dependensi**  
   Pasang paket ekosistem data science sesuai standar kalkulasi model melalui berkas manifest:
   ```bash
pip install -r requirements.txt
```

4. **Inisiasi Server Aplikasi**  
   Aktifkan server lokal untuk membuka dashboard pada browser internal secara otomatis:
   ```bash
   streamlit run app.py
```

### Opsi B: Deployment Produksi Global (Streamlit Cloud Integration)
Untuk mengonlinekan dashboard agar dapat diakses secara publik 24 jam oleh jajaran stakeholder eksternal tanpa infrastruktur fisik lokal, ikuti prosedur orkestrasi cloud berikut:

1. **Otentikasi Platform Cloud**  
   Akses portal resmi **[share.streamlit.io](https://share.streamlit.io)** dan lakukan proses masuk (*Sign In*) menggunakan akun GitHub yang menyimpan repositori kode ini.

2. **Inisiasi Aplikasi Baru**  
   Pada halaman utama konsol manajemen Streamlit, klik tombol **Create app** (atau **New app**) di pojok kanan atas, lalu pilih opsi penyusunan *Deploy a public app*.

3. **Konfigurasi Jalur Kode Sumber (*Source Routing*)**  
   Isi parameter form manajemen cloud sesuai dengan koordinat repositori GitHub tempat file Anda berada:
   * **Repository**: Pilih nama repositori proyek target yang sesuai.
   * **Branch**: Tentukan branch utama (pilih `main` atau `master`).
   * **Main file path**: Ketik secara presisi **`app.py`** sebagai entry-point eksekusi antarmuka utama.

4. **Alokasi Domain Kustom (*App URL Branding*)**  
   Pada kolom *App URL*, hapus string acak bawaan platform dan tentukan sub-domain profesional khusus korporat (Contoh: `nama-perusahaan-churn-analytics`). Alamat akses publik akhir akan tersusun rapi dalam format: `domain-pilihan.streamlit.app`.

5. **Eksekusi Orkestrasi Kontainer Cloud**  
   Klik tombol **Deploy!**. Server cloud akan menginisiasi proses pembacaan file `requirements.txt` secara otomatis, menyusun dependensi pustaka matematika di dalam wadah server (*container instance*) terisolasi, dan meluncurkan dashboard analitis ke publik dalam waktu 1-3 menit.

---

## 🔒 Catatan Keamanan Data & Privasi

Aplikasi ini berjalan secara independen tanpa menyimpan, merekam, atau mentransmisikan data sensitif yang diinput oleh pengguna ke dalam basis data eksternal pihak ketiga manapun (*No In-Memory Storage Data Logging*). 

Seluruh proses transformasi data Z-score statistik dan perhitungan probabilitas keputusan model dilakukan secara efimeral (sementara) di dalam memori server *instance container* cloud yang terisolasi dan akan langsung dibersihkan secara otomatis saat sesi browser pengguna berakhir.
```
