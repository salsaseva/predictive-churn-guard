import streamlit as st
import pickle
import pandas as pd
import numpy as np

# konfigurasi utama tema halaman dashboard
st.set_page_config(
    page_title="Prediksi Churn Pelanggan UDINUS", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# kustom css untuk menyelaraskan dengan tema dark mode premium
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        font-size: 16px;
    }
    .card-box {
        background-color: #1e293b; /* warna boks dark slate */
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border-left: 5px solid #4f46e5;
    }
    .card-box h4 {
        color: #f8fafc !important; /* memaksa teks judul boks tetap putih terang */
        margin-top: 0px;
        margin-bottom: 15px;
        font-weight: bold;
    }
    .metric-container {
        background-color: #0f172a;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# header utama dengan gaya modern gradient
st.markdown("""
    <div style='background: linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%); padding: 25px; border-radius: 12px; margin-bottom: 25px; color: white;'>
        <h1 style='margin: 0; font-size: 28px; font-weight: bold; color: white;'>📊 Customer Churn Analytics Dashboard</h1>
        <p style='margin: 5px 0 0 0; opacity: 0.9; color: #e2e8f0;'>Sistem Kecerdasan Buatan berbasis Random Forest Classifier untuk Deteksi Dini Resiko Pelanggan Berhenti Berlangganan</p>
        <p style='margin: 5px 0 0 0; opacity: 0.9; color: #e2e8f0;'>Oleh : Salsa Seva Yuliana</p>
        <p style='margin: 5px 0 0 0; opacity: 0.9; color: #e2e8f0;'>NIM : A11.2022.14672</p>
    </div>
""", unsafe_allow_html=True)

# teks informasi panduan di area sidebar kiri
st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h3 style='margin: 0; color: #38bdf8;'>⚙️ Control Panel</h3>
        <small style='color: #94a3b8;'>Sistem Prediksi Churn v1.0</small>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("ℹ️ Kamus Informasi Fitur")
st.sidebar.markdown("""
* **Total Spent**: Akumulasi biaya belanja finansial pelanggan selama ini.
* **Satisfaction Score**: Penilaian kepuasan pelanggan terhadap kualitas layanan (1-5).
* **Support Tickets**: Jumlah komplain/aduan masalah yang pernah dikirim.
* **Age**: Variabel umur atau usia pelanggan saat ini.
""")

# fungsi load komponen binary pkl model
@st.cache_resource
def muat_komponen_final():
    with open('model_churn_final.pkl', 'rb') as file:
        return pickle.load(file)

try:
    komponen = muat_komponen_final()
    model = komponen['model_final']
    imputer_num = komponen['imputer_numerik']
    scaler = komponen['scaler_standard']
    imputer_cat = komponen['imputer_kategorikal']
    encoder = komponen['encoder_onehot']
    kolom_numerik = komponen['kolom_numerik']
    kolom_kategorikal = komponen['kolom_kategorikal']
except FileNotFoundError:
    st.error("Error: Berkas 'model_churn_final.pkl' tidak ditemukan di folder yang sama dengan app.py!")
    st.stop()

# mapping manual relasi kota dan negara
mapping_negara_kota = {
    'Bangladesh': ['Dhaka', 'Chittagong', 'Khulna'],
    'Germany': ['Berlin', 'Munich', 'Frankfurt', 'Hamburg'],
    'India': ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad'],
    'UK': ['London', 'Manchester', 'Birmingham', 'Liverpool'],
    'USA': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

# config metadata parameter form input (skala dollar)
panduan_numerik = {
    'age': {'label': 'Usia Pelanggan', 'default': 30.0, 'min': 18.0, 'max': 100.0, 'step': 1.0, 'help': 'Satuan: Tahun. Masukkan rentang usia pelanggan aktif saat ini (Contoh: 18 - 80).'},
    'is_premium_user': {'label': 'Status Akun Premium', 'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 1.0, 'help': 'Isi angka 1 jika berlangganan akun premium, isi angka 0 jika hanya akun standar.'},
    'total_visits': {'label': 'Total Kunjungan Situs', 'default': 5.0, 'min': 0.0, 'max': 1000.0, 'step': 1.0, 'help': 'Satuan: Kali. Akumulasi jumlah kedatangan pelanggan ke dalam aplikasi atau website.'},
    'avg_session_time': {'label': 'Rata-rata Durasi Sesi', 'default': 15.0, 'min': 0.0, 'max': 1440.0, 'step': 0.5, 'help': 'Satuan: Menit. Rata-rata lamanya waktu yang dihabiskan pelanggan di setiap sesi kunjungan.'},
    'pages_per_session': {'label': 'Jumlah Halaman Terbuka per Sesi', 'default': 4.0, 'min': 0.0, 'max': 200.0, 'step': 1.0, 'help': 'Satuan: Halaman. Rata-rata jumlah menu atau halaman produk yang diklik (Rentang normal: 1 - 15).'},
    'email_open_rate': {'label': 'Rasio Membuka Email Promosi', 'default': 0.20, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'help': 'Format: Desimal (0.00 sampai 1.00). Persentase seberapa sering pelanggan membuka email iklan (Contoh: 0.20 artinya 20%).'},
    'email_click_rate': {'label': 'Rasio Klik Tautan di Dalam Email', 'default': 0.05, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'help': 'Format: Desimal (0.00 sampai 1.00). Persentase klik link di dalam email. Nilai ini harus lebih kecil atau sama dengan rasio buka email.'},
    'total_spent': {'label': 'Total Nilai Pengeluaran Finansial', 'default': 500.0, 'min': 0.0, 'max': 50000.0, 'step': 10.0, 'help': 'Satuan: Dollar (USD). Total akumulasi nominal uang yang sudah dibelanjakan pelanggan selama ini.'},
    'avg_order_value': {'label': 'Rata-rata Nominal per Transaksi', 'default': 50.0, 'min': 0.0, 'max': 5000.0, 'step': 5.0, 'help': 'Satuan: Dollar (USD). Rata-rata nilai belanja yang dikeluarkan di setiap satu kali checkout transaksi.'},
    'discount_used': {'label': 'Jumlah Penggunaan Potongan Diskon', 'default': 0.0, 'min': 0.0, 'max': 100.0, 'step': 1.0, 'help': 'Satuan: Kali. Frekuensi pelanggan menggunakan kode voucher potongan harga saat melakukan pembelian.'},
    'support_tickets': {'label': 'Jumlah Tiket Aduan / Komplain', 'default': 0.0, 'min': 0.0, 'max': 50.0, 'step': 1.0, 'help': 'Satuan: Tiket. Jumlah keluhan atau laporan masalah teknis yang pernah diajukan ke customer service.'},
    'refund_requested': {'label': 'Jumlah Pengajuan Refund / Pengembalian Dana', 'default': 0.0, 'min': 0.0, 'max': 20.0, 'step': 1.0, 'help': 'Satuan: Kali. Total frekuensi pelanggan meminta pengembalian uang akibat pembatalan atau retur.'},
    'delivery_delay_days': {'label': 'Rata-rata Keterlambatan Pengiriman', 'default': 0.0, 'min': 0.0, 'max': 30.0, 'step': 1.0, 'help': 'Satuan: Hari. Rata-rata durasi keterlambatan pengiriman paket pesanan yang dialami pelanggan.'},
    'satisfaction_score': {'label': 'Skor Kepuasan Pelanggan', 'default': 3.0, 'min': 1.0, 'max': 5.0, 'step': 1.0, 'help': 'Skala: 1 sampai 5. Nilai kepuasan dari pelanggan (1: Sangat Kecewa, 3: Biasa Saja, 5: Sangat Puas).'},
    'nps_score': {'label': 'Net Promoter Score / Loyalitas', 'default': 7.0, 'min': 0.0, 'max': 10.0, 'step': 1.0, 'help': 'Skala: 0 sampai 10. Mengukur loyalitas pelanggan dan kesediaan mereka merekomendasikan produk ke orang lain (0: Buruk, 10: Loyal).'},
    'marketing_spend_per_user': {'label': 'Biaya Pemasaran per Pengguna', 'default': 25.0, 'min': 0.0, 'max': 2000.0, 'step': 1.0, 'help': 'Satuan: Dollar (USD). Besarnya biaya iklan yang dikeluarkan perusahaan untuk menyasar profil pelanggan ini.'},
    'lifetime_value': {'label': 'Customer Lifetime Value', 'default': 1500.0, 'min': 0.0, 'max': 100000.0, 'step': 10.0, 'help': 'Satuan: Dollar (USD). Estimasi total nilai keuntungan finansial yang bisa dihasilkan pelanggan selama masa kontrak berlangganan.'},
    'last_3_month_purchase_freq': {'label': 'Frekuensi Belanja 3 Bulan Terakhir', 'default': 2.0, 'min': 0.0, 'max': 100.0, 'step': 1.0, 'help': 'Satuan: Kali. Jumlah total transaksi pembelian sukses yang dilakukan pelanggan dalam kurun waktu 90 hari terakhir.'}
}

# menggunakan subheader bawaan streamlit agar otomatis adaptasi warna font di dark mode
st.subheader("📥 Form Entri Karakteristik Profil Pelanggan")

data_inputan = {}
col_form1, col_form2 = st.columns(2, gap="large")

with col_form1:
    st.markdown("<div class='card-box'><h4>🔢 Atribut Metrik & Finansial</h4>", unsafe_allow_html=True)
    for col in kolom_numerik:
        if col in panduan_numerik:
            p = panduan_numerik[col]
            data_inputan[col] = st.number_input(
                label=p['label'], value=p['default'],
                min_value=p['min'], max_value=p['max'], step=p['step']
            )
            st.caption(f"ℹ️ {p['help']}")
        else:
            data_inputan[col] = st.number_input(f"Masukkan nilai untuk {col}", value=0.0)
    st.markdown("</div>", unsafe_allow_html=True)

with col_form2:
    st.markdown("<div class='card-box' style='border-left-color: #06b6d4;'><h4>🔤 Atribut Geografis & Segmentasi</h4>", unsafe_allow_html=True)
    
    indeks_country = list(kolom_kategorikal).index('country') if 'country' in kolom_kategorikal else None
    indeks_city = list(kolom_kategorikal).index('city') if 'city' in kolom_kategorikal else None
    
    opsi_negara = list(encoder.categories_[indeks_country]) if indeks_country is not None else []
    negara_terpilih = st.selectbox("Negara Asal (Country)", opsi_negara)
    data_inputan['country'] = negara_terpilih
    st.caption("ℹ️ Lokasi negara pendaftaran kontrak utama pelanggan.")
    
    if indeks_city is not None:
        semua_kota_encoder = list(encoder.categories_[indeks_city])
        kota_sesuai_negara = mapping_negara_kota.get(negara_terpilih, [])
        opsi_kota_terfilter = [k for k in kota_sesuai_negara if k in semua_kota_encoder]
        if not opsi_kota_terfilter:
            opsi_kota_terfilter = semua_kota_encoder
            
        kota_terpilih = st.selectbox("Kota Domisili (City)", opsi_kota_terfilter)
        data_inputan['city'] = kota_terpilih
        st.caption("ℹ️ Kota domisili operasional yang terfilter otomatis berdasarkan negara terpilih.")

    for i, col in enumerate(kolom_kategorikal):
        if col not in ['country', 'city']:
            pilihan_opsi = list(encoder.categories_[i])
            data_inputan[col] = st.selectbox(f"Kategori {col}", pilihan_opsi)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)
tombol_ditekan = st.button("🚀 Jalankan Analisis Keputusan Model", type="primary")
st.markdown("</div>", unsafe_allow_html=True)

# eksekusi pipeline kalkulasi machine learning
if tombol_ditekan:
    df_mentah = pd.DataFrame([data_inputan])
    df_num = df_mentah[kolom_numerik]
    df_cat = df_mentah[kolom_kategorikal]
    
    # transformasi matriks data
    X_num_imputed = imputer_num.transform(df_num)
    X_num_scaled = scaler.transform(X_num_imputed)
    X_cat_imputed = imputer_cat.transform(df_cat)
    X_cat_encoded = encoder.transform(X_cat_imputed)
    if hasattr(X_cat_encoded, 'toarray'):
        X_cat_encoded = X_cat_encoded.toarray()
        
    X_final_siap = np.hstack([X_num_scaled, X_cat_encoded])
    
    hasil_tebakan = model.predict(X_final_siap)[0]
    peluang = model.predict_proba(X_final_siap)[0]
    
    st.subheader("🎯 Hasil Output Analisis Intelijen Model")
    
    col_res1, col_res2 = st.columns([4, 6], gap="large")
    
    with col_res1:
        st.markdown("<div class='card-box' style='border-left-color: #10b981; height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<h4>Status Prediksi Akhir:</h4>", unsafe_allow_html=True)
        
        if hasil_tebakan == 1:
            st.error("⚠️ PELANGGAN DIPREDIKSI CHURN (BERPOTENSI KABUR)")
            st.markdown(f"""
                <div class='metric-container' style='border: 1px solid #dc2626;'>
                    <span style='font-size: 14px; color: #fca5a5;'>Confidence Rate Churn</span><br/>
                    <b style='font-size: 24px; color: #ef4444;'>{peluang[1] * 100:.2f}%</b>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ PELANGGAN DIPREDIKSI LOYAL (TETAP BERTAHAN)")
            st.markdown(f"""
                <div class='metric-container' style='border: 1px solid #10b981;'>
                    <span style='font-size: 14px; color: #a7f3d0;'>Confidence Rate Loyal</span><br/>
                    <b style='font-size: 24px; color: #10b981;'>{peluang[0] * 100:.2f}%</b>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_res2:
        st.markdown("<div class='card-box' style='border-left-color: #f59e0b; height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<h4>📊 Komparasi Probabilitas Hasil Analisis</h4>", unsafe_allow_html=True)
        df_grafik = pd.DataFrame({
            'Status Pelanggan': ['Loyal (Bertahan)', 'Churn (Kabur)'],
            'Persentase Kemungkinan (%)': [peluang[0] * 100, peluang[1] * 100]
        })
        st.bar_chart(data=df_grafik, x='Status Pelanggan', y='Persentase Kemungkinan (%)', use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # expander transparansi data/behind the scenes
    st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
    with st.expander("🔍 Audit Trail Pipeline Data (Behind the Scenes)"):
        st.markdown("**1. Representasi Struktur Data Mentah Single DataFrame:**")
        st.dataframe(df_mentah)
        st.markdown("**2. Nilai Hasil Standardisasi Z-Score Fitur Numerik:**")
        st.code(str(X_num_scaled))
        st.markdown("**3. Output Vektor Transformasi Biner Kategorikal (One-Hot Encoded Matrix):**")
        st.code(str(X_cat_encoded))
        st.markdown("**4. Dimensi Akhir Representasi Array ke Algoritma Random Forest Classifier:**")
        st.info(f"Jumlah Baris: {X_final_siap.shape[0]} Baris | Total Dimensi Kolom Input: {X_final_siap.shape[1]} Kolom")
    st.markdown("</div>", unsafe_allow_html=True)