import pandas as pd
import re
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# --- KONFIGURASI ---
KOLOM_TEKS = 'tweet'
KOLOM_LABEL = 'label'
KAMUS_EMOSI = {
    0: "Positif / Netral",
    1: "NEGATIF / KASAR (OFFENSIVE) ⚠️"
}

print("\n" + "="*40)
print("   SISTEM ANALISIS EMOSI (BALANCED)")
print("="*40)

# 1. MEMUAT DATA
print("[1/5] Memuat Data...")
try:
    df = pd.read_csv('data_sentimen.csv', encoding='latin-1')
    df = df.dropna(subset=[KOLOM_TEKS, KOLOM_LABEL])
    df[KOLOM_LABEL] = df[KOLOM_LABEL].astype(int)
except Exception as e:
    print(f"[ERROR] Gagal memuat data: {e}")
    sys.exit()

# 2. MENYEIMBANGKAN DATA (BAGIAN PENTING BARU)
print(f"   -> Total Data Awal: {len(df)}")
df_negatif = df[df[KOLOM_LABEL] == 1]
df_positif = df[df[KOLOM_LABEL] == 0]

# Kita ambil data positif SEJUMLAH data negatif saja (biar adil 50:50)
jumlah_sampel = len(df_negatif)
df_positif_sample = df_positif.sample(n=jumlah_sampel, random_state=42)

# Gabungkan kembali
df_seimbang = pd.concat([df_negatif, df_positif_sample])
print(f"   -> Setelah Diseimbangkan: {len(df_seimbang)} baris data")
print(f"      (Positif: {len(df_positif_sample)}, Negatif: {len(df_negatif)})")

# 3. PEMBERSIHAN DATA
print("[2/5] Membersihkan Data...")
def bersihkan_teks(teks):
    teks = str(teks).lower()
    teks = re.sub(r'@[A-Za-z0-9]+', '', teks) 
    teks = re.sub(r'#', '', teks)
    teks = re.sub(r'https?:\/\/\S+', '', teks)
    teks = re.sub(r'[^a-z\s]', '', teks)
    return teks

df_seimbang['teks_bersih'] = df_seimbang[KOLOM_TEKS].apply(bersihkan_teks)

# 4. PEMBAGIAN & VEKTORISASI
print("[3/5] Melatih AI dengan Data Seimbang...")
X = df_seimbang['teks_bersih']
y = df_seimbang[KOLOM_LABEL]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizer (Max features dikurangi sedikit agar lebih fokus)
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. MELATIH MODEL
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluasi
prediksi = model.predict(X_test_vec)
akurasi = accuracy_score(y_test, prediksi)
print(f"\n-> Pelatihan Selesai! Akurasi: {akurasi * 100:.2f}%")
print("(Akurasi mungkin turun sedikit dari 94%, tapi AI jadi lebih peka)")

# ==========================================
# INTERAKSI
# ==========================================
print("\n" + "="*50)
print("  AI SIAP! (Ketik 'keluar' untuk berhenti)")
print("  Coba tes kata kasar Inggris: 'stupid', 'racist', 'ugly'")
print("="*50)

while True:
    try:
        kalimat_user = input("\nMasukkan teks (Inggris): ")
        
        if kalimat_user.lower() == 'keluar':
            break
            
        if not kalimat_user.strip():
            continue

        kalimat_bersih = bersihkan_teks(kalimat_user)
        kalimat_vec = vectorizer.transform([kalimat_bersih])
        
        hasil_angka = model.predict(kalimat_vec)[0]
        
        # Tampilkan Probabilitas (Keyakinan AI)
        probabilitas = model.predict_proba(kalimat_vec)[0]
        persen_negatif = probabilitas[1] * 100
        
        hasil_teks = KAMUS_EMOSI.get(hasil_angka, "Tidak Diketahui")
        
        print(f"-> Analisa: [{hasil_teks}]")
        print(f"-> Tingkat Negatif: {persen_negatif:.1f}%")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")