# Classification-of-Indonesian-s-Traditional-Food
# Klasifikasi Citra Makanan Tradisional Indonesia 🇮🇩🍱


## 📖 Deskripsi Proyek

Proyek ini bertujuan untuk membangun model Deep Learning yang mampu mengklasifikasikan citra makanan tradisional Indonesia secara otomatis. Sistem ini diharapkan dapat membantu dalam pengenalan kuliner lokal melalui teknologi visi komputer (*Computer Vision*).

Sistem ini dikembangkan untuk mengenali 5 kelas makanan populer:
1. **Gado-Gado**
2. **Nasi Goreng**
3. **Nasi Padang**
4. **Rendang**
5. **Sate**

Proyek ini membandingkan kinerja antara model CNN yang dibangun dari awal (*scratch*) dengan teknik *Transfer Learning* menggunakan arsitektur populer.

---

## 📂 Dataset & Preprocessing

### Sumber Data
Dataset yang digunakan bersumber dari Kaggle: **[Indonesian Food Dataset](https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification)** 

### Preprocessing & Penyeimbangan Data (Augmentasi)
Berdasarkan analisis kode, ditemukan ketidakseimbangan jumlah data awal. Oleh karena itu, dilakukan **Augmentasi Data kustom** untuk menyeimbangkan kelas hingga mencapai target **1000 gambar per kelas**.

**Teknik Augmentasi yang digunakan:**
* Rotation Range: 30
* Width & Height Shift: 0.1
* Zoom Range: 0.2
* Shear Range: 0.15
* Horizontal Flip: True

**Konfigurasi Input Model:**
* **Ukuran Citra (Image Size):** $160 \times 160$ pixel.
* **Rescaling:** Normalisasi nilai piksel $1./255$.
* **Splitting:** Data dibagi menjadi `train`, `valid`, dan `test` set.

---

## 🧠 Arsitektur Model

Proyek ini mengimplementasikan tiga skenario model menggunakan TensorFlow/Keras:

### 1. Model 1 — CNN Base (Custom Architecture)
Model CNN sederhana yang dibangun dari awal (*scratch*):
* 3 Blok Konvolusi: (`Conv2D` + `ReLU` + `MaxPooling2D`) dengan filter 32, 64, dan 128.
* **Classifier:** `Flatten` -> `Dense` (256 unit) -> `Dropout` (0.5) -> `Output` (5 unit, Softmax).
* **Optimizer:** Adam.

### 2. Model 2 — Transfer Learning (MobileNetV2)
Menggunakan *weights* dari ImageNet dengan arsitektur *lightweight*:
* **Base Model:** MobileNetV2 (Frozen / Trainable = False).
* **Top Layers:** `GlobalAveragePooling2D` -> `Dense` (128 unit, ReLU) -> `Dropout` (0.5) -> `Output` (5 unit, Softmax).
* **Karakteristik:** Model ringan, pelatihan cepat, dan efisien.

### 3. Model 3 — Transfer Learning (ResNet50)
Menggunakan arsitektur residual yang lebih dalam:
* **Base Model:** ResNet50 (Frozen / Trainable = False).
* **Top Layers:** `GlobalAveragePooling2D` -> `Dense` (256 unit, ReLU) -> `Dropout` (0.5) -> `Output` (5 unit, Softmax).
* **Karakteristik:** Model sangat dalam, membutuhkan komputasi lebih besar.

---

## 📊 Hasil Evaluasi & Analisis

Model dilatih selama **10 Epochs** dan dievaluasi menggunakan *Classification Report* dan *Confusion Matrix* pada data uji.

| Model | Akurasi | Macro F1-Score | Analisis |
| :--- | :---: | :---: | :--- |
| **CNN Base** | 72% | 0.71 | Cukup baik untuk model sederhana, namun kesulitan membedakan Nasi Goreng dan Sate. |
| **MobileNetV2** | **90%** | **0.90** | **Performa Terbaik.** Sangat akurat dan stabil di semua kelas berkat pre-trained weights yang efektif. |
| **ResNet50** | 44% | 0.42 | Performa buruk (*Underfitting*). Kemungkinan karena arsitektur terlalu kompleks untuk dataset kecil atau perlu *fine-tuning* lebih lanjut (unfreezing layers). |

**Kesimpulan:**
MobileNetV2 adalah model yang paling cocok untuk studi kasus ini karena memberikan akurasi tertinggi (90%) dengan efisiensi komputasi yang baik dibandingkan ResNet50 yang gagal beradaptasi dengan baik pada konfigurasi saat ini.


### Analisis Detail

#### 1. CNN Base
* **Performa:** Cukup memuaskan dengan akurasi **72%**.
* **Kelemahan:** Mengalami kesulitan membedakan **Nasi Goreng** (Precision rendah: 0.59) dan **Sate** (Recall rendah: 0.52). Model sering salah memprediksi kelas lain sebagai Nasi Goreng.

#### 2. MobileNetV2 (Best Model) 🏆
* **Performa:** Sangat superior dengan akurasi **90%**.
* **Kelebihan:** Nilai Precision dan Recall sangat seimbang dan tinggi (> 0.80) untuk hampir semua kelas.
* **Kelas Terbaik:** **Nasi Padang** (Precision 0.98) dan **Rendang** (Precision 0.96). Arsitektur ini sangat cocok dengan karakteristik dataset makanan ini.

#### 3. ResNet50
* **Performa:** Akurasi hanya **44%**.
* **Masalah:** Model gagal menggeneralisasi dengan baik. Precision dan Recall sangat rendah di semua kelas (terutama **Nasi Goreng** dengan Recall 0.20 dan **Sate** Recall 0.24). Hal ini kemungkinan disebabkan oleh kompleksitas model yang terlalu tinggi untuk jumlah data yang ada (*overfitting* parah) atau parameter *fine-tuning* yang belum optimal.

---

## 💻 Cara Menjalankan Sistem (Lokal)

Berikut adalah panduan untuk menjalankan aplikasi web (misalnya menggunakan Streamlit/Flask) di komputer lokal Anda.

### Prasyarat
* Python 3.8 atau lebih baru
* Git

### Langkah Instalasi

1.  **Clone Repositori ini**
    ```bash
    git clone https://github.com/dvn1sals/Classification-of-Indonesian-s-Traditional-Food.git
    cd Classification-of-Indonesian-s-Traditional-Food
    ```

2.  **Buat Virtual Environment (Opsional tapi Disarankan)**
    ```bash
    python -m venv venv
    # Untuk Windows:
    venv\Scripts\activate
    # Untuk Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Pastikan Anda memiliki file `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Aplikasi**
    Jika menggunakan **Streamlit**:
    ```bash
    streamlit run app.py
    ```
    Jika menggunakan **Flask**:
    ```bash
    python app.py
    ```

5.  **Akses Website**
    Buka browser dan kunjungi URL yang muncul di terminal (biasanya `http://localhost:8501` atau `http://127.0.0.1:5000`).

---

