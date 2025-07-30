# Optical Character Recognition (OCR) Plat Nomor Kendaraan dengan Visual Language Model (VLM)

Proyek ini adalah implementasi Optical Character Recognition (OCR) untuk plat nomor kendaraan menggunakan Visual Language Model (VLM) yang dijalankan secara lokal melalui LMStudio dan diintegrasikan dengan bahasa pemrograman Python. Tujuan utama dari proyek ini adalah untuk mengenali karakter pada plat nomor dari gambar dan mengevaluasi akurasi pengenalan tersebut menggunakan metrik Character Error Rate (CER).

## Daftar Isi

- [Konsep VLM dan Penerapannya untuk OCR](#konsep-vlm-dan-penerapannya-untuk-ocr)
- [Arsitektur Proyek: Integrasi LMStudio dan Python](#arsitektur-proyek-integrasi-lmstudio-dan-python)
- [Proses Inferensi dan Evaluasi Metrik CER](#proses-inferensi-dan-evaluasi-metrik-cer)
- [Analisis Hasil CER: Contoh Sukses dan Gagal](#analisis-hasil-cer-contoh-sukses-dan-gagal)
- [Kesimpulan dan Analisis Mendalam Hasil Proyek](#kesimpulan-dan-analisis-mendalam-hasil-proyek)
- [Potensi Pengembangan dan Rekomendasi](#potensi-pengembangan-dan-rekomendasi)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Cara Eksekusi](#cara-eksekusi)
- [Lisensi](#lisensi)
- [Kontributor](#kontributor)

---

## Konsep VLM dan Penerapannya untuk OCR

**Visual Language Model (VLM)** adalah jenis model kecerdasan buatan yang mampu memahami dan memproses informasi dari dua modalitas atau lebih secara bersamaan, dalam kasus ini, gambar dan teks. Berbeda dengan model OCR tradisional yang hanya fokus pada pengenalan karakter, VLM memiliki kemampuan untuk menginterpretasikan konteks visual dari gambar secara keseluruhan, yang memungkinkan peningkatan akurasi pengenalan teks.

Dalam proyek ini, VLM digunakan untuk "membaca" plat nomor kendaraan. Dengan kemampuan multimodal-nya, VLM dapat lebih baik dalam memahami dan mengenali karakter pada plat nomor karena ia tidak hanya melihat karakter itu sendiri, tetapi juga konteks visual di sekitarnya (misalnya, bentuk plat, posisi di kendaraan, dll.).

---

## Arsitektur Proyek: Integrasi LMStudio dan Python

Proyek ini dirancang dengan arsitektur yang memanfaatkan **LMStudio** sebagai *runtime* lokal untuk model VLM dan **Python** sebagai bahasa pemrograman utama untuk mengorkestrasi seluruh alur kerja.

* **LMStudio:** Ini adalah *platform* fleksibel yang memungkinkan pengguna untuk mengunduh dan menjalankan berbagai Large Language Models (LLMs) dan Visual Language Models (VLMs) secara lokal di mesin mereka. LMStudio menyediakan API lokal (biasanya di `http://localhost:1234`) yang memudahkan interaksi dengan model melalui kode program. Keunggulan utamanya adalah kemampuan untuk menjalankan model *offline*, menjaga privasi data, dan memanfaatkan sumber daya lokal.
* **Model VLM yang Digunakan:** Model spesifik yang dimanfaatkan dalam proyek ini adalah **`llava-v1.6-mistral-7b.Q3_K_XS.gguf`**. Model ini adalah varian dari arsitektur Llava, yang dikenal memiliki kemampuan multimodal (memproses gambar dan teks). Penting untuk dicatat bahwa dalam proyek ini, model dijalankan di LMStudio dengan opsi `--gpu off`, artinya seluruh proses inferensi dilakukan menggunakan **Central Processing Unit (CPU)**, bukan Graphics Processing Unit (GPU).
* **Python:** Bahasa pemrograman Python bertindak sebagai "otak" di balik proyek ini. Fungsinya meliputi:
    * Membaca gambar-gambar plat nomor dari direktori dataset.
    * Mengkodekan setiap gambar ke format Base64, yang merupakan format yang dapat diterima oleh API LMStudio untuk input gambar.
    * Mengirimkan permintaan (request) HTTP POST yang berisi *prompt* tekstual dan gambar Base64 ke API lokal LMStudio.
    * Menerima dan memproses respons yang berisi prediksi teks dari model.
    * Melakukan pembersihan (post-processing) pada hasil prediksi untuk menghilangkan karakter atau teks yang tidak relevan.
    * Menghitung metrik evaluasi Character Error Rate (CER) untuk setiap prediksi.
    * Menyimpan seluruh hasil (nama gambar, *ground truth*, prediksi, CER) ke dalam file CSV untuk analisis lebih lanjut.

---

## Proses Inferensi dan Evaluasi Metrik CER

### Proses Inferensi (Praktik Implementasi)

Proses pengenalan plat nomor kendaraan dilakukan secara otomatis oleh skrip Python (`ptestocr.py`) yang berinteraksi dengan LMStudio:

1.  **Inisiasi Server LMStudio:** Sebelum menjalankan skrip, LMStudio harus sudah berjalan sebagai server lokal (`http://localhost:1234`) dengan model `llava-v1.6-mistral-7b.Q3_K_XS.gguf` yang dimuat. Penekanan pada `--gpu off` berarti seluruh beban komputasi ditangani oleh CPU.
2.  **Iterasi Dataset Gambar:** Skrip Python secara berurutan memproses setiap file gambar plat nomor yang ditemukan di folder `test/`.
3.  **Encoding Gambar:** Untuk setiap gambar, fungsi `encode_image_to_base64` digunakan untuk mengonversi data biner gambar menjadi string Base64. Ini adalah standar untuk mengirimkan data gambar melalui API berbasis teks.
4.  **Konstruksi Prompt & Payload:** Sebuah *prompt* tekstual yang sangat spesifik (`"What is the license plate number shown in this image? Respond only with the plate number without any additional text or explanation."`) digabungkan dengan gambar Base64 untuk membentuk payload permintaan API. Prompt ini dirancang untuk memandu model agar memberikan output yang ringkas dan langsung ke poin.
5.  **Permintaan API ke LMStudio:** Menggunakan pustaka `requests` Python, permintaan POST dikirimkan ke endpoint `/v1/chat/completions` LMStudio. Model `llava-v1.6-mistral-7b.Q3_K_XS.gguf` yang dimuat di LMStudio kemudian melakukan inferensi.
6.  **Pembersihan Prediksi (Post-processing):** Setelah menerima respons dari LMStudio, fungsi `clean_prediction` pada skrip Python diterapkan. Fungsi ini bertanggung jawab untuk menghapus karakter yang tidak perlu (seperti tanda kutip atau spasi berlebih) dan menggunakan *regular expression* (`r'[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{1,3}'`) untuk mencoba mengekstrak hanya bagian yang sesuai dengan format plat nomor Indonesia, sekaligus mengonversi ke huruf kapital dan menghapus spasi di dalamnya. Ini membantu menyaring respons model yang mungkin tidak sempurna.

### Evaluasi Menggunakan Character Error Rate (CER)

Untuk mengukur efektivitas dan akurasi pengenalan karakter, metrik **Character Error Rate (CER)** digunakan. CER sangat relevan untuk tugas OCR karena ia menghitung kesalahan pada level karakter, bukan kata secara keseluruhan.

Formula CER dihitung sebagai:

$$
CER = \frac{(S + D + I)}{N}
$$

Dimana:
* $S$ = Jumlah karakter salah **Substitusi** (misalnya, model memprediksi '8' padahal seharusnya 'B').
* $D$ = Jumlah karakter yang **Dihapus** atau tidak terdeteksi oleh model (misalnya, plat 'ABC' diprediksi 'AC', 'B' adalah *deletion*).
* $I$ = Jumlah karakter yang **Disisipkan** atau diprediksi secara berlebihan oleh model (misalnya, plat 'AB' diprediksi 'AXB', 'X' adalah *insertion*).
* $N$ = Jumlah karakter pada *ground truth* (plat nomor yang sebenarnya).

Nilai CER yang semakin rendah (mendekati 0) menunjukkan akurasi pengenalan karakter yang lebih tinggi. Sebuah CER 0.0 berarti prediksi sempurna tanpa kesalahan karakter.

---

## Analisis Hasil CER: Contoh Sukses dan Gagal

Hasil eksekusi program disimpan dalam `ocr_results.csv` dan dirangkum dalam *output* CMD.

### Contoh Kasus Sukses

Pada kasus-kasus di mana gambar plat nomor memiliki kualitas yang baik (jelas, pencahayaan cukup, tidak ada oklusi, sudut pandang optimal), model `llava-v1.6-mistral-7b.Q3_K_XS.gguf` mampu memprediksi plat nomor dengan sangat akurat, seringkali mencapai **CER 0.0**.

**Contoh:**
