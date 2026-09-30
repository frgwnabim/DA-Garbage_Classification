# Garbage Image Classification (Ebim)

Project deep learning untuk portofolio: klasifikasi foto sampah ke 6 kategori (cardboard, glass, metal, paper, plastic, trash) memakai CNN dan transfer learning. Hasil akhirnya berupa notebook analisis lengkap plus website demo yang menjalankan model langsung di browser, ditautkan dari portofolio frgwnabim.site.

## Struktur repo

```
garbage-classification/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── notebook.ipynb              # notebook utama (alur CRISP-DM)
├── src/                        # kode yang dipakai ulang oleh notebook
│   ├── data.py                 # dataset, transform, dataloader
│   ├── models.py               # definisi model
│   ├── train.py                # loop training, early stopping
│   └── evaluate.py             # metrik, confusion matrix, Grad-CAM
├── data/                       # dataset gambar, TIDAK di-commit
├── checkpoints/                # checkpoint training, TIDAK di-commit
├── models/                     # model final (.onnx) + labels.json
├── outputs/                    # JSON dan gambar hasil untuk website
└── site/                       # website demo (static, GitHub Pages)
```

## Aturan umum

- Bahasa: markdown dan komentar pakai **bahasa Indonesia**. Nama variabel dan fungsi pakai bahasa Inggris.
- Jangan pernah pakai karakter em dash (U+2014) di teks apa pun. Pakai koma, titik, titik dua, atau tanda kurung.
- Metodologi **CRISP-DM**: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment.
- Setiap langkah di notebook: sel markdown singkat (apa dan kenapa), sel kode, lalu interpretasi hasil 1 sampai 2 kalimat.
- Setiap chart wajib punya judul, label sumbu, dan interpretasi.
- Seed 42 untuk semua proses acak (random, numpy, torch, split data).
- Notebook harus lolos Restart & Run All. Setelah membuat atau mengubah notebook, jalankan dan pastikan tidak error.
- Kode yang panjang (training loop, dataset class) ditaruh di `src/` lalu di-import, supaya notebook tetap enak dibaca.
- Test set hanya dipakai **sekali** di tahap evaluasi akhir. Pemilihan model dan hyperparameter pakai validation set.

## Environment

- OS: Windows, GPU NVIDIA RTX 4060 (8 GB VRAM).
- Virtual environment `.venv` di root repo.
- Framework: **PyTorch + torchvision dengan CUDA** (bukan TensorFlow, karena TensorFlow tidak mendukung GPU di Windows native).
- Di Windows, DataLoader dengan `num_workers > 0` di dalam notebook sering bermasalah. Kalau error, pakai `num_workers=0`.
- Pakai mixed precision (`torch.amp`) untuk mempercepat training dan menghemat VRAM.

## Pemahaman untuk presentasi

Ebim akan mempresentasikan project ini ke dosen/recruiter. Setiap keputusan penting (arsitektur, augmentasi, learning rate, class weights, metrik) harus dijelaskan alasannya di markdown dengan bahasa sederhana.

## Git

- Commit per langkah yang bermakna, pesan commit bahasa Inggris (contoh: `feat: add transfer learning with MobileNetV2`).
- Jangan commit: `data/`, `checkpoints/`, `.venv/`, file > 50 MB.
