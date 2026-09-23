# StayReco Recommender — Thực nghiệm

## Mục tiêu
So sánh `score = α·CF(SVD++) + β·Content + γ·Demographic → Knowledge filter`
với các baseline để chứng minh α động tốt hơn α cố định.

## Cấu trúc project
```
stayreco-experiment/
├── data/                  # Load & tiền xử lý dataset
│   ├── __init__.py
│   ├── loader.py          # Download & load dataset
│   └── preprocessor.py    # Tách train/test, tạo cold-start scenarios
├── models/                # Các recommender
│   ├── __init__.py
│   ├── cf_svdpp.py        # Collaborative Filtering (SVD++)
│   ├── content_based.py   # Content-Based (embedding + cosine)
│   ├── demographic.py     # Demographic fallback
│   ├── knowledge.py       # Knowledge pre-filter
│   └── hybrid.py          # Hybrid: trộn α·CF + β·Content + γ·Demo
├── evaluation/            # Đo lường
│   ├── __init__.py
│   └── metrics.py         # RMSE, MAE, Precision@K, NDCG
├── utils/
│   ├── __init__.py
│   └── alpha_scheduler.py # Logic α động theo n
├── results/               # Output CSV, biểu đồ
├── notebooks/             # Jupyter notebooks để phân tích
├── run_experiment.py      # Script chạy toàn bộ thí nghiệm
├── config.py              # Cấu hình hyperparameters
└── requirements.txt
```

## Thí nghiệm cần chạy
1. **CF only** (α=1, β=0, γ=0)
2. **Content only** (α=0, β=1, γ=0)
3. **Demographic only** (α=0, β=0, γ=1)
4. **Hybrid α=0.3** cố định
5. **Hybrid α=0.5** cố định
6. **Hybrid α=0.7** cố định
7. **Hybrid α động** (theo n) ← cách của StayReco

## Cách chạy
```bash
pip install -r requirements.txt
python run_experiment.py
```
