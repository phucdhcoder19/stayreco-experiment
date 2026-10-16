# StayReco Recommender — Thực nghiệm

Capstone FA26SE056 (FPT University HCMC). Chứng minh **α động tốt hơn α cố định**
cho bài toán cold-start hai phía của nền tảng đặt phòng mới.

```
Knowledge pre-filter → score = α(n)·CF + β(n)·Content + γ(n)·Demographic → Top-K
```

## Cài đặt

Thí nghiệm chạy trong conda env `stayreco`:

```bash
conda env create -f environment.yml      # lần đầu
conda activate stayreco                  # ⚠️ BẮT BUỘC, xem ghi chú dưới
pytest tests/ -q                          # kiểm tra trước khi chạy
```

> ⚠️ **Phải `conda activate` trước khi chạy.** Gọi thẳng
> `C:\...\envs\stayreco\python.exe` sẽ làm numpy crash ở phép nhân ma trận dày
> (`a @ b`), vì các DLL của MKL nằm trong `Library/bin` và chỉ được đưa vào
> đường dẫn tìm DLL khi env được activate. Lỗi hiện ra dưới dạng tiến trình chết
> không traceback (Windows exception `0xc06d007f`). Nếu cần chạy không activate,
> dùng `conda run -n stayreco python ...`.

## Cách chạy

```bash
# Thí nghiệm chính: 10 seed, 20% hotel bị giấu toàn bộ rating
python run_experiment.py 42 1 2 3 4 5 6 7 8 9 --cold-item=0.2 --tag=fullcat

# So cặp từng seed + kiểm định Wilcoxon
python analyze_seeds.py _fullcat              # trên RMSE
python analyze_seeds.py _fullcat ndcg@10      # trên NDCG@10

# SVD++ có thật sự hơn SVD không (không cần train lại)
python compare_svd_svdpp.py _fullcat
```

Tham số dòng lệnh:

| Cờ | Ý nghĩa |
|---|---|
| `42 1 2 ...` | danh sách seed; mỗi seed là một cách chia dữ liệu |
| `--cold-item=0.2` | % hotel bị giấu toàn bộ rating (Schein et al. 2002 dùng ~0.2) |
| `--tag=fullcat` | hậu tố file kết quả, để không ghi đè lần chạy trước |

## Hai cách đo — đọc kỹ chỗ này

Thí nghiệm báo cáo **hai nhóm metric riêng biệt**, vì chúng trả lời hai câu khác nhau:

| Nhóm | Metric | Câu hỏi | Rổ ứng viên |
|---|---|---|---|
| Sai số dự đoán | RMSE, MAE | đoán số sao chính xác đến đâu? | từng cặp (user, hotel) độc lập |
| Chất lượng xếp hạng | P@K, R@K, NDCG@K, Cov@K, New@K | chọn K hotel để hiển thị có đúng không? | **toàn bộ catalog** user chưa đặt |

Nhóm thứ hai **phải** xếp hạng trên toàn catalog. Cách làm sai phổ biến là xếp
hạng chỉ trong số item user đã rating trong test set — rổ đó chỉ ~20 item và ~55%
trong đó là item user thích, nên đoán bừa đã được NDCG@10 ≈ 0.84 và mọi phương
pháp dồn vào dải 0.88–0.89. Tệ hơn: hotel mới chưa ai rating không bao giờ có
trong rổ, nên cold-start phía hotel không được đo gì cả.

Cơ sở: Krichene & Rendle, *On Sampled Metrics for Item Recommendation* (KDD 2020).
Chi tiết và số liệu đối chứng: xem đầu file [`evaluation/metrics.py`](evaluation/metrics.py)
và [`evaluation/ranking.py`](evaluation/ranking.py).

`New@K` = tỷ lệ ô hiển thị dành cho hotel chưa ai rating. **Không phải càng cao
càng tốt** — đọc cùng P@K: tốt = giữ được Precision mà vẫn chia chỗ cho hotel mới.

## Các phương pháp được so sánh

Nhóm A — cùng 3 model thành phần, chỉ khác cách trộn:
Popularity · CF only (SVD++) · Content only · Demographic only ·
Hybrid α=0.3/0.5/0.7 cố định · **StayReco α động** (bản thủ công và bản tune)

Nhóm B — dựng lại 4 bài báo trong cùng khung thực nghiệm:
Tarek 2023 · Do & Le 2020 (2 và 3 thành phần) · OHWSF 2022 · SL-ARec 2026

> Đây là **bản thích nghi**, không phải bản sao nguyên gốc. Mỗi bài dùng dataset,
> cách chia và metric riêng, nên copy số trong bài báo rồi so với số của mình là
> sai. Chỗ nào phải sửa để chạy được đều ghi trong docstring
> [`models/paper_baselines.py`](models/paper_baselines.py).

Mọi phương pháp đều được grid search trên **validation** theo RMSE — như nhau cho
tất cả. Tune trên test là gian lận số liệu (Dacrema et al., RecSys 2019).

## Cấu trúc

```
stayreco-experiment/
├── config.py               # hằng số; giá trị THẬT của mỗi lần chạy ở run_meta_<tag>.json
├── run_experiment.py       # chạy toàn bộ thí nghiệm
├── analyze_seeds.py        # so cặp từng seed + Wilcoxon signed-rank
├── compare_svd_svdpp.py    # SVD++ vs SVD, tách theo nhóm cold-start
├── data/
│   ├── loader.py           # MovieLens; hook sẵn cho HotelRec/Expedia
│   └── preprocessor.py     # split train/val/test + mô phỏng cold-start
├── models/
│   ├── cf_svdpp.py         # SVD++ và SVD (Surprise)
│   ├── cf_itemknn.py       # Item-Item CF (cho Tarek 2023)
│   ├── lightgcn.py         # LightGCN (cho SL-ARec 2026)
│   ├── content_based.py    # TF-IDF + cosine
│   ├── demographic.py      # trung bình theo nhóm nhân khẩu
│   ├── popularity.py       # mốc tham chiếu thấp nhất
│   ├── knowledge.py        # pre-filter business rules
│   ├── hybrid.py           # trộn 3 thành phần
│   └── paper_baselines.py  # dựng lại 4 bài báo
├── evaluation/
│   ├── metrics.py          # RMSE, MAE (+ metric cũ, đã đánh dấu cảnh báo)
│   └── ranking.py          # P/R/NDCG/Cov@K trên TOÀN CATALOG  ← dùng cái này
├── utils/
│   ├── alpha_scheduler.py  # công thức α động (đóng góp chính)
│   ├── tuning.py           # grid search trên validation
│   ├── batch.py            # chấm điểm vectorized
│   └── manifest.py         # ghi run_meta_<tag>.json
├── tests/                  # pytest — chạy trước mỗi lần chạy thí nghiệm
└── results/                # CSV, PNG, run_meta_<tag>.json
```

## Output trong `results/`

| File | Nội dung |
|---|---|
| `run_meta_<tag>.json` | **truy xuất nguồn gốc**: commit, config thật, phiên bản thư viện |
| `comparison_<tag>.csv` | metrics tổng thể, mỗi dòng một (phương pháp, seed) |
| `comparison_by_group_<tag>.csv` | metrics theo nhóm cold-start |
| `chosen_params_<tag>.csv` | tham số grid search đã chọn |
| `score_table_seed*_<tag>.csv` | điểm từng model cho từng cặp test (phân tích lại không cần train) |
| `comparison_<tag>.png` | biểu đồ theo nhóm |
| `sensitivity_alpha_<tag>.png` | RMSE validation thay đổi theo α_max |

## Trạng thái

Đã có: MovieLens 100K, 14 phương pháp, 10 seed, xếp hạng toàn catalog, kiểm định
Wilcoxon, test cho vectorization / metric / công thức α.

Còn thiếu: dataset hotel thật (HotelRec, Expedia) — `data/loader.py` đã có hook;
implicit feedback thật (view/click) để biện minh cho việc chọn SVD++;
Knowledge pre-filter chưa vào vòng đánh giá (MovieLens không có budget/phòng trống).
