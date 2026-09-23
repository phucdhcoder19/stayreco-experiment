"""Dựng lại phương pháp của 4 bài báo để so sánh trực tiếp với StayReco.

⚠️ ĐÂY LÀ BẢN THÍCH NGHI (re-implementation), KHÔNG PHẢI BẢN SAO NGUYÊN GỐC.
Mỗi bài dùng dataset, cách chia dữ liệu và metric riêng, nên copy số liệu trong
bài báo rồi so với số của mình là SAI. Cách đúng: dựng lại Ý TƯỞNG của họ trong
cùng một khung thực nghiệm (cùng dataset, cùng split, cùng metric) như file này làm.
Chỗ nào phải sửa để chạy được thì đều ghi rõ trong docstring từng hàm.

Mọi hàm đều nhận `scores` — bảng đã có sẵn điểm của từng model thành phần:
    svdpp | svd | itemknn | lightgcn | content | demo | popular | n | item_n | rating
"""

import numpy as np

from utils.alpha_scheduler import compute_dynamic_alpha
from utils.tuning import grid_search, rmse


def _clip(x):
    return np.clip(x, 1.0, 5.0)


# ── 1. Tarek et al. (2023) ───────────────────────────────
# "A Hybrid Hotel Recommendation Using Collaborative, Content Based and Knowledge Based"
# Gốc: Item-Item CF + Content (NLP) + Knowledge fallback cho cold user. Trọng số CỐ ĐỊNH.
# Thích nghi: trọng số gốc không nêu rõ con số → dùng 50/50, và Knowledge fallback
#             hiện thực bằng popularity (hotel điểm cao) vì offline không có
#             dữ liệu phòng trống / ngân sách.

def tarek_2023(scores, w_cf=0.5):
    """Item-Item CF + Content, trọng số cố định; user trắng → fallback popularity."""
    pred = w_cf * scores["itemknn"] + (1 - w_cf) * scores["content"]
    pred = np.where(scores["n"] == 0, scores["popular"], pred)
    return _clip(pred)


# ── 2. Do & Le (2020) ────────────────────────────────────
# "Dynamic Weighted Hybrid Recommender Systems"
# Gốc (theo ghi chú bài báo): CF dùng **SVD** (không phải SVD++) + CBF (cosine),
#      ĐÚNG 2 THÀNH PHẦN. Trọng số không cố định mà điều chỉnh theo hiệu năng:
#      model nào đoán đúng hơn trên validation thì được trọng số cao hơn.
# Thích nghi: bài gốc không nêu công thức trọng số cụ thể → dùng ∝ 1/RMSE².
#      Bài gốc đánh giá bằng Personalization/Diversity, không phải RMSE — phải ghi
#      rõ trong báo cáo là ta so bằng metric khác với metric họ tối ưu.
# Chạy 2 biến thể:
#   - "2 thành phần": trung thành với bài gốc (SVD + Content)
#   - "3 thành phần": mở rộng thêm Demographic, để tách bạch xem lợi thế của StayReco
#     đến từ Ý TƯỞNG α động hay chỉ từ việc có thêm thành phần

def fit_do_le_2020(val_scores, components=("svd", "content")):
    """Học trọng số từ hiệu năng trên validation. Trả về dict trọng số."""
    errs = {c: rmse(val_scores[c], val_scores["rating"]) for c in components}
    raw = {c: 1.0 / (e ** 2) for c, e in errs.items()}
    total = sum(raw.values())
    weights = {c: w / total for c, w in raw.items()}
    print("    RMSE từng thành phần trên validation: "
          + ", ".join(f"{c}={e:.4f}" for c, e in errs.items()))
    print("    → trọng số: " + ", ".join(f"{c}={w:.3f}" for c, w in weights.items()))
    return weights


def do_le_2020(scores, weights):
    pred = sum(w * scores[c] for c, w in weights.items())
    return _clip(pred)


# ── 3. OHWSF (2022) ──────────────────────────────────────
# "Optimization Hybrid Weighted Switching Filtering"
# Gốc: HWF = σ·SVD + (1−σ)·SVD++ ; HSF = chuyển đổi SVD/SVD++ theo ngưỡng θ ;
#      OHWSF = σ₁·HWF + (1−σ₁)·HSF, với σ₁ tối ưu bằng grid search.
# Thích nghi: luật switching gốc so sai số với rating THẬT (|SVD − true| < θ) —
#             không dùng được lúc dự đoán vì chưa biết true. Thay bằng mức BẤT ĐỒNG
#             giữa 2 model: |SVD − SVD++| < θ → hai model đồng ý → dùng SVD,
#             ngược lại → dùng SVD++ (model mạnh hơn cho ca khó).

def ohwsf_2022(scores, sigma=0.5, theta=0.2, sigma1=0.5):
    hwf = sigma * scores["svd"] + (1 - sigma) * scores["svdpp"]
    agree = np.abs(scores["svd"] - scores["svdpp"]) < theta
    hsf = np.where(agree, scores["svd"], scores["svdpp"])
    return _clip(sigma1 * hwf + (1 - sigma1) * hsf)


OHWSF_GRID = {
    "sigma": [0.0, 0.25, 0.5, 0.75, 1.0],
    "theta": [0.05, 0.1, 0.2, 0.3, 0.5],
    "sigma1": [0.0, 0.25, 0.5, 0.75, 1.0],
}


# ── 4. SL-ARec (2026) ────────────────────────────────────
# "Self-Learning Adaptive Recommendation" (PeerJ CS)
# Gốc: tầng 1 SVD++ + tầng 2 LightGCN trộn bằng α TỰ HỌC; tầng 3 safety layer
#      (Demographic + TF-IDF content) BẬT/TẮT theo cold-start — switch nhị phân.
# Thích nghi: α "tự học" ở đây học bằng grid search trên validation (bài gốc học
#             bằng neural network); LightGCN huấn luyện bằng MSE thay vì BPR để
#             cùng thang rating (xem models/lightgcn.py).

def slarec_2026(scores, w_svdpp=0.5, n_safe=5, w_content=0.5):
    main = w_svdpp * scores["svdpp"] + (1 - w_svdpp) * scores["lightgcn"]
    safety = w_content * scores["content"] + (1 - w_content) * scores["demo"]
    # Switch NHỊ PHÂN: cold thì bật hẳn safety layer, warm thì tắt hẳn
    cold = (scores["n"] < n_safe) | (scores["item_n"] == 0)
    return _clip(np.where(cold, safety, main))


SLAREC_GRID = {
    "w_svdpp": [0.0, 0.25, 0.5, 0.75, 1.0],
    "n_safe": [1, 3, 5, 10, 20],
    "w_content": [0.0, 0.25, 0.5, 0.75, 1.0],
}


# ── 5. StayReco với tham số tune trên validation ─────────
# Giữ nguyên DẠNG công thức α(n) (đóng góp của đề tài), chỉ chọn các con số
# bằng grid search như OHWSF làm — để không còn bị hỏi "vì sao là 0.7?".

def stayreco(scores, alpha_max=0.7, n_threshold=30, gamma_base=0.15, alpha_min=0.0):
    cache = {}

    def weights_for(n, item_n):
        key = (n, item_n == 0)
        if key not in cache:
            cache[key] = compute_dynamic_alpha(
                n, alpha_max=alpha_max, n_threshold=n_threshold,
                gamma_base=gamma_base, alpha_min=alpha_min,
                item_n=item_n,
            )
        return cache[key]

    w = np.array([[v["alpha"], v["beta"], v["gamma"]]
                  for v in (weights_for(n, m) for n, m in zip(scores["n"], scores["item_n"]))])
    pred = (w[:, 0] * scores["svdpp"].to_numpy()
            + w[:, 1] * scores["content"].to_numpy()
            + w[:, 2] * scores["demo"].to_numpy())
    return _clip(pred)


STAYRECO_GRID = {
    "alpha_max": [0.5, 0.6, 0.7, 0.8, 0.9],
    "n_threshold": [10, 20, 30, 50],
    "gamma_base": [0.10, 0.15, 0.20],
    "alpha_min": [0.0, 0.1, 0.2, 0.3],
}


# ── Gom lại: tune trên validation rồi dự đoán trên test ──

def build_paper_methods(val_scores):
    """Tune từng phương pháp trên VALIDATION, trả về dict {tên: hàm dự đoán}.

    Returns:
        methods: {tên: function(scores_df) → mảng dự đoán}
        chosen: {tên: tham số đã chọn} — để ghi vào báo cáo
    """
    methods, chosen = {}, {}

    print("\n  → Tarek et al. (2023): Item-Item CF + Content + Knowledge fallback (trọng số cố định)")
    methods["Tarek 2023 (ItemCF+CB+KB)"] = lambda s: tarek_2023(s)
    chosen["Tarek 2023 (ItemCF+CB+KB)"] = {"w_cf": 0.5, "ghi_chu": "trọng số cố định theo bài gốc"}

    print("\n  → Do & Le (2020): trọng số theo hiệu năng trên validation")
    for label, comps in [("Do & Le 2020 (SVD+CB)", ("svd", "content")),
                         ("Do & Le 2020 (mở rộng 3 TP)", ("svdpp", "content", "demo"))]:
        print(f"    [{label}]")
        w = fit_do_le_2020(val_scores, components=comps)
        methods[label] = lambda s, w=w: do_le_2020(s, w)
        chosen[label] = w

    print("\n  → OHWSF (2022): grid search σ, θ, σ₁")
    best, _ = grid_search(OHWSF_GRID, ohwsf_2022, val_scores)
    methods["OHWSF 2022 (SVD+SVD++)"] = lambda s, p=best: ohwsf_2022(s, **p)
    chosen["OHWSF 2022 (SVD+SVD++)"] = best

    print("\n  → SL-ARec (2026): SVD++ + LightGCN + safety layer (switch nhị phân)")
    best, _ = grid_search(SLAREC_GRID, slarec_2026, val_scores)
    methods["SL-ARec 2026 (SVD+++LightGCN)"] = lambda s, p=best: slarec_2026(s, **p)
    chosen["SL-ARec 2026 (SVD+++LightGCN)"] = best

    print("\n  → StayReco: grid search α_max, n_threshold, γ_base, α_min")
    best, history = grid_search(STAYRECO_GRID, stayreco, val_scores)
    methods["★ StayReco (α tuned)"] = lambda s, p=best: stayreco(s, **p)
    chosen["★ StayReco (α tuned)"] = best

    return methods, chosen, history
