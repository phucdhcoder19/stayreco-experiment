"""Cấu hình cho thí nghiệm StayReco."""

# ── Dataset ──────────────────────────────────────────────
DATASET = "tripadvisor"  # hoặc "movielens-100k" để test nhanh
TEST_RATIO = 0.2
VAL_RATIO = 0.5          # nửa phần đã giấu dùng để CHỌN THAM SỐ, nửa còn lại để chấm điểm
RANDOM_SEED = 42

# ── SVD++ Hyperparameters ────────────────────────────────
SVDPP_FACTORS = 50       # số chiều ẩn (k)
SVDPP_EPOCHS = 30        # số vòng lặp
SVDPP_LR = 0.005         # learning rate
SVDPP_REG = 0.02         # regularization λ

# ── Cold-start simulation ────────────────────────────────
COLD_USER_RATIO = 0.3    # % user bị cắt lịch sử để giả lập user mới
MAX_COLD_N = 19          # user giả lập giữ ngẫu nhiên 0 → 19 rating
COLD_ITEM_RATIO = 0.05   # % hotel bị giấu toàn bộ rating (hotel mới)
WARM_MIN_N = 20          # warm: n ≥ 20
COLD_MAX_N = 5           # cold: n < 5 ; cool: 5 ≤ n < 20

# ── Content-Based ────────────────────────────────────────
EMBEDDING_DIM = 100      # chiều embedding cho hotel description
CONTENT_K = 30           # số hotel giống nhất dùng để dự đoán

# ── Demographic ──────────────────────────────────────────
# Hotel: ("age_group", "country", "trip_type") ; MovieLens không có → dùng tuổi, giới tính, nghề
DEMO_COLS = ("age_group", "gender", "occupation")
DEMO_SHRINKAGE = 5       # nhóm càng ít rating càng bị kéo về trung bình hotel

# ── α Dynamic Schedule ──────────────────────────────────
# α(n) = min(α_max, n / n_threshold)
# n < 5   → α ≈ 0, β cao   (Content chiếm ưu thế)
# n ≈ 10  → α ≈ 0.5        (blend)
# n ≥ 30  → α = 0.7+       (CF chiếm ưu thế)
ALPHA_MAX = 0.7
N_THRESHOLD = 30         # n tại đó α đạt α_max
GAMMA_BASE = 0.15        # trọng số demographic cơ bản

# ── LightGCN (tầng 2 của SL-ARec 2026) ──────────────────
GCN_FACTORS = 32         # chiều embedding
GCN_LAYERS = 3           # số lớp lan truyền trên đồ thị
GCN_EPOCHS = 300
GCN_LR = 0.01

# ── Baselines để so sánh ─────────────────────────────────
FIXED_ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]

# ── Evaluation ───────────────────────────────────────────
TOP_K = [5, 10, 20]      # Precision@K, NDCG@K
