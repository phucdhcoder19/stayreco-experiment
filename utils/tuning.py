"""Grid search trên tập VALIDATION.

Vì sao cần file này: tham số của StayReco (α_max, n_threshold...) hiện được chọn
thủ công — điểm yếu lớn nhất khi so với OHWSF (2022, grid search σ₁) và
Do & Le (2020, trọng số theo hiệu năng). File này cho phép chọn tham số TỪ DỮ LIỆU.

NGUYÊN TẮC: chỉ được tune trên VALIDATION, không bao giờ trên TEST.
Tune trên test = xem trước đáp án = kết quả báo cáo không còn giá trị.
"""

import itertools
import numpy as np


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def grid_search(param_grid, predict_fn, val_df, verbose=True):
    """Thử mọi tổ hợp tham số, chọn tổ hợp có RMSE thấp nhất trên validation.

    Args:
        param_grid: dict {tên tham số: [các giá trị thử]}
        predict_fn: function(scores_df, **params) → mảng dự đoán
        val_df: bảng điểm của tập validation (có cột 'rating' làm đáp án)

    Returns:
        best_params (dict), history (list of (params, rmse))
    """
    keys = list(param_grid)
    history = []
    for values in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, values))
        score = rmse(predict_fn(val_df, **params), val_df["rating"])
        history.append((params, score))

    best_params, best_rmse = min(history, key=lambda x: x[1])
    if verbose:
        print(f"    Grid search: {len(history)} tổ hợp → best val RMSE {best_rmse:.4f}")
        print(f"    Tham số tốt nhất: {best_params}")
    return best_params, history
