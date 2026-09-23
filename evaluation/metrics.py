"""Evaluation metrics cho recommender system.

Cơ sở: Herlocker et al. (2004), "Evaluating Collaborative Filtering
Recommender Systems"

Hai nhóm metrics:
1. Rating prediction: RMSE, MAE
2. Ranking quality: Precision@K, NDCG@K
"""

import numpy as np
from collections import defaultdict


# ── Rating Prediction Metrics ────────────────────────────

def rmse(results):
    """Root Mean Squared Error.

    Args:
        results: list of (user, item, true_rating, pred_rating)
    """
    errors = [(true - pred) ** 2 for _, _, true, pred in results]
    return np.sqrt(np.mean(errors))


def mae(results):
    """Mean Absolute Error.

    Args:
        results: list of (user, item, true_rating, pred_rating)
    """
    errors = [abs(true - pred) for _, _, true, pred in results]
    return np.mean(errors)


# ── Ranking Metrics ──────────────────────────────────────

def precision_at_k(results, k=10, threshold=4.0):
    """Precision@K: trong K gợi ý, bao nhiêu cái user thật sự thích?

    Args:
        results: list of (user, item, true_rating, pred_rating)
        k: số items gợi ý
        threshold: rating >= threshold coi là "thích"

    Returns:
        float: trung bình Precision@K trên tất cả users
    """
    user_preds = defaultdict(list)
    for user, item, true, pred in results:
        user_preds[user].append((pred, true))

    precisions = []
    for user, preds in user_preds.items():
        # Sắp xếp theo predicted rating giảm dần
        preds.sort(key=lambda x: x[0], reverse=True)
        top_k = preds[:k]

        # Đếm relevant items trong top K
        relevant = sum(1 for _, true in top_k if true >= threshold)
        precisions.append(relevant / k)

    return np.mean(precisions) if precisions else 0.0


def ndcg_at_k(results, k=10):
    """NDCG@K: đo chất lượng xếp hạng, có tính thứ tự.

    Items đúng đứng đầu → điểm cao hơn items đúng đứng cuối.

    Args:
        results: list of (user, item, true_rating, pred_rating)
        k: số items

    Returns:
        float: trung bình NDCG@K
    """
    user_preds = defaultdict(list)
    for user, item, true, pred in results:
        user_preds[user].append((pred, true))

    ndcgs = []
    for user, preds in user_preds.items():
        if len(preds) < 2:
            continue

        # DCG: sắp theo predicted rating
        preds.sort(key=lambda x: x[0], reverse=True)
        top_k = preds[:k]
        dcg = sum(true / np.log2(i + 2) for i, (_, true) in enumerate(top_k))

        # IDCG: sắp theo true rating (perfect ranking)
        ideal = sorted(preds, key=lambda x: x[1], reverse=True)[:k]
        idcg = sum(true / np.log2(i + 2) for i, (_, true) in enumerate(ideal))

        if idcg > 0:
            ndcgs.append(dcg / idcg)

    return np.mean(ndcgs) if ndcgs else 0.0


def coverage_at_k(results, k=10):
    """Coverage@K: bao nhiêu % hotel từng lọt vào top-K của ít nhất 1 user?

    Cơ sở: Herlocker et al. (2004) — chỉ đo độ chính xác là chưa đủ. Một hệ chỉ
    gợi ý mãi vài hotel nổi tiếng vẫn có RMSE đẹp nhưng vô dụng với nền tảng mới,
    vì hotel mới không bao giờ được hiển thị.

    Args:
        results: list of (user, item, true_rating, pred_rating)
        k: số item mỗi user được gợi ý
    """
    user_preds = defaultdict(list)
    all_items = set()
    for user, item, true, pred in results:
        user_preds[user].append((pred, item))
        all_items.add(item)

    recommended = set()
    for preds in user_preds.values():
        preds.sort(key=lambda x: x[0], reverse=True)
        recommended.update(item for _, item in preds[:k])

    return len(recommended) / len(all_items) if all_items else 0.0


def evaluate_all(results, top_k_values=[5, 10, 20]):
    """Chạy tất cả metrics.

    Returns:
        dict: {"rmse": ..., "mae": ..., "p@5": ..., "ndcg@10": ...}
    """
    metrics = {
        "rmse": rmse(results),
        "mae": mae(results),
    }

    for k in top_k_values:
        metrics[f"p@{k}"] = precision_at_k(results, k=k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(results, k=k)
        metrics[f"cov@{k}"] = coverage_at_k(results, k=k)

    return metrics


def evaluate_by_group(results, groups, top_k_values=[5, 10, 20]):
    """Chạy tất cả metrics riêng cho từng nhóm (warm / cool / cold / cold_item).

    Args:
        results: list of (user, item, true_rating, pred_rating)
        groups: list tên nhóm, cùng thứ tự với results

    Returns:
        dict: {group: {"n_rows": ..., "rmse": ..., ...}}
    """
    by_group = defaultdict(list)
    for row, group in zip(results, groups):
        by_group[group].append(row)

    return {
        group: {"n_rows": len(rows), **evaluate_all(rows, top_k_values)}
        for group, rows in by_group.items()
    }


def print_metrics(name, metrics):
    """In kết quả đẹp."""
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  RMSE:      {metrics['rmse']:.4f}")
    print(f"  MAE:       {metrics['mae']:.4f}")
    for key in sorted(metrics.keys()):
        if key.startswith("p@"):
            print(f"  {key.upper()}:  {metrics[key]:.4f}")
    for key in sorted(metrics.keys()):
        if key.startswith("ndcg@"):
            print(f"  {key.upper()}: {metrics[key]:.4f}")
    print()
