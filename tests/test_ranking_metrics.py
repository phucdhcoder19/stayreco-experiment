"""Kiểm tra công thức xếp hạng bằng ví dụ TÍNH TAY.

Vì sao cần: đây là các con số sẽ đi vào báo cáo. Một dấu trừ sai trong NDCG
không làm chương trình lỗi, nó chỉ làm kết luận sai. Mỗi test dưới đây có phần
tính tay viết ngay trong docstring để người đọc tự kiểm tra lại được.

    pytest tests/ -q
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.ranking import (build_candidate_frame, cold_item_share,
                                rank_metrics)

LOG2 = np.log2


def make_frame(rows):
    """rows: list of (user_id, item_id, relevant, item_n)"""
    frame = pd.DataFrame(rows, columns=["user_id", "item_id", "relevant", "item_n"])
    frame["n"] = 10
    return frame


def test_perfect_ranking_gives_one():
    """Xếp hoàn hảo → P, R, NDCG đều đạt mức tối đa.

    1 user, 4 ứng viên, 2 cái liên quan, model đẩy đúng 2 cái đó lên đầu.
        P@2    = 2/2 = 1.0
        R@2    = 2/2 = 1.0
        NDCG@2 = 1.0   (DCG = IDCG)
    """
    frame = make_frame([(1, 10, True, 5), (1, 11, True, 5),
                        (1, 12, False, 5), (1, 13, False, 5)])
    pred = np.array([5.0, 4.9, 1.0, 1.0])
    m = rank_metrics(frame, pred, pd.Series({1: 2}), top_k=(2,))
    assert m["p@2"] == 1.0
    assert m["r@2"] == 1.0
    assert abs(m["ndcg@2"] - 1.0) < 1e-12


def test_worst_ranking_gives_zero():
    """Xếp ngược hoàn toàn → tất cả về 0."""
    frame = make_frame([(1, 10, True, 5), (1, 11, True, 5),
                        (1, 12, False, 5), (1, 13, False, 5)])
    pred = np.array([1.0, 1.1, 5.0, 4.9])
    m = rank_metrics(frame, pred, pd.Series({1: 2}), top_k=(2,))
    assert m["p@2"] == 0.0
    assert m["r@2"] == 0.0
    assert m["ndcg@2"] == 0.0


def test_ndcg_rewards_putting_hits_first():
    """NDCG phải phạt việc đặt item đúng xuống dưới.

    1 user, 3 ứng viên, CHỈ 1 cái liên quan. So hai trường hợp:
      (a) item đúng ở hạng 1 → DCG = 1/log2(2) = 1.0
      (b) item đúng ở hạng 3 → DCG = 1/log2(4) = 0.5
      IDCG (1 item liên quan) = 1/log2(2) = 1.0
      → NDCG@3 = 1.0  và  0.5
    P@3 của cả hai đều = 1/3 → chứng tỏ NDCG phân biệt được, Precision thì không.
    """
    frame = make_frame([(1, 10, True, 5), (1, 11, False, 5), (1, 12, False, 5)])
    rel_first = rank_metrics(frame, np.array([9.0, 2.0, 1.0]),
                             pd.Series({1: 1}), top_k=(3,))
    rel_last = rank_metrics(frame, np.array([1.0, 9.0, 2.0]),
                            pd.Series({1: 1}), top_k=(3,))
    assert abs(rel_first["ndcg@3"] - 1.0) < 1e-12
    assert abs(rel_last["ndcg@3"] - 0.5) < 1e-12
    assert abs(rel_first["p@3"] - rel_last["p@3"]) < 1e-12


def test_precision_recall_differ_when_more_relevant_than_k():
    """Khi số item liên quan > K thì Precision và Recall phải khác nhau.

    1 user, 5 ứng viên, 4 cái liên quan, top-2 bắt được 2 cái:
        P@2 = 2/2 = 1.0     (2 ô hiển thị đều đúng)
        R@2 = 2/4 = 0.5     (nhưng chỉ bắt được nửa số item user thích)
        IDCG@2 = 1/log2(2) + 1/log2(3) ; DCG@2 giống hệt → NDCG@2 = 1.0
    """
    frame = make_frame([(1, 10, True, 5), (1, 11, True, 5), (1, 12, True, 5),
                        (1, 13, True, 5), (1, 14, False, 5)])
    pred = np.array([5.0, 4.9, 1.0, 1.0, 1.0])
    m = rank_metrics(frame, pred, pd.Series({1: 4}), top_k=(2,))
    assert m["p@2"] == 1.0
    assert m["r@2"] == 0.5
    assert abs(m["ndcg@2"] - 1.0) < 1e-12


def test_metrics_average_over_users_not_rows():
    """Trung bình phải tính theo USER, không theo dòng.

    User 1 xếp hoàn hảo (P@1 = 1), user 2 xếp sai (P@1 = 0), nhưng user 2 có
    nhiều ứng viên hơn. Nếu trung bình theo dòng thì user 2 sẽ lấn át.
        P@1 đúng = (1 + 0) / 2 = 0.5
    """
    frame = make_frame([(1, 10, True, 5), (1, 11, False, 5),
                        (2, 10, True, 5), (2, 11, False, 5),
                        (2, 12, False, 5), (2, 13, False, 5)])
    pred = np.array([9.0, 1.0,
                     1.0, 9.0, 8.0, 7.0])
    m = rank_metrics(frame, pred, pd.Series({1: 1, 2: 1}), top_k=(1,))
    assert abs(m["p@1"] - 0.5) < 1e-12


def test_coverage_counts_distinct_items_shown():
    """Coverage@K = % catalog từng lọt top-K của ít nhất 1 user.

    2 user, catalog 4 item. Top-1 của user 1 là item 10, của user 2 là item 11
    → 2 item khác nhau được hiển thị / 4 item catalog = 0.5
    """
    frame = make_frame([(1, 10, True, 5), (1, 11, False, 5),
                        (2, 11, True, 5), (2, 10, False, 5)])
    pred = np.array([9.0, 1.0, 9.0, 1.0])
    m = rank_metrics(frame, pred, pd.Series({1: 1, 2: 1}), top_k=(1,), n_catalog=4)
    assert abs(m["cov@1"] - 0.5) < 1e-12


def test_tie_break_is_deterministic():
    """Mọi điểm bằng nhau → kết quả vẫn phải cố định, không phụ thuộc thứ tự dòng.

    Popularity trả về cùng một điểm cho mọi hotel chưa ai rating. Nếu không
    tie-break theo item_id thì mỗi lần chạy ra một số khác → không tái lập được.
    """
    rows = [(1, 10, True, 5), (1, 11, False, 5), (1, 12, False, 5)]
    pred = np.array([3.0, 3.0, 3.0])
    a = rank_metrics(make_frame(rows), pred, pd.Series({1: 1}), top_k=(1,))
    b = rank_metrics(make_frame(rows[::-1]), pred, pd.Series({1: 1}), top_k=(1,))
    assert a["p@1"] == b["p@1"]
    assert a["ndcg@1"] == b["ndcg@1"]


def test_cold_item_share():
    """newitem@K = tỷ lệ ô hiển thị dành cho hotel chưa ai rating (item_n == 0).

    1 user, top-2 gồm item 12 (item_n=0, hotel mới) và item 10 (item_n=5)
    → newitem@2 = 1/2 = 0.5
    """
    frame = make_frame([(1, 10, True, 5), (1, 11, False, 5), (1, 12, False, 0)])
    pred = np.array([8.0, 1.0, 9.0])
    m = cold_item_share(frame, pred, top_k=(2,))
    assert abs(m["newitem@2"] - 0.5) < 1e-12


# ── Dựng rổ ứng viên ─────────────────────────────────────

def test_candidate_frame_excludes_train_items():
    """Item user đã rating trong TRAIN phải bị loại khỏi rổ.

    Gợi ý lại khách sạn khách vừa ở xong không phải là gợi ý.
    """
    df_train = pd.DataFrame({"user_id": [1, 1], "item_id": [10, 11],
                             "rating": [5.0, 3.0]})
    df_eval = pd.DataFrame({"user_id": [1], "item_id": [12], "rating": [5.0]})
    frame, n_rel = build_candidate_frame(df_train, df_eval, catalog_items=[10, 11, 12, 13])

    assert set(frame["item_id"]) == {12, 13}       # 10, 11 đã thấy trong train
    assert n_rel[1] == 1
    assert frame.set_index("item_id").loc[12, "relevant"]
    assert not frame.set_index("item_id").loc[13, "relevant"]


def test_candidate_frame_keeps_new_items():
    """Hotel chưa ai rating PHẢI có trong rổ — đây là điểm cốt lõi của đề tài.

    Item 99 không xuất hiện trong train của bất kỳ ai (item_n = 0) nhưng vẫn
    phải được xếp hạng, nếu không thì cold-start phía hotel không đo được.
    """
    df_train = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 10],
                             "rating": [5.0, 4.0]})
    df_eval = pd.DataFrame({"user_id": [1], "item_id": [11], "rating": [5.0]})
    frame, _ = build_candidate_frame(df_train, df_eval, catalog_items=[10, 11, 99])

    assert 99 in set(frame["item_id"])
    assert frame.set_index("item_id").loc[99, "item_n"] == 0


def test_candidate_frame_skips_users_without_relevant_items():
    """User không có item nào rating >= 4 thì không chấm được → bỏ qua.

    Giữ lại sẽ cho Recall = 0/0 và kéo trung bình xuống một cách vô nghĩa.
    """
    df_train = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 10],
                             "rating": [5.0, 4.0]})
    df_eval = pd.DataFrame({"user_id": [1, 2], "item_id": [11, 11],
                            "rating": [5.0, 2.0]})   # user 2 chấm 2 sao
    frame, n_rel = build_candidate_frame(df_train, df_eval, catalog_items=[10, 11, 12])

    assert set(frame["user_id"]) == {1}
    assert list(n_rel.index) == [1]
