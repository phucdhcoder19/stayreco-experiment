"""
StayReco Experiment Runner
==========================

So sánh StayReco với 2 loại baseline:

  A. Baseline trọng số (cùng 3 model thành phần, chỉ khác cách trộn):
     CF only · Content only · Demographic only · Hybrid α=0.3/0.5/0.7 cố định · StayReco α động

  B. Phương pháp của 4 bài báo (dựng lại trong CÙNG khung thực nghiệm):
     Tarek 2023 · Do & Le 2020 · OHWSF 2022 · SL-ARec 2026 · StayReco (α tune trên validation)

Dữ liệu chia 3 phần: train (học model) → validation (chọn tham số) → test (chấm điểm).

Cách chạy:
    conda activate stayreco
    python run_experiment.py            # 1 seed
    python run_experiment.py 42 1 2     # nhiều seed, lấy trung bình ± độ lệch chuẩn

Output trong results/:
    comparison.csv           — metrics tổng thể trên tập test
    comparison_by_group.csv  — metrics theo nhóm cold-start
    chosen_params.csv        — tham số grid search đã chọn (để ghi vào báo cáo)
    comparison.png           — biểu đồ theo nhóm
    sensitivity_alpha.png    — RMSE thay đổi thế nào theo α_max (trả lời "vì sao 0.7?")
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import *
from data.loader import load_movielens_full
from data.preprocessor import (simulate_cold_start, split_validation, assign_groups,
                               to_surprise_trainset)
from models.cf_svdpp import CollaborativeFilter, PlainSVD
from models.cf_itemknn import ItemItemCF
from models.content_based import ContentBasedFilter
from models.demographic import DemographicFilter
from models.popularity import PopularityBaseline
from models.lightgcn import LightGCN
from models import paper_baselines as papers
from utils.alpha_scheduler import compute_fixed_alpha
from evaluation.metrics import evaluate_all, evaluate_by_group

GROUP_ORDER = ["cold_item", "cold", "cool", "warm"]
GROUP_LABELS = {
    "cold_item": "Cold item (hotel mới)",
    "cold": f"Cold user (n<{COLD_MAX_N})",
    "cool": f"Cool user ({COLD_MAX_N}≤n<{WARM_MIN_N})",
    "warm": f"Warm user (n≥{WARM_MIN_N})",
}
STAYRECO_TUNED = "★ StayReco (α tuned)"
STAYRECO_MANUAL = "★ StayReco (α thủ công)"


def weight_methods():
    """Nhóm A: 7 cách trộn cùng 3 model thành phần."""
    def blend(a, b, g):
        return lambda s: np.clip(a * s["svdpp"] + b * s["content"] + g * s["demo"], 1, 5)

    methods = {
        "CF Only (SVD++)": blend(1, 0, 0),
        "Content Only": blend(0, 1, 0),
        "Demographic Only": blend(0, 0, 1),
    }
    for a in FIXED_ALPHAS:
        if 0.0 < a < 1.0:
            w = compute_fixed_alpha(a)
            methods[f"Hybrid α={a} (fixed)"] = blend(w["alpha"], w["beta"], w["gamma"])
    methods[STAYRECO_MANUAL] = lambda s: papers.stayreco(
        s, alpha_max=ALPHA_MAX, n_threshold=N_THRESHOLD, gamma_base=GAMMA_BASE, alpha_min=0.0)
    return methods


def run_one_seed(seed):
    """Chạy trọn 1 lần thí nghiệm với 1 seed. Trả về (overall_df, by_group_df, chosen, history)."""
    print("\n" + "=" * 70)
    print(f"  SEED {seed}")
    print("=" * 70)

    # ── 1. Dữ liệu ───────────────────────────────────────
    print("\n[1/4] Loading dataset...")
    ratings_df, items_df, users_df = load_movielens_full()

    print("\n[2/4] Splitting data...")
    df_train, df_held = simulate_cold_start(
        ratings_df, test_ratio=TEST_RATIO, cold_user_ratio=COLD_USER_RATIO,
        max_cold_n=MAX_COLD_N, cold_item_ratio=COLD_ITEM_RATIO, seed=seed)
    df_held = assign_groups(df_held, df_train, warm_min=WARM_MIN_N, cold_max=COLD_MAX_N)
    df_held = split_validation(df_held, val_ratio=VAL_RATIO, seed=seed)
    print("  Rows theo nhóm:", df_held["group"].value_counts().reindex(GROUP_ORDER).to_dict())

    # ── 2. Train các model thành phần ────────────────────
    print("\n[3/4] Training models...")
    trainset = to_surprise_trainset(df_train)
    components = {}

    svdpp = CollaborativeFilter(SVDPP_FACTORS, SVDPP_EPOCHS, SVDPP_LR, SVDPP_REG, seed=seed)
    svdpp.fit(trainset)
    components["svdpp"] = svdpp

    svd = PlainSVD(SVDPP_FACTORS, SVDPP_EPOCHS, SVDPP_LR, SVDPP_REG, seed=seed)
    svd.fit(trainset)
    components["svd"] = svd

    itemknn = ItemItemCF(k_neighbors=CONTENT_K)
    itemknn.fit(df_train)
    components["itemknn"] = itemknn

    lightgcn = LightGCN(n_factors=GCN_FACTORS, n_layers=GCN_LAYERS,
                        n_epochs=GCN_EPOCHS, lr=GCN_LR, seed=seed)
    lightgcn.fit(df_train)
    components["lightgcn"] = lightgcn

    content = ContentBasedFilter(k_neighbors=CONTENT_K)
    content.fit(items_df, df_train)
    components["content"] = content

    demo = DemographicFilter(group_cols=DEMO_COLS, shrinkage=DEMO_SHRINKAGE)
    demo.fit(df_train, users_df)
    components["demo"] = demo

    popular = PopularityBaseline(shrinkage=DEMO_SHRINKAGE)
    popular.fit(df_train)
    components["popular"] = popular

    # ── 3. Hỏi mọi model 1 lần cho mọi dòng val+test ─────
    print("\n[4/4] Scoring & evaluating...")
    scores = df_held.copy()
    for name, model in components.items():
        scores[name] = [model.predict(u, i)
                        for u, i in zip(scores["user_id"], scores["item_id"])]

    val_scores = scores[scores["split"] == "val"]
    test_scores = scores[scores["split"] == "test"]

    # Lưu lại bảng điểm để xem bằng mắt: mỗi dòng là 1 câu hỏi, mỗi cột là ý kiến 1 model
    os.makedirs("results", exist_ok=True)
    cols = ["user_id", "item_id", "rating", "n", "item_n", "group",
            "svdpp", "svd", "itemknn", "lightgcn", "content", "demo", "popular"]
    test_scores[cols].round(3).to_csv(f"results/score_table_seed{seed}.csv", index=False)
    # Bản rút gọn: 50 dòng mỗi nhóm, để mở bằng Excel xem cho dễ
    (test_scores[cols].groupby("group", group_keys=False)
     .apply(lambda g: g.head(50)).round(3)
     .to_csv(f"results/score_table_sample_seed{seed}.csv", index=False))

    # ── 4. Tune các phương pháp trên validation ──────────
    methods = weight_methods()
    paper_methods, chosen, history = papers.build_paper_methods(val_scores)
    methods.update(paper_methods)

    # ── 5. Chấm điểm trên test ───────────────────────────
    overall, by_group_rows = {}, []
    for name, predict_fn in methods.items():
        preds = predict_fn(test_scores)
        results = list(zip(test_scores["user_id"], test_scores["item_id"],
                           test_scores["rating"], preds))
        overall[name] = evaluate_all(results, TOP_K)
        for group, metrics in evaluate_by_group(results, test_scores["group"], TOP_K).items():
            by_group_rows.append({"method": name, "group": group, **metrics})

    overall_df = pd.DataFrame(overall).T
    overall_df.index.name = "method"
    return overall_df, pd.DataFrame(by_group_rows), chosen, history


def run(seeds):
    all_overall, all_by_group = [], []
    chosen, history = None, None

    for seed in seeds:
        overall_df, by_group_df, chosen, history = run_one_seed(seed)
        all_overall.append(overall_df.assign(seed=seed))
        all_by_group.append(by_group_df.assign(seed=seed))

    overall = pd.concat(all_overall)
    by_group = pd.concat(all_by_group)
    save_results(overall, by_group, chosen, history, seeds)


def save_results(overall, by_group, chosen, history, seeds):
    os.makedirs("results", exist_ok=True)
    overall.to_csv("results/comparison.csv")
    by_group.to_csv("results/comparison_by_group.csv", index=False)
    pd.DataFrame(chosen).T.to_csv("results/chosen_params.csv")

    # Trung bình qua các seed
    order = list(dict.fromkeys(overall.index))
    mean = overall.groupby("method").mean(numeric_only=True).reindex(order)
    std = overall.groupby("method").std(numeric_only=True).reindex(order)

    print("\n" + "=" * 78)
    print(f"  KẾT QUẢ TRÊN TEST ({len(seeds)} seed: {seeds})")
    print("=" * 78)
    summary = pd.DataFrame({
        "RMSE": [f"{m:.4f} ± {s:.4f}" if not np.isnan(s) else f"{m:.4f}"
                 for m, s in zip(mean["rmse"], std["rmse"])],
        "MAE": [f"{m:.4f}" for m in mean["mae"]],
        "P@10": [f"{m:.4f}" for m in mean["p@10"]],
        "NDCG@10": [f"{m:.4f}" for m in mean["ndcg@10"]],
    }, index=mean.index)
    print(summary.to_string())

    group_mean = by_group.groupby(["method", "group"])["rmse"].mean().unstack()
    group_mean = group_mean.reindex(index=order, columns=GROUP_ORDER)
    print("\n  RMSE THEO NHÓM")
    print(group_mean.round(4).to_string())

    print("\n  THAM SỐ ĐÃ CHỌN TRÊN VALIDATION (seed cuối)")
    for name, params in chosen.items():
        print(f"    {name}: {params}")

    print("\n✓ Saved results/comparison.csv, comparison_by_group.csv, chosen_params.csv")
    for plot_fn, path in [(plot_by_group, "comparison.png"), (plot_sensitivity, "sensitivity_alpha.png")]:
        try:
            plot_fn(by_group if plot_fn is plot_by_group else history, order)
            print(f"✓ Chart saved to results/{path}")
        except Exception as e:
            print(f"⚠ Could not create results/{path}: {e}")


def _style(ax, ink_muted="#52514e"):
    ax.tick_params(axis="x", labelsize=8, colors=ink_muted)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")


def plot_by_group(by_group_df, method_order):
    """Small multiples: mỗi cột 1 nhóm, hàng trên RMSE, hàng dưới NDCG@10."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    highlight, muted, ink = "#2a78d6", "#b4b2a9", "#0b0b0b"
    metrics = [("rmse", "RMSE (thấp = tốt)"), ("ndcg@10", "NDCG@10 (cao = tốt)")]
    agg = by_group_df.groupby(["method", "group"]).mean(numeric_only=True)

    fig, axes = plt.subplots(len(metrics), len(GROUP_ORDER),
                             figsize=(19, 0.45 * len(method_order) * len(metrics) + 3), sharey=True)
    y = list(range(len(method_order)))[::-1]
    for row, (metric, metric_label) in enumerate(metrics):
        for col, group in enumerate(GROUP_ORDER):
            ax = axes[row][col]
            values = np.array([agg.loc[(m, group), metric] for m in method_order])
            colors = [highlight if m.startswith("★") else muted for m in method_order]
            ax.barh(y, values, color=colors, height=0.7, edgecolor="white", linewidth=2)

            best = values.min() if metric == "rmse" else values.max()
            for yi, v in zip(y, values):
                ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8,
                        color=ink, fontweight="bold" if v == best else "normal")

            lo, hi = values.min(), values.max()
            pad = (hi - lo) * 0.25 or 0.05
            ax.set_xlim(max(0, lo - pad), hi + pad * 1.8)
            if row == 0:
                n_rows = int(agg.loc[(method_order[0], group), "n_rows"])
                ax.set_title(f"{GROUP_LABELS[group]}\n{n_rows:,} test ratings", fontsize=10, color=ink)
            if col == 0:
                ax.set_yticks(y)
                ax.set_yticklabels(method_order, fontsize=8.5, color=ink)
            ax.set_xlabel(metric_label, fontsize=9, color="#52514e")
            _style(ax)

    fig.suptitle("StayReco vs baselines & 4 bài báo, theo nhóm cold-start (MovieLens 100K)",
                 fontsize=13, color=ink)
    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sensitivity(history, _order):
    """Biểu đồ độ nhạy: RMSE trên validation thay đổi thế nào theo α_max và n_threshold.

    Đây là câu trả lời bằng hình ảnh cho câu hỏi "vì sao chọn α_max = 0.7?".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame([{**p, "rmse": r} for p, r in history])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

    for ax, (x, series) in zip(axes, [("alpha_max", "n_threshold"), ("n_threshold", "alpha_max")]):
        best = df.loc[df.groupby([x, series])["rmse"].idxmin()]
        for color, (key, part) in zip(palette, best.groupby(series)):
            part = part.sort_values(x)
            ax.plot(part[x], part["rmse"], marker="o", markersize=6, linewidth=2,
                    color=color, label=f"{series}={key}")
        ax.set_xlabel(x, fontsize=9, color="#52514e")
        ax.set_ylabel("validation RMSE (thấp = tốt)", fontsize=9, color="#52514e")
        ax.legend(fontsize=8, frameon=False)
        _style(ax)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)

    fig.suptitle("Độ nhạy của StayReco theo tham số α (chọn trên validation)", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/sensitivity_alpha.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [RANDOM_SEED]
    run(seeds)
