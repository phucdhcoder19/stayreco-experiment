"""α Dynamic Scheduler — đóng góp chính của StayReco.

Cơ sở: Kết hợp weighted + switching hybrid (Burke, 2002)
α trượt liên tục theo n thay vì switch cứng.

Công thức:
    score = α(n)·CF + β(n)·Content + γ(n)·Demographic

    n = số tương tác của user
    n=0, không có sở thích  → γ cao, dựa Demographic + Content
    n=0, có sở thích        → β=1, Content 100%
    n tăng                  → α tăng tuyến tính tới α_max tại n_threshold
    hotel mới (item_n=0)    → β=1, chỉ Content biết gì về hotel này
"""


def compute_dynamic_alpha(n, alpha_max=0.7, n_threshold=30, gamma_base=0.15,
                          has_preferences=False, item_n=None, alpha_min=0.0):
    """Tính trọng số (α, β, γ) theo số tương tác n.

    Args:
        n: số tương tác của user
        alpha_max: α tối đa khi n lớn (default 0.7)
        n_threshold: n tại đó α đạt alpha_max (default 30)
        gamma_base: trọng số cơ bản cho Demographic
        has_preferences: user đã khai sở thích chưa
        item_n: số tương tác của hotel (None = không xét phía hotel)
        alpha_min: α tối thiểu, kể cả khi n=0. Với user mới, CF vẫn còn biết
            "hotel nào nói chung được chấm cao" (qua b_i), nên bỏ hẳn CF (α=0)
            là lãng phí. Giá trị này được chọn bằng grid search trên validation.

    Returns:
        dict: {"alpha": float, "beta": float, "gamma": float}
              sao cho alpha + beta + gamma = 1.0
    """
    if item_n == 0:
        # Hotel mới → CF và Demographic chưa có rating nào của hotel → chỉ tin Content
        return {"alpha": 0.0, "beta": 1.0, "gamma": 0.0}

    if n == 0 and not has_preferences:
        # User trắng hoàn toàn → Demographic + Content (chia phần còn lại 40/60)
        rest = 1.0 - alpha_min
        return {"alpha": alpha_min, "beta": 0.4 * rest, "gamma": 0.6 * rest}

    if n == 0 and has_preferences:
        # Có sở thích nhưng chưa tương tác → Content
        return {"alpha": alpha_min, "beta": 1.0 - alpha_min, "gamma": 0.0}

    # n > 0: α tăng dần theo n
    alpha = max(alpha_min, min(alpha_max, (n / n_threshold) * alpha_max))

    # γ giảm dần khi n tăng
    gamma = max(0.0, gamma_base * (1 - n / n_threshold))

    # β = phần còn lại
    beta = max(0.0, 1.0 - alpha - gamma)

    # Normalize để tổng = 1
    total = alpha + beta + gamma
    return {
        "alpha": alpha / total,
        "beta": beta / total,
        "gamma": gamma / total,
    }


def compute_fixed_alpha(alpha_value):
    """Trả về trọng số cố định (baseline để so sánh).

    Args:
        alpha_value: giá trị α cố định (0.0 → 1.0)

    Returns:
        dict: {"alpha": float, "beta": float, "gamma": float}
    """
    beta = (1.0 - alpha_value) * 0.7   # 70% phần còn lại cho Content
    gamma = (1.0 - alpha_value) * 0.3   # 30% cho Demographic
    return {"alpha": alpha_value, "beta": beta, "gamma": gamma}


# ── Test nhanh ───────────────────────────────────────────
if __name__ == "__main__":
    print("=== α Dynamic Schedule ===")
    print(f"{'n':>5} | {'α':>6} | {'β':>6} | {'γ':>6}")
    print("-" * 35)
    print(f"{'0*':>5} | " + " | ".join(f"{v:>6.3f}" for v in compute_dynamic_alpha(0, has_preferences=True).values())
          + "   (* đã khai sở thích)")
    for n in [0, 1, 3, 5, 10, 15, 20, 30, 50]:
        w = compute_dynamic_alpha(n)
        print(f"{n:>5} | {w['alpha']:>6.3f} | {w['beta']:>6.3f} | {w['gamma']:>6.3f}")
