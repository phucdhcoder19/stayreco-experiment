"""So sánh THEO CẶP giữa StayReco và từng đối thủ trên cùng seed.

Vì sao cần: cùng một seed thì mọi phương pháp dùng CHUNG một cách chia dữ liệu,
nên so chênh lệch trên từng seed chính xác hơn nhiều so với chỉ nhìn trung bình
± độ lệch chuẩn (độ lệch đó phần lớn đến từ việc chia dữ liệu khác nhau, không
phải từ chất lượng phương pháp).
"""

import pandas as pd

REF = "★ StayReco (α tuned)"


def main():
    df = pd.read_csv("results/comparison.csv")
    pivot = df.pivot(index="seed", columns="method", values="rmse")

    print("RMSE từng seed (thấp = tốt):")
    print(pivot.round(4).T.to_string())

    print(f"\nChênh lệch theo cặp: (đối thủ − {REF}); dương = StayReco tốt hơn")
    rows = []
    for method in pivot.columns:
        if method == REF:
            continue
        diff = pivot[method] - pivot[REF]
        rows.append({
            "đối thủ": method,
            "chênh lệch TB": round(diff.mean(), 4),
            "độ lệch chuẩn": round(diff.std(), 4),
            "StayReco thắng": f"{int((diff > 0).sum())}/{len(diff)} seed",
        })
    print(pd.DataFrame(rows).sort_values("chênh lệch TB").to_string(index=False))


if __name__ == "__main__":
    main()
