"""Popularity baseline — "hotel nào được chấm cao nói chung".

Dùng 2 chỗ:
1. Knowledge-based fallback của Tarek et al. (2023): user hoàn toàn mới thì gợi ý
   hotel phổ biến / điểm cao.
2. Mốc tham chiếu thấp nhất: mọi model phải tốt hơn cái này.

Có shrinkage: hotel chỉ 1 người chấm 5 sao thì chưa chắc tốt.
"""


class PopularityBaseline:
    def __init__(self, shrinkage=5):
        self.shrinkage = shrinkage
        self.item_means = {}
        self.global_mean = 3.0

    def fit(self, ratings_df):
        self.global_mean = ratings_df["rating"].mean()
        stats = ratings_df.groupby("item_id")["rating"].agg(["sum", "count"])
        means = (stats["sum"] + self.shrinkage * self.global_mean) / (stats["count"] + self.shrinkage)
        self.item_means = means.to_dict()
        print(f"✓ Popularity fitted: {len(self.item_means)} items")

    def predict(self, user_id, item_id):
        return self.item_means.get(item_id, self.global_mean)
