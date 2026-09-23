"""Collaborative Filtering bằng SVD++ (Surprise library).

Cơ sở: Koren (2008), "Factorization Meets the Neighborhood"
"""

from surprise import SVDpp, SVD
from surprise import accuracy


class CollaborativeFilter:
    """SVD++ wrapper cho StayReco."""

    def __init__(self, n_factors=50, n_epochs=30, lr=0.005, reg=0.02, seed=None):
        self.model = SVDpp(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr,
            reg_all=reg,
            random_state=seed,
            verbose=True,
        )
        self.is_trained = False

    def fit(self, trainset):
        """Train SVD++ trên trainset (Surprise format)."""
        print("Training SVD++...")
        self.model.fit(trainset)
        self.is_trained = True
        print("✓ SVD++ trained")

    def predict(self, user_id, item_id):
        """Dự đoán rating cho 1 cặp (user, item).

        Returns:
            float: predicted rating (1-5)
        """
        if not self.is_trained:
            raise RuntimeError("Model chưa train. Gọi fit() trước.")
        # User/hotel chưa có trong train → Surprise tự dùng μ + bias còn lại
        pred = self.model.predict(str(user_id), str(item_id))
        return pred.est

    def predict_batch(self, testset):
        """Dự đoán cho toàn bộ testset.

        Args:
            testset: list of (user_id, item_id, true_rating)

        Returns:
            list of Prediction objects
        """
        predictions = self.model.test(testset)
        return predictions

    def evaluate(self, testset):
        """Tính RMSE và MAE trên testset."""
        predictions = self.predict_batch(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        return {"rmse": rmse, "mae": mae, "predictions": predictions}


class PlainSVD(CollaborativeFilter):
    """SVD thường (KHÔNG có implicit feedback).

    Dùng để dựng lại OHWSF (2022) — bài đó trộn SVD với SVD++, nên cần cả hai.
    Cũng là bằng chứng cho lựa chọn "SVD++ thay vì SVD" của StayReco.
    """

    def __init__(self, n_factors=50, n_epochs=30, lr=0.005, reg=0.02, seed=None):
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr,
            reg_all=reg,
            random_state=seed,
        )
        self.is_trained = False

    def fit(self, trainset):
        print("Training SVD...")
        self.model.fit(trainset)
        self.is_trained = True
        print("✓ SVD trained")
