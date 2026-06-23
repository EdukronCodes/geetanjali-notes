import numpy as np
import pandas as pd


class FeatureEngineer:
    """Engineer transaction features for anomaly detection."""

    FEATURE_COLUMNS = [
        "amount_zscore",
        "vendor_posting_count",
        "vendor_recurrence_ratio",
        "cost_center_deviation",
        "is_weekend",
        "duplicate_posting_flag",
        "amount_log",
        "posting_day_of_week",
    ]

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = df.copy()
        data["posting_date"] = pd.to_datetime(data["posting_date"])
        data["posting_day_of_week"] = data["posting_date"].dt.dayofweek
        data["is_weekend"] = data["posting_date"].dt.dayofweek >= 5

        vendor_stats = data.groupby("vendor_id")["amount"].agg(["mean", "std"]).rename(
            columns={"mean": "vendor_mean", "std": "vendor_std"}
        )
        data = data.merge(vendor_stats, on="vendor_id", how="left")
        data["vendor_std"] = data["vendor_std"].fillna(1.0).replace(0, 1.0)
        data["amount_zscore"] = (data["amount"] - data["vendor_mean"]) / data["vendor_std"]

        cc_mode = data.groupby("vendor_id")["cost_center"].agg(lambda x: x.mode().iloc[0] if len(x) else "")
        cc_mode.name = "vendor_primary_cc"
        data = data.merge(cc_mode, on="vendor_id", how="left")
        data["cost_center_deviation"] = (data["cost_center"] != data["vendor_primary_cc"]).astype(int)

        vendor_counts = data.groupby("vendor_id")["transaction_id"].transform("count")
        data["vendor_posting_count"] = vendor_counts
        data["vendor_recurrence_ratio"] = vendor_counts / len(data)

        dup_keys = data.groupby(["account_id", "vendor_id", "amount", "posting_date"]).cumcount()
        data["duplicate_posting_flag"] = (dup_keys > 0).astype(int)

        data["amount_log"] = np.log1p(data["amount"].clip(lower=0))

        features = data[self.FEATURE_COLUMNS].astype(float)
        return data, features
