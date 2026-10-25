"""Model factory. Every model predicts the cross-sectional rank of next-period return."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class RankModel:
    def __init__(self, kind: str, params: dict):
        self.kind = kind
        if kind == "lgbm":
            from lightgbm import LGBMRegressor

            self.est = LGBMRegressor(**params.get("lgbm", {}))
        elif kind == "ridge":
            self.est = make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.5, keep_empty_features=True), StandardScaler(), Ridge(**params.get("ridge", {}))
            )
        else:
            raise ValueError(f"unknown model type {kind!r}")
        self.features: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RankModel":
        self.features = list(X.columns)
        self.est.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.est.predict(X[self.features])

    def importance(self) -> pd.Series:
        if self.kind == "lgbm":
            imp = self.est.booster_.feature_importance(importance_type="gain")
            s = pd.Series(imp, index=self.features)
        else:
            s = pd.Series(self.est[-1].coef_, index=self.features)
        return s
