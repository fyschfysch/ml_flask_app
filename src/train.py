import os
import dill
import logging
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier
)
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class InferencePreprocessor(BaseEstimator, TransformerMixin):
    """Приводит признаки к обученному набору для инференса."""
    def __init__(self):
        self.feature_columns_ = []
    def fit(self, X, y=None):
        self.feature_columns_ = X.columns.tolist()
        return self
    def transform(self, X):
        X = X.copy()
        for col in self.feature_columns_:
            if col not in X.columns:
                X[col] = 0
        extra_cols = [col for col in X.columns if col not in self.feature_columns_]
        X = X.drop(columns=extra_cols, errors='ignore')
        X = X.fillna(0)
        return X[self.feature_columns_]

def initialize_model(model_class, model_name, params):
    if model_name == "CatBoost":
        return model_class(**params, silent=True, random_state=42)
    if model_name == "XGBoost":
        return model_class(
            **params, objective='multi:softprob',
            eval_metric='mlogloss', use_label_encoder=False,
            random_state=42
        )
    if model_name == "LightGBM":
        return model_class(**params, random_state=42, verbosity=-1)
    return model_class(**params, random_state=42)

class ModelTrainer:
    """Обучает модели, сохраняет лучшую по F1-weighted (кросс-валидация) в model.pkl."""
    def __init__(self, model_dir, param_grids=None, cv=5):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {
            "RandomForest": RandomForestClassifier,
            "CatBoost": CatBoostClassifier,
            "LightGBM": lgb.LGBMClassifier,
            "XGBoost": xgb.XGBClassifier,
            "ExtraTrees": ExtraTreesClassifier,
            "HistGradientBoosting": HistGradientBoostingClassifier,
            "GradientBoosting": GradientBoostingClassifier
        }
        self.param_grids = param_grids if param_grids is not None else {}
        self.cv = cv

    def train_and_save_best(self, train_combined_df, target_column='risk_status',
                           columns_to_drop=None):
        if columns_to_drop is None:
            columns_to_drop = [
                'user_id', 'course_id', 'real_course_progress', 'course_success', 'week'
            ]
        X_train = train_combined_df.drop(
            columns=[target_column] + columns_to_drop, errors='ignore'
        )
        y_train = train_combined_df[target_column]
        best_score, best_pipeline, best_model_name = -float('inf'), None, None
        for model_name, model_class in self.models.items():
            logger.info(f"\n--- Обучение модели: {model_name} ---")
            pipeline = Pipeline([
                ('preprocessor', InferencePreprocessor()),
                ('model', initialize_model(model_class, model_name, {}))
            ])
            param_grid = {
                f'model__{k}': v for k, v in
                self.param_grids.get(model_name, {}).items()
            }
            if param_grid:
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid,
                    n_iter=5, cv=self.cv, scoring='f1_weighted',
                    n_jobs=-1, random_state=42, verbose=1
                )
                search.fit(X_train, y_train)
                score = search.best_score_
                pipeline_to_save = search.best_estimator_
            else:
                scores = cross_val_score(
                    pipeline, X_train, y_train,
                    cv=self.cv, scoring='f1_weighted', n_jobs=-1
                )
                score = scores.mean()
                logger.info(f"Кросс-валидация f1_weighted: {scores} (mean={score:.4f})")
                pipeline.fit(X_train, y_train)
                pipeline_to_save = pipeline
            logger.info(f"F1-weighted для {model_name}: {score:.4f}")
            if score > best_score:
                best_score = score
                best_pipeline = pipeline_to_save
                best_model_name = model_name
        save_path = os.path.join(self.model_dir, 'model.pkl')
        with open(save_path, 'wb') as f:
            dill.dump(best_pipeline, f)
        logger.info(f"Лучшая модель ({best_model_name}) сохранена в: {save_path}")
