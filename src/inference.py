import pandas as pd
import dill
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class ModelInferenceDemo:
    """Инференс модели по CSV с фильтрацией по user_id."""

    def __init__(self, model_path, columns_to_drop=None):
        self.model_path = model_path
        self.pipeline = self._load_pipeline(model_path)
        self.columns_to_drop = columns_to_drop or [
            'user_id', 'course_id', 'real_course_progress', 'course_success'
        ]

    def _load_pipeline(self, path):
        with open(path, 'rb') as f:
            logger.info(f"Загружаю модель из {path}")
            return dill.load(f)

    def _drop_unused_columns(self, df):
        week_pattern = re.compile(r'^risk_status_\d+_week$')
        cols_to_drop = [col for col in df.columns if week_pattern.match(col)]
        for col in ['risk_status', 'week']:
            if col in df.columns:
                cols_to_drop.append(col)
        for col in self.columns_to_drop:
            if col in df.columns and col not in cols_to_drop:
                cols_to_drop.append(col)
        if cols_to_drop:
            logger.info(f"Удаляю столбцы: {cols_to_drop}")
        return df.drop(columns=cols_to_drop, errors='ignore')

    def predict_for_user(self, df, user_id):
        logger.info(f"Загружено {df.shape[0]} строк, {df.shape[1]} столбцов.")
        df_user = df[df['user_id'] == user_id]
        if df_user.empty:
            logger.warning(f"user_id {user_id} не найден в датасете.")
            return None
        logger.info(f"Найдено {df_user.shape[0]} строк для user_id={user_id}")
        X_input = self._drop_unused_columns(df_user)
        if X_input.empty:
            logger.warning("Нет признаков для инференса после удаления столбцов.")
            return None
        y_pred = self.pipeline.predict(X_input)
        y_pred_proba = None
        if hasattr(self.pipeline, 'predict_proba'):
            try:
                y_pred_proba = self.pipeline.predict_proba(X_input)
            except Exception:
                y_pred_proba = None
        result_df = df_user.copy()
        result_df['predicted_risk_status'] = y_pred
        if y_pred_proba is not None:
            for i, class_label in enumerate(self.pipeline.classes_):
                result_df[f'proba_{class_label}'] = y_pred_proba[:, i]
        return result_df
