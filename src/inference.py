import pandas as pd
import dill
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelInferenceDemo:
    """Инференс модели: только предсказание по user_id."""

    def __init__(self, model_path, columns_to_drop=None):
        self.pipeline = dill.load(open(model_path, 'rb'))
        self.columns_to_drop = columns_to_drop or [
            'user_id', 'course_id', 'real_course_progress', 'course_success'
        ]

    def predict_for_user(self, df_or_path, user_id):
        if isinstance(df_or_path, pd.DataFrame):
            df = df_or_path
        else:
            df = pd.read_csv(df_or_path)
        logger.info(f"Загружено {df.shape[0]} строк, {df.shape[1]} столбцов.")
        df_user = df[df['user_id'] == user_id]
        if df_user.empty:
            logger.warning(f"user_id {user_id} не найден в датасете.")
            return None
        logger.info(f"Найдено {df_user.shape[0]} строк для user_id={user_id}")

        week_pattern = re.compile(r'^risk_status_\d+_week$')
        drop_cols = set(col for col in df_user.columns if week_pattern.match(col))
        for col in ['risk_status', 'week']:
            if col in df_user.columns:
                drop_cols.add(col)
        for col in self.columns_to_drop:
            if col in df_user.columns:
                drop_cols.add(col)
        X_input = df_user.drop(columns=list(drop_cols), errors='ignore')
        if X_input.empty:
            logger.warning("Нет признаков для инференса после удаления столбцов.")
            return None

        y_pred = self.pipeline.predict(X_input)
        result = pd.DataFrame({
            'user_id': df_user['user_id'],
            'predicted_risk_status': y_pred
        })
        if 'week' in df_user.columns:
            result['week'] = df_user['week'].values

        logger.info("Предсказания успешно получены.")
        return result