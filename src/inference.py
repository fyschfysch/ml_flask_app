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
    """Инференс модели обучения: предикт для user_id."""

    def __init__(self, model_path, columns_to_drop=None):
        self.pipeline = dill.load(open(model_path, 'rb'))
        self.columns_to_drop = columns_to_drop or [
            'user_id', 'course_id', 'real_course_progress', 'course_success'
        ]

    def predict_for_user(self, csv_path, user_id):
        df = pd.read_csv(csv_path)
        logger.info(f"Загружено {df.shape[0]} строк, {df.shape[1]} столбцов.")
        df_user = df[df['user_id'] == user_id]
        if df_user.empty:
            logger.warning(f"user_id {user_id} не найден в датасете.")
            return None
        logger.info(f"Найдено {df_user.shape[0]} строк для user_id={user_id}")

        # Формируем список столбцов для удаления
        week_pattern = re.compile(r'^risk_status_\d+_week$')
        drop_cols = [col for col in df_user.columns if week_pattern.match(col)]
        for col in ['risk_status', 'week']:
            if col in df_user.columns:
                drop_cols.append(col)
        for col in self.columns_to_drop:
            if col in df_user.columns and col not in drop_cols:
                drop_cols.append(col)
        X_input = df_user.drop(columns=drop_cols, errors='ignore')
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
