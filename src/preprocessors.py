import pandas as pd
import re
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class TrainingDataPreprocessor:
    """Готовит train/test DataFrame с обнулением будущих недель и формированием целевого признака."""
    def __init__(self, target_prefix='risk_status', test_size=0.25, random_state=42):
        self.target_prefix = target_prefix
        self.test_size = test_size
        self.random_state = random_state

    @staticmethod
    def _get_week_from_col(col_name):
        match = re.search(r'_(\d+)_week$', col_name)
        return int(match.group(1)) if match else -1

    def _create_weekly_snapshots(self, df):
        all_records = []
        risk_status_cols = [col for col in df.columns if col.startswith(self.target_prefix)]
        id_cols = ['user_id', 'course_id']
        original_feature_cols = [
            col for col in df.columns
            if col not in id_cols and not col.startswith(self.target_prefix)
        ]
        for week_num in range(1, 14):
            target_col = f'{self.target_prefix}_{week_num + 1}_week'
            if target_col not in df.columns:
                continue
            week_df = df.copy()
            week_df['risk_status'] = week_df[target_col]
            week_df['week'] = week_num
            week_cols = [
                col for col in risk_status_cols
                if self._get_week_from_col(col) > week_num
            ]
            week_df[week_cols] = 0
            all_records.append(week_df)
        if not all_records:
            logger.warning("Не удалось создать еженедельные срезы данных.")
            return pd.DataFrame()
        combined_df = pd.concat(all_records, ignore_index=True)
        combined_df = combined_df.drop(columns=risk_status_cols, errors='ignore')
        final_order = ['week'] + id_cols + original_feature_cols + ['risk_status']
        final_order_existing = [col for col in final_order if col in combined_df.columns]
        return combined_df[final_order_existing]

    def process(self, raw_data_paths):
        """Читает пути к исходным CSV, возвращает train_df, test_df."""
        df1 = pd.read_csv(raw_data_paths[0])
        df2 = pd.read_csv(raw_data_paths[1])
        df3 = pd.read_csv(raw_data_paths[2])
        # Корректируем course_id только для БАС
        df3['course_id'] += 8000
        df_list = [df1, df2, df3]
        df_combined = pd.concat(df_list, ignore_index=True)
        train_df, test_df = train_test_split(
            df_combined, test_size=self.test_size, random_state=self.random_state
        )
        train_combined_df = self._create_weekly_snapshots(train_df)
        test_combined_df = self._create_weekly_snapshots(test_df)
        return train_combined_df, test_combined_df
