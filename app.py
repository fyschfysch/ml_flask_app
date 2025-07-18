import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from src.preprocessors import TrainingDataPreprocessor
from src.train import ModelTrainer
from src.inference import ModelInferenceDemo
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_PATHS = [
    './data/13_week_data1.csv',
    './data/13_week_data2.csv',
    './data/13_week_bas.csv'
]
SAVE_DIR = './saved_datasets'
MODEL_DIR = './models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
TARGET_COLUMN = 'risk_status'
COLUMNS_TO_DROP = [
    'week', 'user_id', 'course_id', 'real_course_progress', 'course_success'
]
PARAM_GRIDS = {
    "RandomForest": {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [10, 20, 30],
        'max_features': ['sqrt']
    },
    "GradientBoosting": {
        'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5], 'subsample': [0.8]
    },
    "CatBoost": {
        'iterations': [50, 100, 200], 'learning_rate': [0.05, 0.1],
        'depth': [4, 6]
    },
    "LightGBM": {
        'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1],
        'max_depth': [5, 10], 'num_leaves': [31]
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5], 'subsample': [0.8]
    },
    "ExtraTrees": {
        'n_estimators': [50, 100, 200], 'max_depth': [10, 20],
        'max_features': ['sqrt']
    },
    "HistGradientBoosting": {
        'max_iter': [50, 100, 200], 'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
}

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess():
    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        df1 = pd.read_csv(RAW_DATA_PATHS[0])
        df2 = pd.read_csv(RAW_DATA_PATHS[1])
        df3 = pd.read_csv(RAW_DATA_PATHS[2])
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'One or more input files are empty.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    df3['course_id'] += 8000
    df_list = [df1, df2, df3]

    temp_paths = []
    for i, df in enumerate(df_list):
        temp_path = os.path.join(SAVE_DIR, f'temp_input_{i}.csv')
        df.to_csv(temp_path, index=False)
        temp_paths.append(temp_path)

    preprocessor = TrainingDataPreprocessor()
    train_combined_df, test_combined_df = preprocessor.process(temp_paths)

    train_path = os.path.join(SAVE_DIR, 'train_combined.csv')
    test_path = os.path.join(SAVE_DIR, 'test_combined.csv')
    train_combined_df.to_csv(train_path, index=False)
    test_combined_df.to_csv(test_path, index=False)

    for path in temp_paths:
        os.remove(path)

    return jsonify({'status': 'ok', 'train_path': train_path, 'test_path': test_path})

@app.route('/train', methods=['POST'])
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_path = os.path.join(SAVE_DIR, 'train_combined.csv')

    if not os.path.exists(train_path):
        return jsonify({'error': 'train_combined.csv not found, выполните препроцессинг'}), 400

    try:
        train_df = pd.read_csv(train_path)
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'train_combined.csv is empty.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    trainer = ModelTrainer(model_dir=MODEL_DIR, param_grids=PARAM_GRIDS, cv=5)
    trainer.train_and_save_best(
        train_combined_df=train_df,
        target_column=TARGET_COLUMN,
        columns_to_drop=COLUMNS_TO_DROP
    )

    return jsonify({'status': 'ok', 'model_path': MODEL_PATH})

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Начало обработки запроса /predict")

    if 'csv_file' not in request.files:
        logger.error("Нет части файла в запросе")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['csv_file']

    if file.filename == '':
        logger.error("Файл не выбран")
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            logger.info("Проверка размера файла")
            file.seek(0, os.SEEK_END)
            file_length = file.tell()
            file.seek(0)

            if file_length == 0:
                logger.error("Загруженный файл пуст")
                return jsonify({'error': 'Uploaded file is empty'}), 400

            logger.info("Чтение файла")
            df = pd.read_csv(file)
            user_id = int(request.form['user_id'])

            logger.info("Загрузка модели")
            model_infer = ModelInferenceDemo(MODEL_PATH)

            # Убедитесь, что метод predict_for_user ожидает правильные аргументы
            logger.info("Предсказание для пользователя")
            result = model_infer.predict_for_user(df, user_id)

            if result is None or result.empty:
                logger.error("user_id не найден или нет признаков")
                return jsonify({'error': 'user_id not found or no features'}), 404

            logger.info("Успешное завершение обработки запроса /predict")
            return result.to_json(orient='records', force_ascii=False)

        except pd.errors.EmptyDataError:
            logger.error("Загруженный файл пуст или недействителен")
            return jsonify({'error': 'Uploaded file is empty or invalid'}), 400
        except Exception as e:
            logger.error(f"Произошла ошибка: {str(e)}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
