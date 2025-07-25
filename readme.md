# ML Flask App (локальный запуск)

## Описание

Локальная версия веб-приложения препроцессинга данных, обучения моделей прогнозирования групп риска на следующей неделе и инференса лучшей модели. Пользовательский интерфейс реализован на Flask.

## Структура каталогов проекта

```
.
├── app.py                   # Основной файл запуска Flask-приложения
├── requirements.txt         # Список необходимых Python-зависимостей
├── saved_datasets/          # Каталог для сохраненных обучающего и тестового датасетов
│   ├── train_combined.csv
│   └── test_combined.csv
├── models/                  # Каталог для сериализованных моделей
│   └── model.pkl
├── data/                    # Каталог с исходными датасетами (выборки по курсам 1Т Старт, Дата, БАС)
├── static/                  # Статические файлы интерфейса (CSS)
│   └── style.css
├── templates/               # HTML-шаблоны для веб-интерфейса
│   └── index.html
├── src/
│   ├── preprocessors.py     # Препроцессинг обучающих данных
│   ├── train.py             # Логика обучения моделей
│   └── inference.py         # Выполнение инференса по загруженной модели
└── readme.md                # Файл документации
```


## Установка зависимостей

Для корректной работы проекта требуется установить указанные в `requirements.txt` зависимости. Установку производить командой:

```
pip install -r requirements.txt
```

Если используется отдельная среда (рекомендуется), все команды выполнять внутри неё.

## Описание этапов работы

### 1. Подготовка данных

- Каталог `data/` содержит три исходных CSV-файла: `13_week_data1.csv`, `13_week_data2.csv`, `13_week_bas.csv`.
- Для исключения дублирования номеров курсов используется корректировка столбца `course_id` в третьем датасете.


### 2. Препроцессинг

- Реализован в модуле `src/preprocessors.py`.
- Для всех трёх датасетов выполняется объединение, формирование срезов по неделям и удаление необязательных признаков.
- После обработки формируются два файла: `train_combined.csv` и `test_combined.csv` в каталоге `saved_datasets/`.


### 3. Обучение модели

- Производится запуск модуля `src/train.py` или через соответствующий маршрут веб-интерфейса.
- Автоматически перебираются несколько модельных алгоритмов (RandomForest, CatBoost, LightGBM, XGBoost и др.).
- Лучший пайплайн сохраняется как `model.pkl` в каталоге `models/`.


### 4. Развёртывание веб-интерфейса и инференс

- Запуск сервиса осуществляется командой:

```
python app.py
```

- Открывается веб-интерфейс по адресу `http://localhost:5000`
- Для инференса пользователь загружает CSV-файл с исходными признаками и указывает идентификатор пользователя.


## Краткое описание файлов

| Файл/Каталог | Назначение |
| :-- | :-- |
| app.py | Flask-приложение, маршруты для препроцессинга, обучения, инференса |
| requirements.txt | Зависимости Python |
| saved_datasets/ | Хранение промежуточных и итоговых датасетов |
| models/ | Сохранение обученной модели |
| data/ | Исходные обучающие датасеты |
| static/style.css | Оформление веб-интерфейса |
| templates/index.html | HTML-шаблон интерфейса |
| src/preprocessors.py | Обработка и агрегация данных перед обучением |
| src/train.py | Обучение моделей и их параметрический подбор |
| src/inference.py | Выполнение предсказаний по загруженной модели |

## Особенности реализации

- Корректировка идентификаторов курсов для третьего датасета.
- Автоматическая фильтрация и формирование обучающих срезов по неделям.
- Возможность выбора пользователя для персонализанного инференса.
- Используемая сериализация моделей — dill.
- Поддержка большинства современных ML-алгоритмов табличных данных.


## Рекомендации по запуску

- Все операции рекомендуется проводить в консольном окне с активацией соответствующей Python-среды.
- В случае необходимости актуализировать версии зависимостей согласно `requirements.txt` и убедиться в их корректной установке перед запуском.


## Предупреждения

- Весь процесс инференса и обучения требует неизменного состава файловых каталогов и корректно сформированных исходных датасетов.
- Версии библиотек, используемых для обучения и инференса, должны совпадать — это критично для сериализации/десериализации моделей.