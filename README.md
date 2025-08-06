# MLCourse  
Курс машинного обучения для школьников

**Авторы**: *Владимир Михайлович Гуровиц (школа "Летово", [@gurovic](https://t.me/gurovic)), DeepSeek, Qwen*

🟢 тема прорецензирована экспертом

🟡 тема подготовлена автором

🔴 тема в процессе разработки

* [Классификация задач и методов их решения](problems.md)
* [Этапы решения задач ML](methods.md)
* [Олимпиады и соревнования](olympiads.md)
* [GPU-замены в коде](gpu.md)

**Блок 1: Анализ данных**  
* 🟡 [Чтение данных](010_read.ipynb)
  * 🔴 [Задачи](010_read_tasks.md)
* 🟡 [Типы данных и их специфика](012_types.ipynb): числовые/категориальные/временные/гео/текст)
  * 🔴 [Задачи](012_types_tasks.md)
* 🔴 [Создание синтетических данных](013_create_data.ipynb): numpy.random, sklearn.datasets.make_classification, sklearn.datasets.make_regression, данные для A/B-тестирования
  * 🔴 [Задачи](013_create_data_tasks.md)
* 🔴 [Базовая статистика](016_base_stat.ipynb): описательные статистики, группировка.
  * Практика: COVID-19 или House pricing
  * 🔴 [Задачи](016_base_stat_tasks.md)
* 🟡 [Визуализация](015_visualization.ipynb): Plotly, Folium.
  * 🔴 [*Диаграммы с усами*](015_10_boxplot_whiskers.md) 
  * 🔴 [Задачи](015_visualization_tasks.md)
  * 🔴 Практика: [COVID-19 Global Forecasting](https://www.kaggle.com/imdevskp/corona-virus-report) (Kaggle).  
* 🔴 [Простая очистка данных](017_clean_data.ipynb): заполнение пропусков и устранение выбросов.
  * 🔴 [Задачи](017_clean_data_tasks.md)
  * Практика: COVID-19 или House pricing
* 🟡 [Глубокий EDA](019_EDA.ipynb): Pandas Profiling, анализ распределений и корреляций, предварительная гипотеза
  * 🟡 [Задачи](019_EDA_tasks.md)
  * Практика: [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-datasets) (Kaggle)  
* 🟡 [Дисбаланс классов](030_disbalance.ipynb): SMOTE, Weighted Loss.
  * Практика: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle)
  * 🔴 [Задачи](030_disbalance_tasks,md)

**Блок 1.5: Вспомогательные темы и приемы**
* 🟡 [Предобработка для моделей](107_scaling.ipynb): концепция масштабирования (StandardScaler/MinMaxScaler)
  * 🔴 [Практика](107_scaling_practice.md): Показать разницу в качестве kNN с масштабированием и без на moons/iris.
* 🔴 [Градиентный спуск](080_gradient_descent.md)
* 🔴 [Estimator и Transformer](090_estimator_transformer.md) в sklearn

**Блок 2: Классические алгоритмы**  
* 🟡 [kNN](103_knn.ipynb)
  * Практика: Iris Dataset (библиотека sklearn)
* 🔴 [Деревья решений](150_decision_tree.ipynb): Критерии Gini/энтропия.
  * Практика: Iris Dataset (библиотека sklearn)
  * 🔴 Практика: [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) (scikit-learn).  
* **ЛИНЕЙНЫЕ МОДЕЛИ**
  * 🟡 [*Что такое линейные модели?*](108_linear.md) 
  * 🔴 [Линейная регрессия](110_linreg.md): MSE, градиентный спуск.
    * Практика: Iris Dataset (библиотека sklearn)
    * Практика: [Boston Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) (scikit-learn).
    * 🔴 [Регуляризация](116_regularization.ipynb) (Ridge/Lasso)
    * Практика: прогнозирование цен на жилье.  
  * 🔴 [Логистическая регрессия](120_logreg.md): Sigmoid, бинарная классификация.
    * Практика: Iris Dataset (библиотека sklearn)
    * Практика: [SMS Spam Collection](https://www.kaggle.com/uciml/sms-spam-collection-dataset) (UCI / Kaggle).
  * 🔴 [SVM](130_svm.ipynb): линейное/нелинейное разделение
    SVM является **линейной** только с линейным ядром*
    * Практика: [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) (scikit-learn).
  * 🔴 [Перцептрон и однослойные нейросети](135_perceptron.ipynb)
* 🔴 [Наивный байесовский классификатор](140_naive_bayes.ipynb)
  * [Объяснение без математики](140_naive_bayes_child.md)
  * Практика: классификация текстов (SMS Spam Collection)
  * Практика: [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) (scikit-learn).  

**Блок 3: Валидация и оценка моделей** 
* 🟡 [Метрики качества](130_metrics.ipynb): F1, ROC-AUC, матрица качества, log_loss
  * 🔴 [Что такое ROC-AUC?](135_roc_auc.md)
  * 🔴 [Что такое R^2?](137_r_2.md)
  * 🟡 [Metrics vs loss function](130_1_metrics_vs_loss_function.md)
  * 🔴 Практика: [Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) (scikit-learn).  
  * Практика: Сравнить метрики на датасете с дисбалансом (Credit Card Fraud).
* [Разделение данных](138_data_split.ipynb)
* 🟡 [Кросс-валидация](140_kfold.ipynb): Stratified K-Fold.
  * 🔴 Практика: [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) (Kaggle).
* 🔴 [Bias-Variance Tradeoff](140_10_bias_variance.md)
* 🔴 [Кривые обучения](150_learning_curves.ipynb)
  
**Блок 4: Ансамбли** 
* 🟡 [Ансамблевые методы: обзор](145_ensemble.ipynb)
* ◯ Voting
  * Практика: Iris/Titanic.
* 🔴 [Бэггинг - Случайный лес](160_random_forest.md): Бутстрэп, OOB-оценка.
  * 🔴 Практика: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) (Kaggle), feature importance анализ.  
* 🔴 [Бустинг](170_boosting.md):
  * ◯ AdaBoost (базовый)
  * ◯ Градиентный бустинг (общий принцип)
  * ◯ Реализации: CatBoost/XGBoost/LightGBM.
  * 🔴 [Категориальные признаки и CatBoost](180_cat_features.md): автокодирование.
    * 🔴 Практика: [Wine Quality](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009) (UCI / Kaggle), сравнение XGBoost/LightGBM/CatBoost по скорости/качеству.  
    * 🔴 Практика: [Amazon Employee Access Challenge](https://www.kaggle.com/c/amazon-employee-access-challenge) (Kaggle).  
* 🔴 Стекинг: CatBoost + ...
  * Практика: [Tabular Playground Series](https://www.kaggle.com/c/tabular-playground-series) (Kaggle).  
* ◯ Интерпретация: Важность признаков (feature importances) для бэггинга и бустинга.
  * Практика: House Prices/Wine Quality.
 
**Блок 5: Feature Engineering**  
- 🔴 [Пропуски данных](310_drops.md). Практика: [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-datasets) (Kaggle).  
- 🔴 [Выбросы](320_outliers.md): Isolation Forest. Практика: [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) (Kaggle).  
- ◯ [Категориальные признаки](323_cat_features.ipynb): One-Hot Encoding, Label Encoding, Target Encoding
  -  Практика: House Prices/Amazon Employee. 
- 🔴 [Создание признаков](325_creating_features.ipynb): генерация полиномиальных признаков (для линейных моделей), взаимодействие признаков, агрегаты (для реляционных данных), признаки из дат (день недели, месяц).
- 🔴 [Временные ряды](330_time_series.md): Лаги, скользящие средние. Практика: [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only) (Kaggle).  
- 🔴 [Текст](340_text_feature_engineering.md): TF-IDF, FastText. Практика: [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) (Stanford).  
- 🔴 [Геоданные](350_geo_features.md): Кластеризация, расстояния. Практика: [NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) (Kaggle).
- 🔴 [Утечки данных](360_leak.md)
  - Практика: Анализ на Spaceship Titaniс
- ◯ Автоматический Feature Engineering 
- ◯ Kaggle Challenge: Полный цикл решения. Практика: [Spaceship Titanic](https://www.kaggle.com/c/spaceship-titanic) (Kaggle).  

**Блок 6: Нейросети**  
- 🔴 [PyTorch Basics](410_pytorch.md): Тензоры, autograd. Практика: [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/) (Yann LeCun).  
- ◯ CNN: Сверточные слои, аугментация. Практика: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (University of Toronto).  
- ◯ Трансферное обучение: Fine-tuning ResNet. Практика: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) (Kaggle).  
- ◯ NLP: BERT, Hugging Face. Практика: [Jigsaw Toxic Comments Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Kaggle).  

**Блок 7: Продвинутые соревновательные методы и алгоритмы**  
- ◯ Гиперпараметры: Optuna для CatBoost. Практика: [Santander Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction) (Kaggle).  
- ◯ AutoML: H2O, TPOT. Практика: Сравнение с ручными моделями.  
- ◯ Кастомные метрики: QWK, MAP@K. Практика: [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction) (Kaggle).  
- ◯ Uplift-модели: CatBoost (S-Learner). Практика: [Marketing Campaign Effectiveness](https://www.kaggle.com/miroslavsabo/young-people-survey) (Kaggle).  
- ◯ Кластеризация: Метрики ARI/AMI. Практика: [Customer Segmentation](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python) (Kaggle).  
- ◯ Иерархическая кластеризация + практика: биологические данные (гены)
- ◯ Мультимодальность: Объединение таблиц, текста, изображений. Практика: [H&M Personalized Fashion Recommendations](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations) (Kaggle).
- ◯ PCA/t-SNE + практика: визуализация многомерных данных (например, MNIST)
- ◯ DBSCAN + практика: обнаружение аномалий в транзакциях.
  
**Дополнительные темы**
* 🔴 [Оптимизация памяти](040_memory.md): Сжатие типов данных.
  * Практика: [NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) (Kaggle).  
