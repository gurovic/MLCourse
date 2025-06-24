# MLCourse  
Курс машинного обучения для школьников

**Авторы**: *Владимир Михайлович Гуровиц (школа "Летово", [@gurovic](https://t.me/gurovic)), DeepSeek, Qwen*

🟢 тема прорецензирована экспертом
🟡 тема подготовлена автором
🔴 тема в процессе разработки

* [Классификация задач и методов их решения](problems.md)
* [Этапы решения задач ML](methods.md)
* [Олимпиады и соревнования](olympiads.md)

**Блок 1: Анализ данных**  
* 🟡 [Чтение данных](010_read.ipynb)
* 🟡 [Визуализация](015_visualization.ipynb): Plotly, Folium.
  * 🔴 Практика: [COVID-19 Global Forecasting](https://www.kaggle.com/imdevskp/corona-virus-report) (Kaggle).  
* 🟡 [Глубокий EDA](019_EDA.ipynb): Pandas Profiling, анализ распределений и корреляций.
  * 🟡 [Задачи](019_EDA_tasks.md)
  * 🔴 Практика: [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-datasets) (Kaggle).  
* 🟡 [Дисбаланс классов](030_disbalance.ipynb): SMOTE, Weighted Loss.
  * 🔴 Практика: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle).  

**Блок 2: Классические алгоритмы**  
* 🔴 [Линейная регрессия](110_linreg.md): MSE, градиентный спуск.
  * 🔴 Практика: [Boston Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) (scikit-learn).
* Регуляризация (Ridge/Lasso) + практика: прогнозирование цен на жилье.  
* 🔴 [Логистическая регрессия](120_logreg.md): Sigmoid, бинарная классификация.
  * 🔴 Практика: [SMS Spam Collection](https://www.kaggle.com/uciml/sms-spam-collection-dataset) (UCI / Kaggle).
* kNN + практика: Iris Dataset (библиотека sklearn)
* SVM + практика: линейное/нелинейное разделение (например, moons dataset)
* Naive Bayes + практика: классификация текстов (дублирование SMS Spam Collection)
* 🟡 [Метрики качества](130_metrics.ipynb): F1, ROC-AUC.
  * 🔴 Практика: [Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) (scikit-learn).  
* 🟡 [Кросс-валидация](140_kfold.ipynb): Stratified K-Fold.
  * 🔴 [Bias-Variance Tradeoff](140_10_bias_variance.md)
  * 🔴 Практика: [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) (Kaggle).
* 🔴 [Деревья решений](150_decision_tree.ipynb): Критерии Gini/энтропия.
  * 🔴 Практика: [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) (scikit-learn).  

**Блок 3: Ансамбли** 
* 🟡 [Ансамблевые методы](145_ensemble.ipynb)
* 🔴 [Бэггинг - Случайный лес](160_random_forest.md): Бутстрэп, OOB-оценка.
  * 🔴 Практика: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) (Kaggle).  
* 🔴 [Бустинг. AdaBoost. Градиентный бустинг](170_boosting.md): CatBoost/XGBoost/LightGBM.
  * 🔴 Практика: [Wine Quality](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009) (UCI / Kaggle).  
* 🔴 [Категориальные признаки](180_cat_features.md): CatBoost (автокодирование).
  * 🔴 Практика: [Amazon Employee Access Challenge](https://www.kaggle.com/c/amazon-employee-access-challenge) (Kaggle).  
* 🔴 Стекинг CatBoost + нейросети.
  * Практика: [Tabular Playground Series](https://www.kaggle.com/c/tabular-playground-series) (Kaggle).  

**Блок 4: Feature Engineering**  
- 🔴 [Пропуски данных](310_drops.md). Практика: [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-datasets) (Kaggle).  
- 🔴 [Выбросы](320_outliers.md): Isolation Forest. Практика: [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) (Kaggle).  
- 🔴 [Временные ряды](330_time_series.md): Лаги, скользящие средние. Практика: [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only) (Kaggle).  
- 🔴 [Текст](340_text_feature_engineering.md): TF-IDF, FastText. Практика: [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) (Stanford).  
- 🔴 [Геоданные](350_geo_features.md): Кластеризация, расстояния. Практика: [NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) (Kaggle).
- 🔴 [Утечки данных](360_leak.md)
- 🔴 Kaggle Challenge: Полный цикл решения. Практика: [Spaceship Titanic](https://www.kaggle.com/c/spaceship-titanic) (Kaggle).  

**Блок 5: Нейросети**  
- 🔴 [PyTorch Basics](410_pytorch.md): Тензоры, autograd. Практика: [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/) (Yann LeCun).  
- 🔴 CNN: Сверточные слои, аугментация. Практика: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (University of Toronto).  
- 🔴 Трансферное обучение: Fine-tuning ResNet. Практика: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) (Kaggle).  
- 🔴 NLP: BERT, Hugging Face. Практика: [Jigsaw Toxic Comments Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Kaggle).  

**Блок 6: Соревновательные методы**  
- 🔴 Гиперпараметры: Optuna для CatBoost. Практика: [Santander Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction) (Kaggle).  
- 🔴 AutoML: H2O, TPOT. Практика: Сравнение с ручными моделями.  
- 🔴 Кастомные метрики: QWK, MAP@K. Практика: [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction) (Kaggle).  
- 🔴 Uplift-модели: CatBoost (S-Learner). Практика: [Marketing Campaign Effectiveness](https://www.kaggle.com/miroslavsabo/young-people-survey) (Kaggle).  
- 🔴 Кластеризация: Метрики ARI/AMI. Практика: [Customer Segmentation](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python) (Kaggle).  
- 🔴 Мультимодальность: Объединение таблиц, текста, изображений. Практика: [H&M Personalized Fashion Recommendations](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations) (Kaggle).
- PCA/t-SNE + практика: визуализация многомерных данных (например, MNIST)
- PCA/t-SNE + практика: визуализация многомерных данных (например, MNIST)
- Иерархическая кластеризация + практика: биологические данные (гены)
  
**Дополнительные темы**
* 🔴 [Оптимизация памяти](040_memory.md): Сжатие типов данных.
  * 🔴 Практика: [NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) (Kaggle).  
