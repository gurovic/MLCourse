# MLCourse  
Курс машинного обучения для школьников.

* [Классификация задач и методов их решения](problems.md)
* [Этапы решения задач ML](methods.md)
* [Олимпиады и соревнования](olympiads.md)

**Блок 1: Анализ данных**  
* [Чтение данных](010_read.md)
* [Глубокий EDA](020_EDA.md): Pandas Profiling, анализ распределений и корреляций. Практика: [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-datasets) (Kaggle).  
* [Дисбаланс классов](030_disbalance.md): SMOTE, Weighted Loss. Практика: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle).  
* [Оптимизация памяти](040_memory.md): Сжатие типов данных. Практика: [NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) (Kaggle).  
* [Визуализация](050_visual.md): Plotly, Folium. Практика: [COVID-19 Global Forecasting](https://www.kaggle.com/imdevskp/corona-virus-report) (Kaggle).  

**Блок 2: Классические алгоритмы**  
- [Линейная регрессия](110_linreg.md): MSE, градиентный спуск. Практика: [Boston Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) (scikit-learn).  
- [Логистическая регрессия](120_logreg.md): Sigmoid, бинарная классификация. Практика: [SMS Spam Collection](https://www.kaggle.com/uciml/sms-spam-collection-dataset) (UCI / Kaggle).  
- [Метрики качества](130_metrics.md): F1, ROC-AUC. Практика: [Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) (scikit-learn).  
- [Кросс-валидация](140_kfold.md): Stratified K-Fold. Практика: [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) (Kaggle).  
- [Деревья решений](150_decision_tree.md): Критерии Gini/энтропия. Практика: [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) (scikit-learn).  
- Случайный лес: Бутстрэп, OOB-оценка. Практика: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) (Kaggle).  
- Градиентный бустинг: CatBoost/XGBoost/LightGBM. Практика: [Wine Quality](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009) (UCI / Kaggle).  
- Категориальные признаки: CatBoost (автокодирование). Практика: [Amazon Employee Access Challenge](https://www.kaggle.com/c/amazon-employee-access-challenge) (Kaggle).  

**Блок 3: Feature Engineering**  
- [Пропуски данных](310_drops.md). Практика: [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-datasets) (Kaggle).  
- Выбросы: Isolation Forest. Практика: [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) (Kaggle).  
- Временные ряды: Лаги, скользящие средние. Практика: [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only) (Kaggle).  
- Текст: TF-IDF, FastText. Практика: [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) (Stanford).  
- Геоданные: Кластеризация, расстояния. Практика: [NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) (Kaggle).
- [Утечки данных](360_leak.md)
- Kaggle Challenge: Полный цикл решения. Практика: [Spaceship Titanic](https://www.kaggle.com/c/spaceship-titanic) (Kaggle).  

**Блок 4: Нейросети**  
- PyTorch Basics: Тензоры, autograd. Практика: [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/) (Yann LeCun).  
- CNN: Сверточные слои, аугментация. Практика: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (University of Toronto).  
- Трансферное обучение: Fine-tuning ResNet. Практика: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) (Kaggle).  
- NLP: BERT, Hugging Face. Практика: [Jigsaw Toxic Comments Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Kaggle).  

**Блок 5: Соревновательные методы**  
- Ансамбли: Стекинг CatBoost + нейросети. Практика: [Tabular Playground Series](https://www.kaggle.com/c/tabular-playground-series) (Kaggle).  
- Гиперпараметры: Optuna для CatBoost. Практика: [Santander Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction) (Kaggle).  
- AutoML: H2O, TPOT. Практика: Сравнение с ручными моделями.  
- Кастомные метрики: QWK, MAP@K. Практика: [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction) (Kaggle).  
- Uplift-модели: CatBoost (S-Learner). Практика: [Marketing Campaign Effectiveness](https://www.kaggle.com/miroslavsabo/young-people-survey) (Kaggle).  
- Кластеризация: Метрики ARI/AMI. Практика: [Customer Segmentation](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python) (Kaggle).  
- Мультимодальность: Объединение таблиц, текста, изображений. Практика: [H&M Personalized Fashion Recommendations](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations) (Kaggle).  
