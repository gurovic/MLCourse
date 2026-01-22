# MLCourse  
Курс машинного обучения для школьников

**Авторы**: *Владимир Михайлович Гуровиц (школа "Летово", [@gurovic](https://t.me/gurovic)), DeepSeek, Qwen, ChatGPT*

🟢 тема прорецензирована экспертом  🟡 тема подготовлена автором  🔴 тема в процессе разработки

**Общая информация**

* [Классификация задач и методов их решения](problems.md)
* [Этапы решения задач ML](methods.md)
* [Олимпиады и соревнования](olympiads.md)
* [GPU-замены в коде](gpu.md)

**Список уроков (2025/26)**

<details>
<summary><b>Первое полугодие</b></summary>

* **Пара 1. Введение**
  * DS, AI, ML, DL, LLM
  * Обучение с учителем/без учителя/RL
  * Кружки и олимпиады
  * Как устроен курс
  * Python vs numpy vs pandas
  * colab - kaggle - Jupiter
  * этапы решения задачи ML
 
 * **Пара 2. Линейная регрессия. Чтение данных**
   * Задача регрессии
   * Линейная регрессия
   * MAE, MSE
   * Градиентный спуск "на пальцах"
   * Feature engineering для линейной регрессии
   * Python, numpy, pandas
   * Чтение данных. Типичные проблемы.
  
* **Пара 3. Чтение данных**
  * Теория (см. Блок 1)
  * Задачи (см. Блок 1) 

* **Пара 4. Типы данных и их специфика**
  * Теория (см. Блок 1)
  * Задачи (см. Блок 1)
  * KNN (теория)

* **Пара 5. Первая обученная модель: kNN**
  * Обсуждаем тонкие вопросы: размерность данных, классификацию более чем на два класса, отсутствие обучающей фазы (но необходимость использовать .fit в sklearn)
  * Практика: ирисы с kNN

* **Пара 6. Линейная регрессия/Синтетические данные**
  * Обсуждаем kNN (разделение на train-val-test, что делает .fit, зачем нормализация, что делать с тремя классами в случае равенства, какова размерность...)
  * Практика: Бостон и линейная регрессия (начинающие)
  * Обсуждение синтетических данных (плотность вероятности, нормальное распределение, дисбаланс классов) (продолжающие, см. Блок 1)
  * Логистическая регрессия: введение

* **Пара 7. Логистическая регрессия**
  * Обсуждение логистической регрессии
  * Решение упражнений и кейсов

* **Пара 8. Анализ данных**
  * Кратко обсудили все основные аспекты анализа данных
  * Начали решать задачи по темам: базовая статистика, визуализация, простая очистка данных, глубокий EDA (Блок 1)

* **Пара 8. Когда применять KNN?**
  * Обсуждение алгоритмической сложности и проклятия размерности.
  * Продолжаем решать задачи по темам: базовая статистика, визуализация, простая очистка данных, глубокий EDA (Блок 1)
 
* **Пара 9. Дорешивание EDA. Деревья**
  * Продолжаем решать задачи по темам: базовая статистика, визуализация, простая очистка данных, глубокий EDA (Блок 1)
  * Деревья
 
* **Пара 10. Ансамбли.**
  * Обсуждении теории по [презентации](https://github.com/gurovic/MLCourse/blob/main/%D0%90%D0%BD%D1%81%D0%B0%D0%BC%D0%B1%D0%BB%D0%B8%20%D0%B2%20ML.%20%D0%93%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BD%D1%8B%D0%B9%20%D0%B1%D1%83%D1%81%D1%82%D0%B8%D0%BD%D0%B3.pptx)
 
* **Пара 11. Решение задач на ансамбли (см. ниже)**

* **Пара 12. Решение задач на ансамбли. Обсуждение методов обработки данных (пропуски, нормализация, ...)**
</details> 

<details>
<summary><b>Второе полугодие</b></summary>

* **Пара 1. Подготовка к региону. Линейная алгебра.**
* **Пара 2. Кластеризация (kmeans, кратко о dbscan)**
* **Пара 3. Подготовка к региону: Метод Монте-Карло. Оптимизация (частные производные).**

Deep Learning

* **Пара 4. Pytorch. MLP**
</details>

**Теоретический материал, задачи, датасеты**

**Блок 1: Анализ данных**  
* 🟡 [Чтение данных](010_read.ipynb)
  * 🟡 [Задачи](010_read_tasks.md)
* 🟡 [Типы данных и их специфика](012_types.ipynb): числовые/категориальные/временные/гео/текст
  * 🔴 [Задачи](012_types_tasks.md)
* 🔴 [Создание синтетических данных*](013_create_data.ipynb): numpy.random, sklearn.datasets.make_classification, sklearn.datasets.make_regression, данные для A/B-тестирования
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
* 🟡 [Дисбаланс классов*](030_disbalance.ipynb): SMOTE, Weighted Loss.
  * Практика: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle)
  * 🔴 [Задачи](030_disbalance_tasks,md)

**Блок 1.5: Вспомогательные темы и приемы**
* 🔴 [Особенности синтаксиса pandas](105_pandas_syntax.md)
* 🟡 [Предобработка для моделей](107_scaling.ipynb): концепция масштабирования (StandardScaler/MinMaxScaler)
  * 🔴 [Практика](107_scaling_practice.md): Показать разницу в качестве kNN с масштабированием и без на moons/iris.
* 🔴 [Градиентный спуск](080_gradient_descent.md)
  * [Почему размер батча - степень двойки?](081_batch2.md)
* 🔴 [Estimator и Transformer](090_estimator_transformer.md) в sklearn
* Pipeline в sklearn

**Блок 2: Классические алгоритмы**  
* 🟡 [kNN](103_knn.ipynb)
  * Практика: Iris Dataset (библиотека sklearn)
  * Когда использовать KNN: ускорение алгоритмов, проклятие размерности
  * [cheatsheet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCourse/blob/main/knn_cheatsheet.html)
* 🔴 [Деревья решений](150_decision_tree.ipynb): Критерии Gini/энтропия.
  * 🟡 [Почему мы строим дерево жадно?](150_decision_tree_greedy.md) 
  * Практика: Iris Dataset (библиотека sklearn)
  * 🔴 Практика: [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) (scikit-learn).  
* **ЛИНЕЙНЫЕ МОДЕЛИ**
  * 🟡 [*Что такое линейные модели?*](108_linear.md)
  * 🔴 [*Почему линейные модели до сих пор используются?*](109_why_linear.md)
  * 🔴 [Линейная регрессия](110_linreg.ipynb): MSE, градиентный спуск.
    * Практика: Boston Housing Dataset
    * 🔴 [Регуляризация](116_regularization.ipynb) (Ridge/Lasso)
    * Практика: прогнозирование цен на жилье.
    * [cheatsheet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCourse/blob/main/linreg_cheatsheet.html)  
  * 🔴 [Логистическая регрессия](120_logreg.md): Sigmoid, бинарная классификация.
    * 🟡 [*Почему сигмоида?*](121_why_sigmoid.md)
    * [Что делать с log(0)](122_log0.md)
    * Практика: Iris Dataset (библиотека sklearn)
    * [cheatsheet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCourse/blob/main/logreg_cheatsheet.html)
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
* 🔴 [Voting](150_voting.ipynb)
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
  * [cheatsheet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCourse/blob/main/boosting_cheatsheet.html)
* 🔴 [Стекинг](190_stacking.ipynb): CatBoost + ...
  * Практика: [Tabular Playground Series](https://www.kaggle.com/c/tabular-playground-series) (Kaggle).  
* 🔴 [Интерпретация: важность признаков](195_feature_importances.ipynb) (feature importances) для бэггинга и бустинга.
  * Практика: House Prices/Wine Quality.
 
**Блок 5: Feature Engineering**  
- 🔴 [Пропуски данных](310_drops.md). Практика: [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-datasets) (Kaggle).  
- 🔴 [Выбросы](320_outliers.md): Isolation Forest. Практика: [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) (Kaggle).  
- ◯ [Категориальные признаки](323_cat_features.ipynb): One-Hot Encoding, Label Encoding, Target Encoding
  -  Практика: House Prices/Amazon Employee. 
- 🔴 [Создание признаков](325_creating_features.ipynb): генерация полиномиальных признаков (для линейных моделей), взаимодействие признаков, агрегаты (для реляционных данных), признаки из дат (день недели, месяц).
- 🔴 [Временные ряды](330_time_series.md): Лаги, скользящие средние. Практика: [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only) (Kaggle).  
- 🔴 [Текст](340_text_feature_engineering.md): TF-IDF, word2vec, FastText. Практика: [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) (Stanford).  
- 🔴 [Геоданные](350_geo_features.md): Кластеризация, расстояния. Практика: [NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) (Kaggle).
- 🔴 [Утечки данных](360_leak.md)
  - Практика: Анализ на Spaceship Titaniс
- ◯ Автоматический Feature Engineering 
- ◯ Kaggle Challenge: Полный цикл решения. Практика: [Spaceship Titanic](https://www.kaggle.com/c/spaceship-titanic) (Kaggle).  

**Блок 6: Нейросети**  

**6.1 Основы нейронных сетей**
- 🟢 [PyTorch Basics](410_pytorch.md): Тензоры, autograd, базовые операции
  - 🟢 [Задачи](410_pytorch_tasks.md)
- 🟢 [GPU в PyTorch](411_pytorch_gpu.md): Ускорение вычислений на GPU
  - 🔴 [Задачи](411_pytorch_gpu_tasks.md)
- 🟢 [Полносвязные нейросети (MLP)](420_mlp.md): Архитектура, функции активации (ReLU, Sigmoid, Tanh)
  - 🔴 [Задачи](420_mlp_tasks.md)
  - Практика: [MNIST Handwritten Digits](https://pytorch.org/vision/stable/datasets.html#mnist) (PyTorch datasets)
- 🔴 [Batch, dataset, dataloader](4205_batch_dataset_dataloader.md)
  - 🔴 [Задачи](4205_batch_dataset_dataloader.md)
- 🔴 [Обратное распространение ошибки](421_backpropagation.md): Математика градиентного спуска, цепное правило
  - 🔴 [Задачи](421_backpropagation_tasks.md)
  - Практика: Реализация простой нейросети с нуля на NumPy ([XOR Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html))
- 🔴 [Функции потерь](422_loss_functions.md): MSE, Cross-Entropy, Binary Cross-Entropy
  - 🔴 [Задачи](422_loss_functions_tasks.md)
  - Практика: Сравнение функций потерь на задачах регрессии и классификации ([Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html))
- 🔴 [Оптимизаторы](423_optimizers.md): SGD, Adam, RMSprop, AdaGrad
  - 🔴 [Задачи](423_optimizers_tasks.md)
  - Практика: Сравнение скорости сходимости разных оптимизаторов ([Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist))

**6.2 Продвинутые техники обучения**
- 🔴 [Регуляризация в нейросетях](430_regularization.md): Dropout, Batch Normalization, Layer Normalization, Weight Decay
  - 🔴 [Задачи](430_regularization_tasks.md)
  - Практика: Борьба с переобучением на [MNIST](https://pytorch.org/vision/stable/datasets.html#mnist)
- 🔴 [Инициализация весов](431_weight_init.md): Xavier, He initialization
  - 🔴 [Задачи](431_weight_init_tasks.md)
  - Практика: Влияние инициализации на сходимость ([Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist))
- 🔴 [Learning Rate Scheduling](432_lr_scheduling.md): StepLR, ReduceLROnPlateau, CosineAnnealing
  - 🔴 [Задачи](432_lr_scheduling_tasks.md)
  - Практика: Подбор оптимального schedule для обучения ([CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html))
- 🔴 [Early Stopping и Callbacks](433_callbacks.md): Мониторинг метрик, сохранение лучших моделей
  - 🔴 [Задачи](433_callbacks_tasks.md)
  - Практика: Настройка пайплайна обучения с early stopping ([Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist))
- 🔴 [Data Augmentation](434_augmentation.md): Техники аугментации для различных типов данных
  - 🔴 [Задачи](434_augmentation_tasks.md)
  - Практика: Улучшение качества модели через аугментацию ([CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html))

**6.3 Сверточные нейронные сети (CNN)**
- 🔴 [Основы CNN](440_cnn_basics.md): Сверточные слои, pooling, stride, padding
  - 🔴 [Задачи](440_cnn_basics_tasks.md)
  - Практика: Простая CNN для [MNIST](https://pytorch.org/vision/stable/datasets.html#mnist)
- 🔴 [Архитектуры CNN](441_cnn_architectures.md): LeNet, AlexNet, VGG, ResNet, Inception
  - 🔴 [Задачи](441_cnn_architectures_tasks.md)
  - Практика: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (University of Toronto)
- 🔴 [Transfer Learning](442_transfer_learning.md): Fine-tuning, feature extraction, предобученные модели
  - 🔴 [Задачи](442_transfer_learning_tasks.md)
  - Практика: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) (Kaggle)
- 🔴 [Современные архитектуры](443_modern_cnn.md): EfficientNet, MobileNet, Vision Transformer (ViT)
  - 🔴 [Задачи](443_modern_cnn_tasks.md)
  - Практика: Сравнение производительности разных архитектур ([ImageNet subset](https://www.kaggle.com/c/imagenet-object-localization-challenge))
- 🔴 [Object Detection](444_object_detection.md): YOLO, R-CNN, SSD
  - 🔴 [Задачи](444_object_detection_tasks.md)
  - Практика: Детекция объектов на пользовательских данных ([COCO Dataset](https://cocodataset.org/))

**6.4 Рекуррентные нейронные сети (RNN)**
- 🔴 [Основы RNN](450_rnn_basics.md): Архитектура RNN, проблема затухающих градиентов
  - 🔴 [Задачи](450_rnn_basics_tasks.md)
  - Практика: Генерация текста на уровне символов ([Shakespeare Text](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays))
- 🔴 [LSTM и GRU](451_lstm_gru.md): Long Short-Term Memory, Gated Recurrent Unit
  - 🔴 [Задачи](451_lstm_gru_tasks.md)
  - Практика: Прогнозирование временных рядов ([Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality))
- 🔴 [Bidirectional RNN](452_bidirectional_rnn.md): Обработка последовательностей в обоих направлениях
  - 🔴 [Задачи](452_bidirectional_rnn_tasks.md)
  - Практика: Анализ тональности отзывов ([IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/))
- 🔴 [Sequence-to-Sequence](453_seq2seq.md): Encoder-Decoder архитектура, машинный перевод
  - 🔴 [Задачи](453_seq2seq_tasks.md)
  - Практика: Простой переводчик текста ([WMT English-German](https://www.statmt.org/wmt14/translation-task.html))

**6.5 Natural Language Processing (NLP)**
- 🔴 [Эмбеддинги слов](460_embeddings.md): Word2Vec, GloVe, FastText
  - 🔴 [Задачи](460_embeddings_tasks.md)
  - Практика: Визуализация word embeddings ([Text8 Corpus](https://mattmahoney.net/dc/textdata.html))
- 🔴 [Attention механизм](461_attention.md): Self-attention, multi-head attention
  - 🔴 [Задачи](461_attention_tasks.md)
  - Практика: Визуализация attention weights ([WMT English-German](https://www.statmt.org/wmt14/translation-task.html))
- 🔴 [Transformer](462_transformer.md): Архитектура Transformer, позиционное кодирование
  - 🔴 [Задачи](462_transformer_tasks.md)
  - Практика: Sentiment analysis с Transformer ([SST-2 Dataset](https://nlp.stanford.edu/sentiment/))
- 🔴 [BERT и его варианты](463_bert.md): BERT, RoBERTa, DistilBERT, предобучение и fine-tuning
  - 🔴 [Задачи](463_bert_tasks.md)
  - Практика: [Jigsaw Toxic Comments Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Kaggle)
- 🔴 [Hugging Face Transformers](464_huggingface.md): Использование готовых моделей, tokenizers, pipelines
  - 🔴 [Задачи](464_huggingface_tasks.md)
  - Практика: Текстовая классификация с предобученными моделями ([AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset))
- 🔴 [Генеративные модели для текста](465_text_generation.md): GPT, T5, генерация и fine-tuning
  - 🔴 [Задачи](465_text_generation_tasks.md)
  - Практика: Fine-tuning GPT-2 для генерации текста ([WikiText-2](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/))

**6.6 Генеративные модели**
- 🔴 [Autoencoders](470_autoencoders.md): Variational Autoencoders (VAE), применения
  - 🔴 [Задачи](470_autoencoders_tasks.md)
  - Практика: Сжатие и реконструкция изображений [MNIST](https://pytorch.org/vision/stable/datasets.html#mnist)
- 🔴 [Generative Adversarial Networks (GAN)](471_gan.md): Архитектура GAN, discriminator, generator
  - 🔴 [Задачи](471_gan_tasks.md)
  - Практика: Генерация изображений цифр ([MNIST](https://pytorch.org/vision/stable/datasets.html#mnist))
- 🔴 [Продвинутые GAN](472_advanced_gan.md): DCGAN, StyleGAN, условные GAN (cGAN)
  - 🔴 [Задачи](472_advanced_gan_tasks.md)
  - Практика: Генерация изображений с условиями ([CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))
- 🔴 [Диффузионные модели](473_diffusion.md): DDPM, Stable Diffusion, основы
  - 🔴 [Задачи](473_diffusion_tasks.md)
  - Практика: Генерация изображений с диффузионной моделью ([CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html))

**6.7 Специализированные темы**
- 🔴 [Graph Neural Networks (GNN)](480_gnn.md): Основы GNN, Graph Convolutional Networks
  - 🔴 [Задачи](480_gnn_tasks.md)
  - Практика: Классификация узлов графа ([Cora Dataset](https://relational.fit.cvut.cz/dataset/CORA))
- 🔴 [Рекомендательные системы](481_recommender.md): Коллаборативная фильтрация, нейронные подходы
  - 🔴 [Задачи](481_recommender_tasks.md)
  - Практика: [MovieLens](https://grouplens.org/datasets/movielens/) рекомендации
- 🔴 [Meta-Learning](482_meta_learning.md): Few-shot learning, MAML
  - 🔴 [Задачи](482_meta_learning_tasks.md)
  - Практика: Обучение на малых данных ([Omniglot Dataset](https://github.com/brendenlake/omniglot))
- 🔴 [Reinforcement Learning основы](483_rl_basics.md): Q-learning, DQN, Policy Gradient
  - 🔴 [Задачи](483_rl_basics_tasks.md)
  - Практика: Обучение агента в простой среде ([OpenAI Gym CartPole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/))

**6.8 Практические аспекты**
- 🔴 [Отладка нейросетей](490_debugging.md): Диагностика проблем обучения, визуализация
  - 🔴 [Задачи](490_debugging_tasks.md)
  - Практика: Исправление типичных ошибок ([MNIST](https://pytorch.org/vision/stable/datasets.html#mnist) с намеренными ошибками)
- 🔴 [Оптимизация производительности](491_performance.md): Mixed precision, gradient accumulation, distributed training
  - 🔴 [Задачи](491_performance_tasks.md)
  - Практика: Ускорение обучения больших моделей ([ImageNet subset](https://www.kaggle.com/c/imagenet-object-localization-challenge))
- 🔴 [Развертывание моделей](492_deployment.md): ONNX, TorchScript, сервисы для inference
  - 🔴 [Задачи](492_deployment_tasks.md)
  - Практика: Развертывание модели в продакшн ([MNIST](https://pytorch.org/vision/stable/datasets.html#mnist) classifier deployment)
- 🔴 [MLOps для нейросетей](493_mlops.md): Tracking экспериментов (Weights & Biases, MLflow), версионирование моделей
  - 🔴 [Задачи](493_mlops_tasks.md)
  - Практика: Настройка MLOps пайплайна ([Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist))

**6.9 Интерпретируемость и продвинутые темы**
- 🔴 [Интерпретируемость нейросетей](500_interpretability.md): Saliency Maps, Grad-CAM, SHAP, LIME, Integrated Gradients
  - 🔴 [Задачи](500_interpretability_tasks.md)
  - Практика: Объяснение предсказаний CNN ([MNIST](https://pytorch.org/vision/stable/datasets.html#mnist), CIFAR-10)
- 🔴 [Мультимодальное обучение](510_multimodal.md): CLIP, Image Captioning, VQA, Multimodal Transformers
  - 🔴 [Задачи](510_multimodal_tasks.md)
  - Практика: Image-Text retrieval ([Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/), [COCO](https://cocodataset.org/))

**Блок 7: Продвинутые соревновательные методы и алгоритмы**  
- 🔴 [Метод Монте-Карло: cheatsheet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCourse/blob/main/monte-carlo_cheatsheet.html)
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
* 🟢 [Обзор и сравнение библиотек Deep Learning](520_dl_frameworks.md): PyTorch, TensorFlow/Keras, JAX, ONNX
  * [Задачи](520_dl_frameworks_tasks.md)  
