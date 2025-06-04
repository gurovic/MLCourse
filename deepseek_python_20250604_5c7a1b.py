{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Exploratory Data Analysis (EDA): Исследовательский анализ данных**  \n",
    "\n",
    "## **Введение в EDA**  \n",
    "EDA — это искусство задавать правильные вопросы данным. Это первый и самый важный этап работы с данными, где мы:  \n",
    "- 🕵️‍♂️ **Знакомимся** с данными  \n",
    "- 🔍 **Выявляем** закономерности и аномалии  \n",
    "- 📊 **Визуализируем** ключевые характеристики  \n",
    "- 💡 **Формулируем** гипотезы для дальнейшего анализа  \n",
    "\n",
    "**Цели EDA:**  \n",
    "1. Понимание структуры данных  \n",
    "2. Обнаружение выбросов и ошибок  \n",
    "3. Выявление взаимосвязей между переменными  \n",
    "4. Проверка предположений для моделей"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## **🟢 Базовый уровень: Статистический анализ данных**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **1.1 Первичное знакомство с данными**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "\n",
    "# Загрузка данных\n",
    "df = sns.load_dataset('titanic')\n",
    "\n",
    "# Основная информация\n",
    "print(\"Размер данных:\", df.shape)\n",
    "print(\"\\nПервые 5 строк:\")\n",
    "display(df.head())\n",
    "\n",
    "print(\"\\nТипы данных:\")\n",
    "display(df.dtypes)\n",
    "\n",
    "print(\"\\nПропуски:\")\n",
    "display(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **1.2 Описательная статистика**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Для числовых переменных\n",
    "print(\"Описательная статистика числовых колонок:\")\n",
    "display(df.describe())\n",
    "\n",
    "# Для категориальных переменных\n",
    "print(\"\\nОписательная статистика категориальных колонок:\")\n",
    "display(df.describe(include=['object']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **1.3 Простые визуализации**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Гистограмма распределения\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(df['age'].dropna(), kde=True, bins=20)\n",
    "plt.title('Распределение возраста пассажиров')\n",
    "plt.xlabel('Возраст')\n",
    "plt.ylabel('Частота')\n",
    "plt.show()\n",
    "\n",
    "# Boxplot для обнаружения выбросов\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.boxplot(x=df['fare'])\n",
    "plt.title('Распределение стоимости билетов')\n",
    "plt.xlabel('Стоимость билета')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## **🟡 Продвинутый уровень: Анализ взаимосвязей**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **2.1 Корреляционный анализ**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Матрица корреляций\n",
    "plt.figure(figsize=(12, 8))\n",
    "corr_matrix = df.corr(numeric_only=True)\n",
    "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=\".2f\")\n",
    "plt.title('Матрица корреляций')\n",
    "plt.show()\n",
    "\n",
    "# Парные взаимосвязи\n",
    "sns.pairplot(df[['age', 'fare', 'pclass']].dropna())\n",
    "plt.suptitle('Парные распределения', y=1.02)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **2.2 Анализ категориальных переменных**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Круговые диаграммы\n",
    "plt.figure(figsize=(8, 8))\n",
    "df['sex'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'salmon'])\n",
    "plt.title('Распределение по полу')\n",
    "plt.ylabel('')\n",
    "plt.show()\n",
    "\n",
    "# Столбчатые диаграммы с группировкой\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.countplot(data=df, x='class', hue='survived', palette='pastel')\n",
    "plt.title('Класс билета vs Выживаемость')\n",
    "plt.xlabel('Класс')\n",
    "plt.ylabel('Количество')\n",
    "plt.legend(title='Выжил', labels=['Нет', 'Да'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## **🔴 Экспертный уровень: Продвинутые техники EDA**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **3.1 Многомерный анализ**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# FacetGrid для глубокого анализа\n",
    "g = sns.FacetGrid(df, col=\"sex\", row=\"survived\", height=4, aspect=1.5)\n",
    "g.map(sns.histplot, \"age\", bins=20, kde=True)\n",
    "g.add_legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **3.2 Автоматизированное EDA**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Установка библиотек (раскомментировать при первом запуске)\n",
    "# !pip install pandas-profiling sweetviz\n",
    "\n",
    "# Pandas Profiling\n",
    "from pandas_profiling import ProfileReport\n",
    "profile = ProfileReport(df, title='Автоматический EDA отчет Titanic')\n",
    "profile.to_file('titanic_report.html')\n",
    "\n",
    "# SweetViz\n",
    "import sweetviz as sv\n",
    "report = sv.analyze(df)\n",
    "report.show_html('sweetviz_report.html')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **3.3 Интерактивная визуализация**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Установка библиотек (раскомментировать при первом запуске)\n",
    "# !pip install plotly\n",
    "\n",
    "import plotly.express as px\n",
    "\n",
    "# Интерактивный scatter plot\n",
    "fig = px.scatter(\n",
    "    df.dropna(), \n",
    "    x='age', \n",
    "    y='fare',\n",
    "    color='survived',\n",
    "    size='pclass',\n",
    "    hover_data=['sex', 'class'],\n",
    "    title='Интерактивный анализ выживаемости'\n",
    ")\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## **📊 Чеклист по EDA**  \n",
    "\n",
    "### **Обязательные шаги:**  \n",
    "1. Анализ структуры данных (размеры, типы)  \n",
    "2. Проверка пропусков и выбросов  \n",
    "3. Анализ распределений (гистограммы, boxplot)  \n",
    "4. Исследование корреляций  \n",
    "5. Анализ категориальных переменных  \n",
    "\n",
    "### **Продвинутые техники:**  \n",
    "- Временные ряды (тренды, сезонность)  \n",
    "- Многомерная визуализация  \n",
    "- Геопространственный анализ  \n",
    "- Автоматизированные отчеты  \n",
    "\n",
    "### **Экспертные подходы:**  \n",
    "- Интерактивная визуализация  \n",
    "- Статистическое тестирование гипотез  \n",
    "- Анализ текстовых данных (word clouds, sentiment)  \n",
    "- PCA-анализ для уменьшения размерности"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## **💡 Золотые правила EDA**  \n",
    "1. **Начинайте с простого** — сначала базовые статистики, затем сложные визуализации  \n",
    "2. **Задавайте вопросы** — каждая визуализация должна отвечать на конкретный вопрос  \n",
    "3. **Итеративный процесс** — EDA редко бывает линейным, возвращайтесь к предыдущим шагам  \n",
    "4. **Документируйте** — сохраняйте графики и наблюдения  \n",
    "5. **Проверяйте гипотезы** — используйте статистические тесты для подтверждения идей  \n",
    "\n",
    "> \"EDA — это как первое свидание с данными: нужно задавать правильные вопросы, внимательно слушать ответы и не делать поспешных выводов.\"\n",
    "\n",
    "**Инструменты для углубленного изучения:**  \n",
    "- `Seaborn` — продвинутая статистическая визуализация  \n",
    "- `Plotly` — интерактивные графики  \n",
    "- `Geopandas` — геопространственный анализ  \n",
    "- `Pandas-profiling` — автоматизированные отчеты"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  },
  "colab": {
   "provenance": []
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}