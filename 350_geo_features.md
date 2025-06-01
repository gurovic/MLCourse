# **Feature Engineering для геоданных: Кластеризация, расстояния и пространственные признаки**  

## **Введение в работу с геоданными**  
Геоданные — информация с координатной привязкой (широта/долгота) — мощный источник для анализа в:  
- 🚕 Транспортных системах (оптимизация маршрутов)  
- 🏪 Ритейле (выбор локаций для магазинов)  
- 🌳 Экологии (анализ зон загрязнения)  

**Основные задачи feature engineering:**  
- Преобразование координат в информативные признаки  
- Выявление пространственных закономерностей  
- Учет географического контекста в моделях ML  

---

## **🟢 Базовый уровень: Преобразование координат**  

### **1.1 Расчет расстояний**  
```python
from geopy.distance import geodesic

# Расстояние между точками (в км)
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Пример: расстояние Москва - Санкт-Петербург
distance = calculate_distance(55.7558, 37.6173, 59.9343, 30.3351)
print(f"Расстояние: {distance:.1f} км")  # ~635 км
```

### **1.2 Геокодирование адресов**  
```python
from geopy.geocoders import Nominatim

def address_to_coords(address):
    geolocator = Nominatim(user_agent="geo_features")
    location = geolocator.geocode(address)
    return (location.latitude, location.longitude) if location else (None, None)

# Пример: координаты Красной площади
lat, lon = address_to_coords("Красная площадь, Москва")
print(f"Координаты: {lat}, {lon}")  # 55.7539, 37.6208
```

### **1.3 Базовые пространственные признаки**  
```python
import pandas as pd

def add_basic_features(df, ref_point=(55.7558, 37.6173)):
    """Добавляет признаки относительно контрольной точки"""
    df['distance_to_center'] = df.apply(
        lambda row: geodesic(ref_point, (row['lat'], row['lon'])).km, 
        axis=1
    )
    df['is_north'] = (df['lat'] > ref_point[0]).astype(int)
    return df
```

---

## **🟡 Продвинутый уровень: Кластеризация и пространственные взаимосвязи**  

### **2.1 Кластеризация точек DBSCAN**  
```python
from sklearn.cluster import DBSCAN
import numpy as np

def cluster_locations(coords, eps=0.01, min_samples=5):
    """Кластеризация географических точек"""
    # eps в градусах (~1.1 км на широте Москвы)
    coords_rad = np.radians(coords)  # преобразование для haversine
    clusterer = DBSCAN(
        eps=eps, 
        min_samples=min_samples, 
        metric='haversine'
    )
    labels = clusterer.fit_predict(coords_rad)
    return labels

# Пример использования
coords = np.array([[55.7522, 37.6156], [55.7530, 37.6165], ...])
df['cluster_label'] = cluster_locations(coords)
```

### **2.2 Расчет плотности точек**  
```python
from sklearn.neighbors import KernelDensity

def calculate_density(coords, radius=0.005):
    """Плотность точек в радиусе (в градусах)"""
    kde = KernelDensity(bandwidth=radius, metric='haversine')
    kde.fit(np.radians(coords))
    densities = np.exp(kde.score_samples(np.radians(coords)))
    return densities

df['point_density'] = calculate_density(coords)
```

### **2.3 Расстояние до ближайшего объекта**  
```python
from sklearn.neighbors import BallTree

def distance_to_nearest(target_coords, reference_coords):
    """Расстояние до ближайшей точки в reference_coords"""
    tree = BallTree(np.radians(reference_coords), metric='haversine')
    distances, _ = tree.query(np.radians(target_coords), k=1)
    return distances * 6371  # преобразование в километры

# Пример: расстояние до ближайшей станции метро
df['dist_to_metro'] = distance_to_nearest(coords, metro_coords)
```

---

## **🔴 Экспертный уровень: Пространственные структуры и H3**  

### **3.1 Геошестиугольники H3 (Uber)**  
```python
import h3

def add_h3_features(df, resolution=9):
    """Добавляет H3-индексы и соседние ячейки"""
    df['h3_index'] = df.apply(
        lambda row: h3.geo_to_h3(row['lat'], row['lon'], resolution), 
        axis=1
    )
    df['h3_neighbors'] = df['h3_index'].apply(
        lambda x: h3.k_ring(x, 1)  # соседи первого порядка
    )
    return df
```

### **3.2 Изохроны доступности**  
```python
import osmnx as ox
from networkx import shortest_path_length

def calculate_isochrone(lat, lon, travel_time=15, speed=4.5):
    """Зона доступности за указанное время (пешком)"""
    G = ox.graph_from_point((lat, lon), network_type='walk', dist=2000)
    center_node = ox.distance.nearest_nodes(G, lon, lat)
    
    isochrone = set()
    for node in G.nodes:
        try:
            path_length = shortest_path_length(G, center_node, node, weight='length')
            if path_length <= travel_time * 60 * speed:  # метры
                isochrone.add(node)
        except:
            continue
    return isochrone
```

### **3.3 Пространственные автокорреляции (Moran's I)**  
```python
from libpysal.weights import DistanceBand
from esda.moran import Moran

def calculate_spatial_autocorrelation(values, coords):
    """Расчет пространственной автокорреляции"""
    w = DistanceBand(coords, threshold=0.02)  # порог в градусах
    moran = Moran(values, w)
    return moran.I

# Пример для цен на недвижимость
autocorr = calculate_spatial_autocorrelation(df['price'], coords)
print(f"Индекс Морана: {autocorr:.3f}")
```

---

## **Практические кейсы**  

### **Оптимизация логистики**  
```python
# 1. Кластеризация заказов
# 2. Построение выпуклых оболочек для кластеров
# 3. Расчет оптимальных маршрутов внутри кластеров
```

### **Анализ городской инфраструктуры**  
```python
# 1. Расчет зон пешеходной доступности (изохроны)
# 2. Выявление "пустышек" - районов без инфраструктуры
# 3. Сравнение плотности объектов в H3-ячейках
```

---

## **Заключение**  
**Ключевые принципы работы с геоданными:**  
1. **Используйте сферические метрики** (Haversine вместо Евклида)  
2. **Учитывайте масштаб** (оптимальный размер кластеров/ячеек)  
3. **Комбинируйте техники** (кластеризация + плотность + пространственные метрики)  

**Полезные инструменты:**  
- `geopy` - геокодирование и расчет расстояний  
- `h3-py` - геошестиугольная индексация  
- `osmnx` - работа с уличными сетями  
- `libpysal` - пространственная статистика  

> **"Пространственные данные - это не просто точки на карте, а отражение сложных систем. Ваша задача - сделать эти системы понятными для алгоритмов."**
