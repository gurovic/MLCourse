# Задачи: Reinforcement Learning

## 🟢 Базовый уровень

### Задача 1: Знакомство с Gym
Создайте агента для CartPole-v1, который выбирает случайные действия. Запустите 100 эпизодов и постройте график средней награды.

### Задача 2: Эвристическая политика
Реализуйте эвристическую политику для MountainCar-v0. Попробуйте разные стратегии и найдите лучшую.

### Задача 3: Дискретизация состояний
Реализуйте функцию дискретизации непрерывного пространства состояний для CartPole. Протестируйте разное количество bins (5, 10, 20).

## 🟡 Средний уровень

### Задача 4: Q-Learning для FrozenLake
Реализуйте Q-Learning для FrozenLake-v1 (дискретная среда). Визуализируйте Q-таблицу после обучения.

### Задача 5: DQN для CartPole
Обучите DQN агента на CartPole-v1. Достигните средней награды > 195 за 100 последовательных эпизодов.

### Задача 6: Experience Replay
Сравните DQN с и без experience replay. Постройте графики обучения для обоих вариантов.

### Задача 7: Target Network
Реализуйте DQN с и без target network. Сравните стабильность обучения.

## 🔴 Продвинутый уровень

### Задача 8: REINFORCE для LunarLander
Реализуйте REINFORCE алгоритм для LunarLander-v2. Используйте baseline для уменьшения дисперсии.

### Задача 9: Actor-Critic (A2C)
Реализуйте A2C алгоритм для CartPole или LunarLander. Сравните с REINFORCE и DQN.

### Задача 10: Hyperparameter Tuning
Проведите поиск гиперпараметров для DQN:
- Learning rate
- Epsilon decay
- Batch size
- Network architecture
Найдите оптимальную конфигурацию для CartPole-v1.

**Environments:**
- CartPole-v1: https://www.gymlibrary.dev/environments/classic_control/cart_pole/
- MountainCar-v0: https://www.gymlibrary.dev/environments/classic_control/mountain_car/
- FrozenLake-v1: https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
- LunarLander-v2: https://www.gymlibrary.dev/environments/box2d/lunar_lander/
