# Reinforcement Learning: Основы

## 🟢 Основы (Basic Level)

### Введение в RL

**Reinforcement Learning (Обучение с подкреплением)** - область машинного обучения, где агент учится принимать решения, взаимодействуя со средой и получая награды.

**Основные компоненты:**
- **Agent (Агент)**: принимает решения
- **Environment (Среда)**: мир, в котором действует агент
- **State (Состояние)**: текущая ситуация в среде
- **Action (Действие)**: что может сделать агент
- **Reward (Награда)**: обратная связь от среды

```python
import gym
import numpy as np

# Создание простой среды CartPole
env = gym.make('CartPole-v1')

# Сброс среды в начальное состояние
state = env.reset()
print(f"Начальное состояние: {state}")
print(f"Пространство действий: {env.action_space}")  # Discrete(2): left or right
print(f"Пространство состояний: {env.observation_space}")  # Box(4,): позиция, скорость, угол, угловая скорость

# Случайное взаимодействие
total_reward = 0
for step in range(100):
    action = env.action_space.sample()  # Случайное действие
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    
    if done:
        print(f"Эпизод завершен на шаге {step + 1}, награда: {total_reward}")
        break

env.close()
```

### Основные концепции

**Markov Decision Process (MDP):**
- **States** S: множество всех возможных состояний
- **Actions** A: множество всех возможных действий
- **Transition** P(s'|s,a): вероятность перехода в s' из s при действии a
- **Reward** R(s,a,s'): награда за переход
- **Discount factor** γ ∈ [0,1]: важность будущих наград

**Return (Возврат):**
```
G_t = r_t + γ*r_(t+1) + γ²*r_(t+2) + ... = Σ γ^k * r_(t+k)
```

**Policy (Политика):**
π(a|s) - вероятность выбрать действие a в состоянии s

**Value Function (Функция ценности):**
V^π(s) - ожидаемый return, начиная из состояния s и следуя политике π

```python
# Пример: простая политика для CartPole
def simple_policy(state):
    """Простая эвристика: толкать в сторону, куда падает шест"""
    angle = state[2]
    if angle < 0:
        return 0  # Налево
    else:
        return 1  # Направо

# Тестирование политики
env = gym.make('CartPole-v1')
state = env.reset()
total_reward = 0

for _ in range(200):
    action = simple_policy(state)
    state, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        break

print(f"Общая награда: {total_reward}")
env.close()
```

## 🟡 Средний уровень (Intermediate Level)

### Q-Learning

**Q-функция:** Q(s,a) - ожидаемый return при выполнении действия a в состоянии s

**Bellman уравнение:**
```
Q(s,a) = R(s,a) + γ * max_a' Q(s',a')
```

**Q-Learning алгоритм:**
```
Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
```
где α - learning rate

```python
import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, 
                 discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: дискретизируем непрерывное пространство состояний
        self.q_table = {}
        
    def discretize_state(self, state):
        """Преобразование непрерывных состояний в дискретные"""
        # Для CartPole: [position, velocity, angle, angular_velocity]
        bins = [
            np.linspace(-4.8, 4.8, 10),  # position
            np.linspace(-4, 4, 10),       # velocity
            np.linspace(-0.418, 0.418, 10),  # angle
            np.linspace(-4, 4, 10)        # angular_velocity
        ]
        discrete_state = tuple([np.digitize(state[i], bins[i]) for i in range(len(state))])
        return discrete_state
    
    def get_action(self, state):
        """ε-greedy политика"""
        discrete_state = self.discretize_state(state)
        
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Exploitation
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[discrete_state])
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.action_size)
        
        # Q-learning update rule
        current_q = self.q_table[discrete_state][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[discrete_next_state])
        
        self.q_table[discrete_state][action] += self.learning_rate * (target_q - current_q)
        
        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Обучение Q-Learning агента
env = gym.make('CartPole-v1')
agent = QLearningAgent(state_size=4, action_size=2)

num_episodes = 1000
rewards_history = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(500):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    rewards_history.append(total_reward)
    
    if episode % 100 == 0:
        avg_reward = np.mean(rewards_history[-100:])
        print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

env.close()
```

### Deep Q-Network (DQN)

Использование нейросети для аппроксимации Q-функции.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, 
                 discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, batch_size=64, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def get_action(self, state):
        """ε-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on batch from memory"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
        
        # Loss and optimization
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.load_state_dict(self.model.state_dict())

# Обучение DQN агента
env = gym.make('CartPole-v1')
agent = DQNAgent(state_size=4, action_size=2)

num_episodes = 500
target_update_freq = 10

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(500):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Модифицированная награда для CartPole
        reward = reward if not done else -10
        
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Update target network periodically
    if episode % target_update_freq == 0:
        agent.update_target_model()
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

env.close()
```

## 🔴 Продвинутый уровень (Expert Level)

### Policy Gradient

Вместо Q-функции, напрямую оптимизируем политику π(a|s).

**REINFORCE алгоритм:**
```
∇_θ J(θ) = E_π[∇_θ log π(a|s) * G_t]
```

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99):
        self.discount_factor = discount_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.saved_log_probs = []
        self.rewards = []
        
    def get_action(self, state):
        """Sample action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        self.saved_log_probs.append(action_dist.log_prob(action))
        return action.item()
    
    def update(self):
        """Policy gradient update"""
        returns = []
        G = 0
        
        # Calculate returns (backward)
        for r in reversed(self.rewards):
            G = r + self.discount_factor * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize
        
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.saved_log_probs = []
        self.rewards = []

# Обучение REINFORCE агента
env = gym.make('CartPole-v1')
agent = REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.01)

num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    
    for step in range(500):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.rewards.append(reward)
        state = next_state
        
        if done:
            break
    
    agent.update()
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Steps: {step + 1}")

env.close()
```

### Actor-Critic

Комбинация Policy Gradient (Actor) и Value Function (Critic).

```python
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Linear(state_size, hidden_size)
        
        # Actor head
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.shared(x))
        policy = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value

# Полный код A2C доступен в практических задачах
```

## Ссылки

- [Sutton & Barto: RL Book](http://incompleteideas.net/book/the-book.html)
- [DQN Paper (2015)](https://www.nature.com/articles/nature14236)
- [Policy Gradient Methods](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
- [OpenAI Gym](https://www.gymlibrary.dev/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

## Datasets/Environments

- [CartPole-v1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)
- [MountainCar-v0](https://www.gymlibrary.dev/environments/classic_control/mountain_car/)
- [LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)
- [Atari Games](https://www.gymlibrary.dev/environments/atari/)
