# `dqn_agent.py` — Detailed Explanation

This file contains the actual **Artificial Intelligence** — a Deep Q-Network (DQN). Think of it as a student that starts knowing nothing, but gradually learns to predict server crashes by trial and error.

---

## Part 1: Imports (Lines 1–7)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
```

- **`torch` (PyTorch)** — The engine that powers the neural network. It's a library for building and training AI brains.
- **`torch.nn`** — Contains ready-made building blocks for neural networks (layers, activation functions).
- **`torch.optim`** — Contains optimizers (the math that adjusts the brain to learn).
- **`deque`** — A list with a max size (old memories get automatically deleted when it's full).

---

## Part 2: Replay Buffer — The Memory Bank (Lines 9–22)

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
```

**What's happening:**
Think of this as a **diary** the AI writes in after every step:

> *"The server had [state]. I pressed [action]. I got [reward]. Then it became [next_state]. Game over? [done]."*

- **`push()`** — Writes one new diary entry.
- **`sample(batch_size=64)`** — Randomly flips open 64 old diary entries to study. **Why random?** If the AI only studied its most recent memories, it would forget old lessons. Random sampling keeps the learning balanced.
- **`capacity=10000`** — The diary only holds 10,000 entries. When it's full, the oldest memory gets thrown out automatically.
- **`__len__()`** — Returns how many memories are stored so far.

---

## Part 3: DQN Network — The Brain Structure (Lines 24–38)

```python
class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),    # 8 inputs  → 128 neurons
            nn.ReLU(),                     # Activation function
            nn.Linear(128, 128),           # 128 → 128 neurons
            nn.ReLU(),
            nn.Linear(128, 64),            # 128 → 64 neurons
            nn.ReLU(),
            nn.Linear(64, action_dim)      # 64 → 3 outputs
        )

    def forward(self, x):
        return self.fc(x)
```

**What's happening:**
This is the **physical structure of the brain** — layers of artificial neurons connected together:

```
INPUT (8 numbers)          HIDDEN LAYERS              OUTPUT (3 numbers)
┌──────────────┐     ┌─────┐    ┌─────┐    ┌────┐    ┌──────────────────┐
│ cpu_util     │     │     │    │     │    │    │    │ Score for NORMAL │
│ mem_util     │     │ 128 │    │ 128 │    │ 64 │    │ Score for WARNING│
│ disk_io      │────►│nodes│───►│nodes│───►│nodes│───►│ Score for CRITICAL│
│ net_latency  │     │     │    │     │    │    │    └──────────────────┘
│ error_rate   │     └─────┘    └─────┘    └────┘
│ backup_age   │
│ sla_risk     │
│ workload_trend│
└──────────────┘
```

- **`nn.Linear(8, 128)`** — Connects 8 inputs to 128 neurons. Each connection has a "weight" (a number) that gets adjusted during training.
- **`nn.ReLU()`** — A filter that says "if the number is negative, make it zero." This simple trick lets the brain learn complex patterns instead of just straight lines.
- **`forward(x)`** — "Given these 8 health numbers, push them through all the layers and give me 3 output scores."

The 3 output numbers are **Q-values** — the AI's prediction of "how much total future reward will I get if I press this button?" The AI always picks the button with the **highest Q-value**.

---

## Part 4: DQN Agent — The Decision Maker (Lines 40–66)

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim,
                 lr=5e-4,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay=0.992):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )

        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=20000)
```

**Key parameters explained:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `lr` (learning rate) | 0.0005 | How big each learning step is. Too big = chaotic. Too small = learns too slowly. |
| `gamma` | 0.99 | How much the AI cares about **future** rewards vs **immediate** rewards. 0.99 = "I care a LOT about the future" |
| `epsilon_start` | 1.0 | Start 100% random (pure exploration) |
| `epsilon_end` | 0.05 | End at 5% random (mostly smart, occasional experiment) |
| `epsilon_decay` | 0.992 | Each episode, randomness shrinks by 0.8% |

**Two brains? Why?**
- **`policy_net`** — The "student" brain. Updated every single step. Makes the decisions.
- **`target_net`** — The "teacher" brain. A frozen copy that only updates every 10 episodes. It provides stable "correct answers" for the student to learn from. Without it, the student would be chasing a moving target and never stabilize.

**Device selection:** The code checks if you have a GPU (NVIDIA → `cuda`, Apple M-chip → `mps`). If not, it uses regular CPU. GPUs make training ~10x faster.

---

## Part 5: Choosing an Action (Lines 68–75)

```python
def select_action(self, state, evaluate=False):
    if not evaluate and random.random() < self.epsilon:
        return random.randrange(self.action_dim)    # Random button!
        
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.policy_net(state_t)
        return q_values.argmax().item()              # Smartest button!
```

**What's happening:**
This is the **explore vs exploit** dilemma:

- **Early training** (`epsilon = 1.0`): The AI presses **random buttons** 100% of the time. It's like a baby pressing everything to see what happens.
- **Late training** (`epsilon = 0.05`): The AI uses its brain 95% of the time, only pressing random buttons 5% of the time (to occasionally discover new strategies).
- **During evaluation** (`evaluate=True`): Epsilon is ignored. The AI **always** uses its brain. No randomness.

`q_values.argmax()` means "pick the action with the highest predicted score."

---

## Part 6: Getting Q-Values (Lines 77–80)

```python
def get_q_values(self, state):
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.policy_net(state_t).squeeze(0).cpu().numpy().tolist()
```

**What's happening:**
This is used by the **dashboard** (`app.py`) to display the AI's thinking. It returns all 3 Q-values (one for each action) so the dashboard can show:

> "NORMAL: 2.3 | WARNING: 5.1 | CRITICAL: 8.7"

The `torch.no_grad()` wrapper means "don't track math for learning — I'm just peeking at the answer."

---

## Part 7: The Learning Step — `train_step()` (Lines 82–108)

```python
def train_step(self, batch_size=64):
    if len(self.memory) < batch_size:
        return 0.0                              # Not enough memories yet
        
    states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
    
    states = torch.FloatTensor(states).to(self.device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
    next_states = torch.FloatTensor(next_states).to(self.device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
    
    # Current Q: "What did I THINK would happen?"
    q_values = self.policy_net(states).gather(1, actions)
    
    # Target Q: "What ACTUALLY happened + best future estimate?"
    with torch.no_grad():
        next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
    loss = nn.MSELoss()(q_values, target_q_values)
    
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    return loss.item()
```

**What's happening — step by step:**

This is the **actual learning**. Think of it as a student checking their exam answers:

1. **Grab 64 random memories** from the diary.

2. **"What did I predict?"** — The student brain (`policy_net`) looks at each old state and says "I think pressing that button was worth X points."

3. **"What actually happened?"** — The teacher brain (`target_net`) calculates the **real** value:
   ```
   real_value = reward_I_got + 0.99 × best_future_reward
   ```
   - The `0.99` is gamma — it means "future rewards are almost as important as immediate ones."
   - `(1 - dones)` means "if the game ended, there IS no future reward."

4. **"How wrong was I?"** — The **loss** is the difference between prediction and reality, squared:
   ```
   loss = (predicted - actual)²
   ```
   If the AI predicted +2 but the answer was -10, loss = (2 - (-10))² = 144. That's a big error!

5. **"Adjust my brain"** — The three magic lines:
   - `zero_grad()` — Clear old adjustments
   - `loss.backward()` — Calculate which weights caused the error (backpropagation)
   - `optimizer.step()` — Nudge those weights slightly to reduce the error

Over thousands of steps, the loss shrinks and the AI's predictions become accurate.

---

## Part 8: Epsilon Decay & Target Update (Lines 110–116)

```python
def decay_epsilon(self):
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

def update_target_network(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())
```

- **`decay_epsilon()`** — Called once per episode. Multiplies epsilon by 0.992, making the AI slightly less random each time. After ~150 episodes: `1.0 × 0.992^150 ≈ 0.30` (still 30% random). By ~500 episodes, it's close to 5%.
- **`update_target_network()`** — Copies the student brain into the teacher brain. Done every 10 episodes. This keeps the "correct answers" fresh but not too jumpy.

---

## Part 9: Save & Load the Brain (Lines 118–128)

```python
def save(self, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(self.policy_net.state_dict(), filepath)

def load(self, filepath):
    if os.path.exists(filepath):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = self.epsilon_min   # Fully trained, pure exploit
        return True
    return False
```

**What's happening:**
- **`save()`** — Saves all the learned weights (the "knowledge") to a file (`model/dqn_cloud_dr.pth`). This is the trained brain.
- **`load()`** — Loads a previously saved brain. When loaded:
  - Both policy and target networks get the saved weights.
  - Epsilon is set to minimum (0.05) because a loaded brain is already trained — no need for random exploration.
  - Returns `True` if the file was found, `False` if not (so `app.py` can print a warning).

---

## Summary: How the AI Learns

```
Episode 1 (epsilon=1.0 — 100% random):
  "I randomly pressed NORMAL during a crash... got -10 points. Ouch."
  → Stores this in memory

Episode 50 (epsilon=0.67 — 67% random):
  "Hmm, I've seen high error_rate before. Let me try CRITICAL..."
  → Got +1 point! Stores this in memory
  → train_step: "CRITICAL during high errors = good. Adjusting weights..."

Episode 150 (epsilon=0.30 — 30% random):
  "Error rate is climbing + backup_age is high = CRITICAL!"
  → Consistently getting +1. Loss is shrinking.

Episode 500 (epsilon=0.05 — 95% smart):
  "I can now predict crashes 2-3 steps in advance by watching
   error_rate trends + backup_age together."
  → Trained brain saved to model/dqn_cloud_dr.pth
```
