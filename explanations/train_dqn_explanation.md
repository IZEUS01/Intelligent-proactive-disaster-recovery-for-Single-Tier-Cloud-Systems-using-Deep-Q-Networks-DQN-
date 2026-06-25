# `train_dqn.py` — The Training Camp (Detailed Explanation)

This file is responsible for putting the AI through "school". It repeatedly runs the datacenter simulation so the AI can practice, make mistakes, and learn. After training, it tests the AI and draws charts of its progress.

---

## Part 1: Imports and Settings (Lines 1–16)

```python
import numpy as np
import random
from rl_env import CloudDisasterRecoveryEnv
from dqn_agent import DQNAgent
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
TRAIN_EPISODES = 500
EVAL_STEPS = 2000
MODEL_PATH = "model/dqn_cloud_dr.pth"
```

**What's happening:**
- We import the `CloudDisasterRecoveryEnv` (the video game) and the `DQNAgent` (the player).
- We use `matplotlib` to draw graphs of the AI's learning progress. `matplotlib.use('Agg')` tells it to save the graphs as images instead of trying to pop up a window on your screen.
- **`TRAIN_EPISODES = 500`**: The AI will play through 500 full games to learn.
- **`EVAL_STEPS = 2000`**: After it's done learning, it will take a "final exam" that lasts for 2000 ticks of the clock to see how well it performs.

---

## Part 2: The Training Setup (Lines 18–33)

```python
def train(episodes=TRAIN_EPISODES):
    env = CloudDisasterRecoveryEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    batch_size = 64
    target_update = 10
    
    rewards_history = []
    loss_history = []
```

**What's happening:**
- It creates the environment (`env`) and the agent (`agent`).
- It creates two empty lists: `rewards_history` and `loss_history`. These are the "report cards" used to draw the graphs at the end.
- `target_update = 10` means the "teacher brain" in the AI will update its answers every 10 episodes.

---

## Part 3: The Training Loop (Lines 35–68)

```python
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        episode_loss = []
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step(batch_size)
            if loss > 0:
                episode_loss.append(loss)
                
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
            
        if episode % target_update == 0:
            agent.update_target_network()
```

**What's happening:**
This is the core training cycle. For each of the 500 episodes:
1. **`env.reset()`**: Start a fresh simulation.
2. **`while not done:`**: Keep ticking the clock until the episode reaches its 1000-step limit.
3. **`select_action` & `env.step`**: The AI looks at the server, pushes a button, and the simulation reacts (rewarding or punishing it).
4. **`memory.push`**: The AI writes what just happened into its diary.
5. **`train_step`**: The AI reviews its diary and updates its neural network to be smarter.
6. **`total_reward += reward`**: Keep a running tally of the score for this episode.

At the end of the episode:
- **`decay_epsilon()`**: Make the AI slightly less random and slightly more confident for the next episode.
- **`update_target_network()`**: Every 10 episodes, update the "teacher" brain.

---

## Part 4: Saving the Brain & Drawing Graphs (Lines 70–97)

```python
    os.makedirs('model', exist_ok=True)
    agent.save(MODEL_PATH)
    
    # Save training graphs
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(10,5))
    plt.plot(rewards_history, alpha=0.3, color='steelblue')
    # ... calculates a smooth line ...
    plt.savefig("results/dqn_rewards.png", dpi=150, bbox_inches='tight')
    
    plt.figure(figsize=(10,5))
    plt.plot(loss_history, alpha=0.3, color='orchid')
    # ... calculates a smooth line ...
    plt.savefig("results/dqn_loss.png", dpi=150, bbox_inches='tight')
```

**What's happening:**
- Once all 500 episodes are done, the trained AI is saved to `model/dqn_cloud_dr.pth`. This is incredibly important — without this, the AI would forget everything when the script closes.
- It draws two charts and saves them in the `results/` folder:
  1. **Rewards graph**: You want to see this line going **UP** over time, proving the AI is winning more points.
  2. **Loss graph**: You want to see this line going **DOWN** over time, proving the AI is making fewer mathematical prediction errors.

---

## Part 5: The Final Exam (Lines 99–127)

```python
    random.seed(SEED)
    np.random.seed(SEED)
    eval_env = CloudDisasterRecoveryEnv(max_steps=EVAL_STEPS)
    state, _ = eval_env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, evaluate=True)
        state, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
    
    metrics = eval_env.get_metrics()
    
    print(f"\n  SLA Targets:  RTO ≤ 5.0 min  |  RPO ≤ 15.0 min")
    print(f"  SLA Compliance Rate:  {metrics['sla_compliance_rate']:>6.1f} %")
    # ... prints the rest of the metrics ...
```

**What's happening:**
Now that the AI is fully trained, we want to know exactly how good it is. 
- It creates a special test environment (`eval_env`) that runs for 2,000 steps.
- **`evaluate=True`**: This is crucial. It tells the AI: "Stop exploring. Do not press any random buttons. Use 100% of your brain power."
- It plays the game, collects the final score (SLA Compliance, total failures, total breaches), and prints it out neatly in the terminal for you to read.

---

### In Summary:
`train_dqn.py` is simply a script to automate the teaching process. It loops the game 500 times, tells the AI to learn, saves the final AI file, draws proof that it learned, and gives it a final exam.
