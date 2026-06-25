"""
train_dqn.py — Train the DQN agent and print RTO / RPO / SLA Compliance results.
"""
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

def train(episodes=TRAIN_EPISODES):
    env = CloudDisasterRecoveryEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    batch_size = 64
    target_update = 10
    
    rewards_history = []
    loss_history = []
    
    print(f"\n{'='*60}")
    print(f"  DQN TRAINING — {episodes} episodes")
    print(f"{'='*60}")
    
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
        
        # Decay epsilon once per episode (not per step)
        agent.decay_epsilon()
            
        if episode % target_update == 0:
            agent.update_target_network()
            
        rewards_history.append(total_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        loss_history.append(avg_loss)
        
        if (episode + 1) % 50 == 0:
            print(f"  Episode {episode+1:>4}/{episodes} | "
                  f"Reward: {total_reward:>8.1f} | "
                  f"Avg Loss: {avg_loss:>8.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
            
    # Save the model
    os.makedirs('model', exist_ok=True)
    agent.save(MODEL_PATH)
    print(f"\n  Model saved to {MODEL_PATH}")
    
    # Save training graphs
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(10,5))
    plt.plot(rewards_history, alpha=0.3, color='steelblue')
    window = max(1, len(rewards_history) // 20)
    smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    plt.plot(range(len(smoothed)), smoothed, color='cyan', linewidth=2)
    plt.title("DQN Episode Rewards (Cumulative)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(alpha=0.3)
    plt.savefig("results/dqn_rewards.png", dpi=150, bbox_inches='tight')
    
    plt.figure(figsize=(10,5))
    plt.plot(loss_history, alpha=0.3, color='orchid')
    smoothed_loss = np.convolve(loss_history, np.ones(window)/window, mode='valid')
    plt.plot(range(len(smoothed_loss)), smoothed_loss, color='magenta', linewidth=2)
    plt.title("DQN Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.savefig("results/dqn_loss.png", dpi=150, bbox_inches='tight')
    
    # ── Evaluate the trained agent ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DQN EVALUATION — {EVAL_STEPS} steps")
    print(f"{'='*60}")
    
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
    print(f"  {'─'*50}")
    print(f"  SLA Compliance Rate:  {metrics['sla_compliance_rate']:>6.1f} %")
    print(f"  Mean RTO:             {metrics['mean_rto_min']:>6.1f} min")
    print(f"  Mean RPO:             {metrics['mean_rpo_min']:>6.1f} min")
    print(f"  Total SLA Breaches:   {metrics['total_sla_breaches']:>6d}")
    print(f"  Total Failures:       {metrics['total_failures']:>6d}")
    print(f"  Cumulative Reward:    {metrics['cumulative_reward']:>6.1f}")
    print(f"  {'─'*50}")
    
    return agent, rewards_history


if __name__ == "__main__":
    train()
