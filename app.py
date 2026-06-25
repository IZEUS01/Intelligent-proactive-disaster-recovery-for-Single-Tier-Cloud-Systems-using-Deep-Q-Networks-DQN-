import time
import threading
import os
from flask import Flask, jsonify, request, render_template
from rl_env import CloudDisasterRecoveryEnv
from dqn_agent import DQNAgent

app = Flask(__name__)

env = CloudDisasterRecoveryEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)
if not agent.load("model/dqn_cloud_dr.pth"):
    print("Warning: Pre-trained model not found. Using untrained agent.")

current_state, _ = env.reset()
is_paused = False
last_action = 0
last_reward = 0
last_q_values = [0.0, 0.0, 0.0]
last_info = {'is_failure': False, 'backup_triggered': False}

def simulation_loop():
    global current_state, last_action, last_reward, last_q_values, last_info, is_paused
    while True:
        if not is_paused:
            last_q_values = agent.get_q_values(current_state)
            action = agent.select_action(current_state, evaluate=True)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            current_state = next_state
            last_action = action
            last_reward = reward
            last_info = info
            
            if done:
                current_state, _ = env.reset()
                
        time.sleep(1.0) # Step every 1 second

threading.Thread(target=simulation_loop, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    return jsonify({
        "features": {
            "cpu_util": env.cpu_util,
            "mem_util": env.mem_util,
            "disk_io": env.disk_io,
            "net_latency": env.net_latency,
            "error_rate": env.error_rate,
            "backup_age": env.backup_age,
            "sla_breach_risk": env.sla_breach_risk,
            "workload_trend": env.workload_trend
        },
        "action": last_action,
        "q_values": last_q_values,
        "reward": last_reward,
        "cumulative_reward": env.cumulative_reward,
        "rto": env.rto,
        "rpo": env.rpo,
        "sla_breaches": env.sla_breaches,
        "is_failure": last_info.get('is_failure', False),
        "backup_triggered": last_info.get('backup_triggered', False),
        "backup_log": list(env.backup_log),
        "mode": "non-stationary",
        "stationary_trace": list(env.stationary_trace),
        "nonstationary_trace": list(env.nonstationary_trace)
    })

@app.route('/api/control', methods=['POST'])
def control():
    global is_paused, current_state
    data = request.json
    command = data.get('action')
    
    if command == 'pause':
        is_paused = True
    elif command == 'play':
        is_paused = False
    elif command == 'reset':
        current_state, _ = env.reset()
    elif command == 'force_failure':
        env.cpu_util = 100
        env.error_rate = 80
        env.net_latency = 2000
    elif command == 'traffic_spike':
        env.cpu_util = min(100, env.cpu_util + 40)
        env.mem_util = min(100, env.mem_util + 40)
        env.disk_io += 300
    elif command == 'reduce_traffic':
        env.cpu_util = max(0, env.cpu_util - 30)
        env.mem_util = max(0, env.mem_util - 30)
        env.disk_io = max(0, env.disk_io - 200)
        env.net_latency = max(1, env.net_latency - 100)
        env.workload_trend = max(-1.0, env.workload_trend - 0.4)
    
    return jsonify({"status": "success", "command": command})

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    app.run(debug=True, port=5000, use_reloader=False)
