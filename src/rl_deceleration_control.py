import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import src.config as config

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# --- 1. DEFINE SIMULATION ENVIRONMENT ---
class ADASDecelerationEnv:
    """
    Custom lightweight simulation environment for training an ADAS agent.
    State space: [distance_to_hazard (m), current_speed (m/s), hazard_severity (0-1), lane_alignment (0-1)]
    Action space (Discrete):
      - 0: Coast / Maintain Speed (no acceleration, no brake)
      - 1: Light Brake (deceleration of -1.5 m/s^2)
      - 2: Medium Brake (deceleration of -4.0 m/s^2)
      - 3: Heavy / Emergency Brake (deceleration of -7.5 m/s^2)
      - 4: Accelerate (acceleration of +1.2 m/s^2)
    """
    def __init__(self):
        self.dt = 0.1  # 100ms time step
        self.reset()
        
    def reset(self):
        # Randomize initial states to promote generalization
        self.distance = random.uniform(35.0, 50.0)      # Distance to hazard in meters
        self.speed = random.uniform(11.1, 16.7)         # Speed in m/s (approx 40 to 60 km/h)
        self.severity = random.choice([0.3, 0.5, 0.7, 0.85, 0.95]) # Severity of hazard
        self.lane_alignment = random.choice([0.1, 0.4, 0.5, 0.7, 1.0]) # Lane relevance factor
        
        self.steps = 0
        self.done = False
        return self._get_state()
        
    def _get_state(self):
        return np.array([self.distance, self.speed, self.severity, self.lane_alignment], dtype=np.float32)
        
    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {}
            
        self.steps += 1
        
        # Determine acceleration based on action
        if action == 0:    # Coast
            accel = -0.2   # Slight engine drag
        elif action == 1:  # Light Brake
            accel = -1.5
        elif action == 2:  # Medium Brake
            accel = -4.0
        elif action == 3:  # Heavy Brake
            accel = -7.5
        elif action == 4:  # Accelerate
            accel = 1.2
        else:
            raise ValueError(f"Invalid action: {action}")
            
        # Update vehicle physics
        prev_speed = self.speed
        self.speed = max(0.0, min(22.2, self.speed + accel * self.dt)) # Cap at 80 km/h (22.2 m/s)
        
        # Average speed during this step
        avg_speed = (prev_speed + self.speed) / 2.0
        
        # Update distance to hazard
        self.distance -= avg_speed * self.dt
        
        # Determine reward and termination
        reward = 0.0
        
        # 1. Collision / Safe Stop Check
        if self.distance <= 1.5:  # Collision zone boundary
            self.done = True
            if self.speed <= 1.5:  # Stopped safely before hazard (stopped right at the boundary)
                reward = 100.0     # Large safety bonus
            else:                  # Crashed!
                reward = -250.0    # Huge crash penalty
                
        # 2. Safe Stop Before Boundary (ideal range: 1.5m to 5.0m)
        elif self.speed <= 0.1 and 1.5 < self.distance <= 5.0:
            self.done = True
            reward = 100.0         # Safe stop bonus (ideal distance)
            
        # 3. Stopped Too Early (stopped > 5.0m away from hazard)
        elif self.speed <= 0.1 and self.distance > 5.0:
            self.done = True
            # Small penalty to encourage getting closer if alignment is high
            reward = -10.0 * self.lane_alignment
            
        # 4. Time-out / Overshoot Check
        elif self.distance < 0.0:  # Vehicle bypassed the hazard
            self.done = True
            reward = -150.0        # Passed hazard without stopping (danger)
        elif self.steps >= 120:    # Too long
            self.done = True
            reward = 0.0
            
        # 5. Running step rewards
        else:
            # Reward progressing if hazard is far or irrelevant (low lane alignment)
            if self.distance > 25.0 or self.lane_alignment <= 0.4:
                reward = 0.15 * (self.speed / 10.0)  # Tiny reward for keeping speed up
            else:
                # Hazard is close and in our path: penalize high speed
                # Closer hazard + higher speed + higher severity = larger penalty
                speed_hazard_penalty = -2.5 * (self.speed / 10.0) * (30.0 / max(1.5, self.distance)) * self.severity * self.lane_alignment
                reward += speed_hazard_penalty
                
            # Penalize excessive emergency braking to promote comfort
            if action == 3:
                reward -= 1.5
            elif action == 2:
                reward -= 0.3
                
        return self._get_state(), reward, self.done, {}

# --- 2. DEEP Q-NETWORK (DQN) MODEL ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# --- 3. REPLAY BUFFER ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32), 
                np.array(next_state), np.array(done, dtype=np.uint8))
                
    def __len__(self):
        return len(self.buffer)

# --- 4. DQN AGENT ---
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, buffer_capacity=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.990 # Slightly faster decay for 500 episodes
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
            return torch.argmax(q_values).item()
            
    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
        # Get current Q-values
        curr_q = self.q_net(states_t).gather(1, actions_t)
        
        # Get target Q-values
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q
            
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# --- 5. TRAIN AGENT ---
def train_rl_agent(num_episodes=500, batch_size=64):
    env = ADASDecelerationEnv()
    agent = DQNAgent(state_dim=4, action_dim=5)
    
    rewards_history = []
    collisions_history = []
    success_history = []
    
    print("[*] Training Reinforcement Learning Deceleration Agent for ADAS...")
    
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            agent.update(batch_size)
            
        agent.decay_epsilon()
        
        if ep % 10 == 0:
            agent.update_target_network()
            
        rewards_history.append(episode_reward)
        
        # Tracks metrics
        # Collision: distance <= 1.5 and speed > 1.5, or distance < 0.0
        is_collision = (env.distance <= 1.5 and env.speed > 1.5) or (env.distance < 0.0)
        # Success: stopped safely between 0.0m and 5.0m
        is_success = (env.speed <= 1.5 and 0.0 <= env.distance <= 5.0)
        
        collisions_history.append(1.0 if is_collision else 0.0)
        success_history.append(1.0 if is_success else 0.0)
        
        if ep % 50 == 0:
            avg_rew = np.mean(rewards_history[-50:])
            collision_rate = np.mean(collisions_history[-50:]) * 100
            success_rate = np.mean(success_history[-50:]) * 100
            print(f"Episode {ep:03d} | Avg Reward: {avg_rew:6.2f} | Epsilon: {agent.epsilon:.3f} | Success Rate: {success_rate:5.1f}% | Collision Rate: {collision_rate:5.1f}%")
            
    # Save trained model weights
    models_dir = config.MODEL_DIR
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "rl_deceleration_agent.pth")
    torch.save(agent.q_net.state_dict(), model_path)
    print(f"[+] RL Model weights saved successfully at: {model_path}")
    
    # Save training curves plot
    save_training_plot(rewards_history, collisions_history, success_history)
    
    return agent

def save_training_plot(rewards, collisions, successes):
    episodes = range(1, len(rewards) + 1)
    
    # Calculate rolling averages (window size = 30)
    window = 30
    rolling_rew = np.convolve(rewards, np.ones(window)/window, mode='valid')
    rolling_col = np.convolve(collisions, np.ones(window)/window, mode='valid') * 100.0
    rolling_suc = np.convolve(successes, np.ones(window)/window, mode='valid') * 100.0
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Rewards
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, alpha=0.15, color='blue', label='Episode Reward')
    plt.plot(episodes[window-1:], rolling_rew, color='blue', linewidth=2, label=f'Rolling Mean (w={window})')
    plt.title('Agent Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Collision & Success Rates
    plt.subplot(1, 2, 2)
    plt.plot(episodes[window-1:], rolling_suc, color='green', linewidth=2, label='Success Rate (%)')
    plt.plot(episodes[window-1:], rolling_col, color='red', linewidth=2, label='Collision Rate (%)')
    plt.title('Collision and Success Rates over Time')
    plt.xlabel('Episode')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    logs_dir = config.LOG_DIR
    os.makedirs(logs_dir, exist_ok=True)
    plot_path = os.path.join(logs_dir, "rl_training_rewards.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[+] RL training curves saved successfully at: {plot_path}")

# --- 6. RUN SIMULATED TRAJECTORY TEST ---
def test_rl_agent(agent):
    print("\n" + "=" * 60)
    print("RUNNING SIMULATED TEST TRAJECTORY WITH TRAINED RL AGENT")
    print("=" * 60)
    
    # Test Scenario: Start at 60 km/h (16.7 m/s), approaching a concrete barrier (severity=0.95)
    # right in our lane (alignment=1.0) 45 meters ahead.
    env = ADASDecelerationEnv()
    env.distance = 45.0
    env.speed = 16.7
    env.severity = 0.95
    env.lane_alignment = 1.0
    
    state = env._get_state()
    done = False
    
    trajectory_log = []
    
    action_names = {
        0: "Coast/Maintain",
        1: "Light Brake",
        2: "Medium Brake",
        3: "Heavy Brake",
        4: "Accelerate"
    }
    
    print(f"{'Step':<6} | {'Distance (m)':<12} | {'Speed (km/h)':<13} | {'Action Selected':<18} | {'Step Reward':<11}")
    print("-" * 70)
    
    while not done:
        # Epsilon = 0 (completely greedy evaluation)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(agent.q_net(state_t)).item()
            
        prev_dist = env.distance
        prev_speed = env.speed
        
        next_state, reward, done, _ = env.step(action)
        
        print(f"{env.steps:<6} | {prev_dist:<12.2f} | {prev_speed * 3.6:<13.1f} | {action_names[action]:<18} | {reward:<11.2f}")
        
        trajectory_log.append([env.steps, prev_dist, prev_speed * 3.6, action_names[action], reward])
        state = next_state
        
    print("-" * 70)
    # Success is defined as vehicle speed <= 1.5 km/h and distance is between 0m and 5m (stopped safely before colliding)
    success = "SUCCESS" if (env.speed <= 0.4 and 0.0 <= env.distance <= 5.0) else "FAILED / COLLISION"
    print(f"Simulation Finished | Final Distance: {env.distance:.2f}m | Final Speed: {env.speed * 3.6:.1f} km/h | Result: {success}")
    
    # Save CSV trajectory log
    logs_dir = config.LOG_DIR
    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, "rl_test_trajectory.csv")
    with open(csv_path, 'w') as f:
        f.write("Step,Distance_m,Speed_kmh,Action,Reward\n")
        for line in trajectory_log:
            f.write(f"{line[0]},{line[1]:.2f},{line[2]:.2f},{line[3]},{line[4]:.2f}\n")
    print(f"[+] Trajectory CSV log saved successfully at: {csv_path}\n")

if __name__ == "__main__":
    trained_agent = train_rl_agent(num_episodes=500)
    test_rl_agent(trained_agent)
