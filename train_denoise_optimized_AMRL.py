import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import matplotlib.pyplot as plt
import copy
import itertools
import warnings
import os
import json

warnings.filterwarnings("ignore")


# ==========================================
# 1. 无人机辅助物联网环境 (MDP)
# ==========================================
class UAV_IoT_Env:
    def __init__(self, K=5, area_size=1000.0):
        self.K = K
        self.area_size = area_size
        self.H = 100.0
        self.V = 20.0
        self.T_stab = 2.0
        self.B = 1e6
        self.P_tx = 0.1
        self.G0 = 1e-4
        self.N0 = 1e-13
        self.Gamma_dB = 8.0
        self.Gamma = 10 ** (self.Gamma_dB / 10.0)
        self.alpha = 2.5
        self.T_sync = 0.5

        self.action_space_n = self.K + 1
        self.state_dim = 3 + (self.K + 1) * 2 + (self.K + 1) + self.K * 3

        self.base_station = np.array([area_size / 2, area_size / 2, 0.0])
        self.device_positions = None
        self.data_arrival_rates = None

        self.uav_pos = None
        self.Q = None
        self.PAoI = None
        self.current_time = 0.0
        self.visited = None
        self.step_count = 0

    def reset_task(self):
        self.device_positions = np.random.uniform(0, self.area_size, size=(self.K, 2))
        self.device_positions = np.hstack([self.device_positions, np.zeros((self.K, 1))])
        self.all_nodes = np.vstack([self.base_station, self.device_positions])
        self.data_arrival_rates = np.random.uniform(5e4, 2e5, size=self.K)

    def reset(self):
        if self.device_positions is None: self.reset_task()
        self.uav_pos = np.copy(self.base_station)
        self.uav_pos[2] = self.H
        self.Q = np.zeros(self.K)
        self.PAoI = np.zeros(self.K)
        self.visited = np.zeros(self.K)
        self.current_time = 0.0
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        norm_pos = self.uav_pos / self.area_size
        norm_Q = self.Q / 1e6
        norm_PAoI = self.PAoI / 100.0
        norm_nodes = self.all_nodes[:, :2].flatten() / self.area_size
        distances = np.linalg.norm(self.all_nodes[:, :2] - self.uav_pos[:2], axis=1) / self.area_size

        return np.concatenate([
            norm_pos, norm_nodes, distances, norm_Q, norm_PAoI, self.visited
        ]).astype(np.float32)

    def get_action_mask(self):
        mask = np.ones(self.action_space_n, dtype=bool)
        if sum(self.visited) < self.K:
            mask[0] = False
        for i in range(self.K):
            if self.visited[i] == 1:
                mask[i + 1] = False
        return mask

    def step(self, action):
        self.step_count += 1
        target_pos = np.copy(self.all_nodes[action])
        target_pos[2] = self.H

        # 物理计算
        dist = np.linalg.norm(self.uav_pos - target_pos)
        T_fly = dist / self.V + self.T_stab if dist > 1e-3 else 0.0

        T_hov = 0.0
        if action != 0:
            k = action - 1
            dist_3d = np.linalg.norm(target_pos - self.all_nodes[action])
            R_k = self.B * math.log2(1 + (self.P_tx * self.G0) / (self.N0 * self.Gamma * (dist_3d ** self.alpha)))
            T_hov = self.T_sync + self.Q[k] / R_k

        tau_t = T_fly + T_hov
        if action != 0: self.visited[action - 1] = 1

        self.current_time += tau_t
        self.Q += self.data_arrival_rates * tau_t
        self.PAoI += tau_t

        if action != 0:
            k = action - 1
            # self.PAoI[k] = tau_t
            self.PAoI[k] = T_hov
            self.Q[k] = 0.0

        self.uav_pos = target_pos

        done = False

        # 【关键修复】：计算距离惩罚，鼓励走最短路
        distance_penalty = dist / 100.0

        if sum(self.visited) == self.K and action == 0:
            reward = 500.0 - np.max(self.PAoI)
            done = True
        else:
            # 加入距离惩罚
            reward = -np.max(self.PAoI) - distance_penalty - 0.05 * T_hov

        # 【救命神药】：全局奖励缩放！
        # 除以 100，防止 MSE Loss 爆炸，同时完美保留轨迹间的好坏分差！
        reward = reward / 100.0

        if self.step_count > self.K + 2: done = True
        return self._get_state(), reward, done, {}


# ==========================================
# 2. 神经网络架构与训练机制 (Actor-Critic AMRL)
# ==========================================
# ==========================================
# 2. 神经网络架构与训练机制
# ==========================================

# ------------------------------------------
# [网络架构 A]：传统 MLP 网络 (供 Standard RL 使用)
# ------------------------------------------
class MLP_ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MLP_ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.actor_out = nn.Linear(512, action_dim)
        self.critic_out = nn.Linear(512, 1)

    def forward(self, x, params=None):
        if params is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            logits = self.actor_out(x)
            value = self.critic_out(x)
        else:
            x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
            x = F.relu(F.linear(x, params['fc2.weight'], params['fc2.bias']))
            logits = F.linear(x, params['actor_out.weight'], params['actor_out.bias'])
            value = F.linear(x, params['critic_out.weight'], params['critic_out.bias'])
        return logits, value.squeeze(-1)


# ------------------------------------------
# [网络架构 B]：高级 Attention 网络 (供 AMRL 专属使用)
# ------------------------------------------
class Attention_ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Attention_ActorCritic, self).__init__()
        self.K = action_dim - 1
        self.hidden_dim = 128

        self.node_embedder = nn.Linear(2, self.hidden_dim)
        self.uav_embedder = nn.Linear(3, self.hidden_dim)

        self.attention_query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attention_key = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.critic_out = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x, params=None):
        # 【修复核心 1】：记录当前输入是不是单步采样（1D）
        is_single = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            is_single = True

        uav_pos = x[:, :3]
        norm_nodes_flat = x[:, 3: 3 + (self.K + 1) * 2]
        node_positions = norm_nodes_flat.view(-1, self.K + 1, 2)

        if params is None:
            node_embeddings = self.node_embedder(node_positions)
            uav_embedding = self.uav_embedder(uav_pos)
        else:
            node_embeddings = F.relu(
                F.linear(node_positions, params['node_embedder.weight'], params['node_embedder.bias']))
            uav_embedding = F.relu(F.linear(uav_pos, params['uav_embedder.weight'], params['uav_embedder.bias']))

        query = self.attention_query(uav_embedding) if params is None else F.linear(uav_embedding,
                                                                                    params['attention_query.weight'],
                                                                                    params['attention_query.bias'])
        keys = self.attention_key(node_embeddings) if params is None else F.linear(node_embeddings,
                                                                                   params['attention_key.weight'],
                                                                                   params['attention_key.bias'])

        attention_scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.hidden_dim)
        logits = attention_scores

        nodes_mean_embed = node_embeddings.mean(dim=1)
        critic_context = torch.cat([nodes_mean_embed, uav_embedding], dim=1)

        if params is None:
            value = self.critic_out(critic_context)
        else:
            # 注意这里的中间变量名是 fc_x
            fc_x = F.relu(F.linear(critic_context, params['critic_out.0.weight'], params['critic_out.0.bias']))
            value = F.linear(fc_x, params['critic_out.2.weight'], params['critic_out.2.bias'])

        # 【修复核心 2】：如果刚才人为加了 Batch 维度，返回前必须挤掉 (squeeze)
        if is_single:
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        return logits, value.squeeze(-1)

# ------------------------------------------
# 公共轨迹收集与 Loss 计算 (带有 Huber 修复防止爆炸)
# ------------------------------------------
def collect_rl_trajectory(env, policy_net, params=None):
    state = env.reset()
    log_probs, rewards, entropies, values = [], [], [], []
    for _ in range(env.K + 2):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        logits, value = policy_net(state_tensor, params)
        mask = torch.tensor(env.get_action_mask(), dtype=torch.bool)
        logits[~mask] = -1e9

        dist = Categorical(logits=logits)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.item())

        log_probs.append(dist.log_prob(action))
        rewards.append(reward)
        entropies.append(dist.entropy())
        values.append(value)
        state = next_state
        if done: break
    return log_probs, rewards, entropies, values


def compute_rl_loss(log_probs, rewards, entropies, values, gamma=0.99):
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

    if discounted_rewards.std() > 1e-6:
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

    log_probs = torch.stack(log_probs)
    values = torch.stack(values)

    advantages = discounted_rewards - values.detach()
    if advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_loss = -(log_probs * advantages).mean()
    # 使用 Huber Loss (smooth_l1) 防止 Standard RL 的损失飙升到十几亿
    critic_loss = F.smooth_l1_loss(values, discounted_rewards)
    entropy_bonus = sum(entropies) / len(entropies)

    return actor_loss + 0.5 * critic_loss - 0.05 * entropy_bonus


# ------------------------------------------
# 算法 1：AMRL 智能体 (使用 Attention 网络)
# ------------------------------------------
class AMRL:  # AMRL, implemented with first-order meta-learning
    def __init__(self, env):
        self.env = env
        # ★ AMRL 专属：使用强大的 Attention 网络
        self.policy = Attention_ActorCritic(env.state_dim, env.action_space_n)
        self.meta_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.inner_lr = 0.01

    def inner_adaptation(self, env, num_trajectories=3):
        params = {name: param for name, param in self.policy.named_parameters()}
        total_loss = 0
        for _ in range(num_trajectories):
            log_probs, rewards, entropies, values = collect_rl_trajectory(env, self.policy, params)
            total_loss += compute_rl_loss(log_probs, rewards, entropies, values)
        total_loss = total_loss / num_trajectories
        # 一阶元学习近似：不保留内环梯度的二阶计算图。
        # create_graph=False 表示外环更新时忽略 Hessian 项，
        # 即近似 ∂theta_i/∂theta ≈ I。
        grads = torch.autograd.grad(total_loss, params.values(), create_graph=False)

        # 注意：这里不能把 fast_params 整体 detach 掉。
        # param - inner_lr * grad.detach() 仍然保留 fast_params 到原始 param 的一阶连接，
        # 因此外层 meta_loss.backward() 可以把 query loss 的梯度回传到 self.policy.parameters()。
        fast_params = {
            name: param - self.inner_lr * grad.detach()
            for (name, param), grad in zip(params.items(), grads)
        }
        return fast_params, total_loss.item()

    def train(self, iterations=500, batch_size=4):
        history_support_loss, history_query_loss = [], []
        for it in range(iterations):
            self.meta_optimizer.zero_grad()
            meta_loss, batch_support_loss = 0.0, 0.0

            for _ in range(batch_size):
                self.env.reset_task()
                fast_params, support_loss = self.inner_adaptation(self.env)
                batch_support_loss += support_loss

                log_probs, rewards, entropies, values = collect_rl_trajectory(self.env, self.policy, fast_params)
                meta_loss += compute_rl_loss(log_probs, rewards, entropies, values)

            meta_loss = meta_loss / batch_size
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.meta_optimizer.step()

            history_support_loss.append(batch_support_loss / batch_size)
            history_query_loss.append(meta_loss.item())
            if (it + 1) % 50 == 0:
                print(f"AMRL Meta Iteration [{it + 1}/{iterations}] - Query Loss: {meta_loss.item():.4f}")
        return history_support_loss, history_query_loss


# ------------------------------------------
# 算法 2：Standard RL 智能体 (保持原状，使用 MLP 网络)
# ------------------------------------------
class Standard_RL:
    def __init__(self, env):
        self.env = env
        # ★ Standard RL 专属：保持原状，使用基础的 MLP 网络
        self.policy = MLP_ActorCritic(env.state_dim, env.action_space_n)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.fine_tune_lr = 0.01

    def train(self, iterations=500, batch_size=4):
        history_loss = []
        for it in range(iterations):
            self.optimizer.zero_grad()
            batch_loss = 0.0
            for _ in range(batch_size):
                self.env.reset_task()
                log_probs, rewards, entropies, values = collect_rl_trajectory(self.env, self.policy, None)
                batch_loss += compute_rl_loss(log_probs, rewards, entropies, values)
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            history_loss.append(batch_loss.item())
            if (it + 1) % 50 == 0:
                print(f"Standard RL Iteration [{it + 1}/{iterations}] - Loss: {batch_loss.item():.4f}")
        return history_loss

    def fine_tune(self, env, num_trajectories=3):
        params = {name: param for name, param in self.policy.named_parameters()}
        total_loss = 0
        for _ in range(num_trajectories):
            log_probs, rewards, entropies, values = collect_rl_trajectory(env, self.policy, params)
            total_loss += compute_rl_loss(log_probs, rewards, entropies, values)
        total_loss /= num_trajectories
        grads = torch.autograd.grad(total_loss, params.values(), create_graph=False)
        return {name: param - self.fine_tune_lr * grad for (name, param), grad in zip(params.items(), grads)}


# ==========================================
# 3. 启发式与理论最优解 Agents
# ==========================================
class RandomAgent:
    def __init__(self): self.name = "Random Walk"

    def prepare_task(self, env): pass

    def select_action(self, env, state): return np.random.choice(np.where(env.get_action_mask())[0])



class MaxPAoIAgent:
    def __init__(self): self.name = "Max-Age First (MAF)"

    def prepare_task(self, env): pass

    def select_action(self, env, state):
        valid = np.where(env.get_action_mask())[0]
        if len(valid) == 1 and valid[0] == 0: return 0
        priorities = [env.PAoI[a - 1] + (env.Q[a - 1] / 1e6) if a != 0 else -1 for a in valid]
        return valid[np.argmax(priorities)]


class OptimalTSPAgent:
    def __init__(self):
        self.name = "Exact Optimal TSP"
        self.path = []
        self.step_idx = 0

    def prepare_task(self, env):
        nodes = list(range(1, env.K + 1))
        best_path = None
        min_cost = float('inf')
        for p in itertools.permutations(nodes):
            cost = 0
            curr = env.base_station[:2]
            for node_idx in p:
                target = env.all_nodes[node_idx][:2]
                dist = np.linalg.norm(curr - target)
                cost += dist / env.V + env.T_stab if dist > 1e-3 else 0.0
                curr = target
            dist = np.linalg.norm(curr - env.base_station[:2])
            cost += dist / env.V + env.T_stab if dist > 1e-3 else 0.0
            if cost < min_cost:
                min_cost = cost
                best_path = list(p) + [0]
        self.path = best_path
        self.step_idx = 0

    def select_action(self, env, state):
        action = self.path[self.step_idx]
        self.step_idx += 1
        return action


class FastTSPAgent:
    def __init__(self):
        self.name = "Fast TSP (Greedy Approximation)"
        self.path = []
        self.step_idx = 0

    def prepare_task(self, env):
        unvisited = list(range(1, env.K + 1))
        curr = env.base_station[:2]
        path = []
        while unvisited:
            distances = [np.linalg.norm(curr - env.all_nodes[idx][:2]) for idx in unvisited]
            best_idx = np.argmin(distances)
            next_node = unvisited[best_idx]
            path.append(next_node)
            curr = env.all_nodes[next_node][:2]
            unvisited.pop(best_idx)
        self.path = path + [0]
        self.step_idx = 0

    def select_action(self, env, state):
        action = self.path[self.step_idx]
        self.step_idx += 1
        return action


class NeuralAgentWrapper:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.fast_params = None

    def prepare_task(self, env):
        if hasattr(self.model, "inner_adaptation"):
            self.fast_params, _ = self.model.inner_adaptation(env)
        else:
            self.fast_params = self.model.fine_tune(env)

    def select_action(self, env, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = self.model.policy(state_tensor, self.fast_params)
            mask = torch.tensor(env.get_action_mask(), dtype=torch.bool)
            logits[~mask] = -1e9
        return torch.argmax(logits).item()


# ==========================================
# 4. 评估与可视化 (修改为仅保存图片)
# ==========================================
def evaluate_agents(agents, env_proto, num_episodes=50):
    results = {agent.name: [] for agent in agents}
    np.random.seed(2024)

    for ep in range(num_episodes):
        env_test = copy.deepcopy(env_proto)
        env_test.reset_task()
        for agent in agents:
            env = copy.deepcopy(env_test)
            agent.prepare_task(env)
            state = env.reset()
            while True:
                action = agent.select_action(env, state)
                state, _, done, _ = env.step(action)
                if done: break
            results[agent.name].append(np.max(env.PAoI))
    return {k: np.mean(v) for k, v in results.items()}



def plot_paoi_sawtooth(agent, env_proto, save_dir):
    env = copy.deepcopy(env_proto)
    env.reset_task()
    agent.prepare_task(env)
    state = env.reset()

    time_steps = [0.0]
    paoi_history = [env.PAoI.copy()]

    while True:
        action = agent.select_action(env, state)
        state, _, done, _ = env.step(action)
        time_steps.append(env.current_time)
        paoi_history.append(env.PAoI.copy())
        if done: break

    paoi_history = np.array(paoi_history)
    plt.figure(figsize=(10, 5))
    for i in range(env.K):
        plt.plot(time_steps, paoi_history[:, i], marker='o', markersize=4, label=f'User {i + 1}')
    plt.title(f'PAoI Evolution (Sawtooth Pattern) - {agent.name} (K={env.K})', fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('PAoI (seconds)')
    if env.K <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # 替换 show() 为 savefig()，并处理文件名中的特殊字符
    safe_agent_name = agent.name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(save_dir, f'paoi_sawtooth_{safe_agent_name}_K{env.K}.png'), dpi=300)
    plt.close()


def plot_comprehensive_comparison(agents, env_proto, mean_results, save_dir):
    np.random.seed(42)
    env_test = copy.deepcopy(env_proto)
    env_test.reset_task()

    plt.style.use('seaborn-v0_8-whitegrid')

    # 图1：无人机轨迹对比图
    fig = plt.figure(figsize=(18, 12))
    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']

    for idx, agent in enumerate(agents):
        env = copy.deepcopy(env_test)
        agent.prepare_task(env)
        state = env.reset()

        trajectory = [env.uav_pos[:2].copy()]
        while True:
            action = agent.select_action(env, state)
            state, _, done, _ = env.step(action)
            trajectory.append(env.uav_pos[:2].copy())
            if done: break
        trajectory = np.array(trajectory)

        ax = fig.add_subplot(2, 3, idx + 1)
        devices_x, devices_y = env.all_nodes[1:, 0], env.all_nodes[1:, 1]
        bs_x, bs_y = env.all_nodes[0, 0], env.all_nodes[0, 1]
        ax.scatter(devices_x, devices_y, c='#1f77b4', s=80, edgecolors='k', zorder=5)
        ax.scatter(bs_x, bs_y, c='#d62728', marker='*', s=200, edgecolors='k', zorder=5)

        if env.K <= 12:
            for i in range(env.K): ax.annotate(f'v{i + 1}', (devices_x[i] + 15, devices_y[i] + 15), fontsize=10)

        ax.plot(trajectory[:, 0], trajectory[:, 1], color=colors[idx % len(colors)], linestyle='--', alpha=0.5,
                zorder=3)
        for i in range(len(trajectory) - 1):
            dx, dy = trajectory[i + 1, 0] - trajectory[i, 0], trajectory[i + 1, 1] - trajectory[i, 1]
            if np.hypot(dx, dy) > 1e-3:
                ax.arrow(trajectory[i, 0], trajectory[i, 1], dx, dy, head_width=20, head_length=25,
                         length_includes_head=True, color=colors[idx % len(colors)], alpha=0.8, zorder=4)

        ax.set_title(f"{agent.name}\nLocal Max PAoI: {np.max(env.PAoI):.2f}s", fontsize=12, fontweight='bold')
        ax.axis('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'trajectories_K{env_proto.K}.png'), dpi=300)
    plt.close()

    # 图2：柱状图
    plt.figure(figsize=(10, 6))
    names = list(mean_results.keys())
    scores = list(mean_results.values())
    bars = plt.bar(names, scores, color=colors[:len(names)], alpha=0.8, edgecolor='k')
    plt.title(f'Performance Comparison over 50 Random Topologies (K={env_proto.K})', fontsize=14, fontweight='bold')
    plt.ylabel('Average Maximum PAoI (Seconds)', fontsize=12)
    plt.xticks(rotation=25, ha='right')

    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{bar.get_height():.1f}s', ha='center',
                 va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'bar_chart_comparison_K{env_proto.K}.png'), dpi=300)
    plt.close()


# ==========================================
# 5. 主程序入口 (添加文件夹自动创建)
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(2024)
    np.random.seed(2024)

    # 创建统一保存结果的文件夹
    SAVE_DIR = "results_K_scenarios"
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_ITERATIONS = 8000
    K_scenarios = [4,8,12,20]

    all_scenarios_data = {}

    for K in K_scenarios:
        print("=" * 60)
        print(f" 开始评估场景: {K} 个节点 ")
        print("=" * 60)

        env = UAV_IoT_Env(K=K)

        print("\n1. Training Standard RL Agent...")
        standard_rl = Standard_RL(env)
        standard_rl.train(iterations=TRAIN_ITERATIONS, batch_size=12)

        print("\n2. Training AMRL Agent...")
        amrl = AMRL(env)
        amrl.train(iterations=TRAIN_ITERATIONS, batch_size=12)

        print("\n3. Setting up Agents for Evaluation...")
        agents = [
            RandomAgent(),
            MaxPAoIAgent(),
            NeuralAgentWrapper(standard_rl, "Standard RL (Fine-tuned)"),
            NeuralAgentWrapper(amrl, "AMRL")
        ]

        if env.K <= 8:
            agents.append(OptimalTSPAgent())
        else:
            print(f"--> K={env.K} 太大，TSP将采用 Fast Greedy Approximation 算法。")
            agents.append(FastTSPAgent())

        print(f"\n--> 自动保存锯齿状 PAoI 变化图到 {SAVE_DIR} ...")
        # 传入 save_dir
        plot_paoi_sawtooth(agents[-2], env, SAVE_DIR)

        print("\n4. Running 50 Episodes Evaluation...")
        mean_results = evaluate_agents(agents, env, num_episodes=50)

        print("\n--- Final Performance Summary ---")
        for name, score in mean_results.items():
            print(f"{name:<35} | Average Max PAoI: {score:.2f} seconds")

        print(f"\n5. 自动保存轨迹图与柱状图到 {SAVE_DIR} ...")
        # 传入 save_dir
        plot_comprehensive_comparison(agents, env, mean_results, SAVE_DIR)

        clean_mean_results = {name: float(score) for name, score in mean_results.items()}

        all_scenarios_data[f"K_{K}"] = {
            "node_count": K,
            "evaluation_results": clean_mean_results
        }

    # 数据同样保存在文件夹下
    save_path = os.path.join(SAVE_DIR, "uav_routing_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_scenarios_data, f, indent=4, ensure_ascii=False)

    print("=" * 60)
    print(f"🎉 所有实验运行完毕！全部图片和 JSON 数据已静默保存至文件夹: {SAVE_DIR}")
    print("=" * 60)