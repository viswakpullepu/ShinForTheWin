"""
Reinforcement Learning Training Pipeline for CSES Arcade RL Hackathon.
Uses exact official game_API.py and schemas.py to train a Deep Q-Network (DQN).
Outputs trained weights to my_bot/rl_weights.json.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Set up paths
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import game_API as api
from schemas import (
    Stage, ActionType, UpgradeType, NextRound,
    Action_PLAY, Action_PICK_UPGRADES
)
from bots.random_bot.bot import Bot as RandomBot

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# -----------------------------------------------------------------------------
# 1. State Extraction
# -----------------------------------------------------------------------------
def extract_state_vector(obs, me, saws_applied=0):
    my_hp = obs.players[me].health
    max_hp = getattr(obs, "max_health", 5)
    my_upg = obs.players[me].upgrades
    others = [i for i, p in enumerate(obs.players) if i != me and p.health > 0]

    opp_hps = [obs.players[i].health for i in others]
    min_opp_hp = min(opp_hps) if opp_hps else 0
    max_opp_hp = max(opp_hps) if opp_hps else 0

    rounds_rem = getattr(obs, "rounds_remaining", 1)
    lives_init = getattr(obs, "lives_init", 1) or 1
    lives_shot = getattr(obs, "lives_shot", 0) or 0
    blanks_init = getattr(obs, "blanks_init", 1) or 1
    blanks_shot = getattr(obs, "blanks_shot", 0) or 0
    ejections = getattr(obs, "ejections", None)
    lives_ej = getattr(ejections, "lives_ejected", 0) if ejections else 0
    blanks_ej = getattr(ejections, "blanks_ejected", 0) if ejections else 0

    live_rem = max(0, lives_init - lives_shot - lives_ej)
    blank_rem = max(0, blanks_init - blanks_shot - blanks_ej)
    total_shells = live_rem + blank_rem

    known_shell = None
    if obs.next_round == NextRound.LIVE:
        known_shell = "LIVE"
    elif obs.next_round == NextRound.BLANK:
        known_shell = "BLANK"
    elif live_rem > 0 and blank_rem == 0:
        known_shell = "LIVE"
    elif blank_rem > 0 and live_rem == 0:
        known_shell = "BLANK"

    is_live = 1.0 if known_shell == "LIVE" else 0.0
    is_blank = 1.0 if known_shell == "BLANK" else 0.0
    is_unknown = 1.0 if known_shell is None else 0.0
    session_no = getattr(obs, "session_no", 0)

    features = [
        my_hp / max_hp if max_hp > 0 else 1.0,
        min_opp_hp / max_hp if max_hp > 0 else 0.0,
        max_opp_hp / max_hp if max_hp > 0 else 0.0,
        live_rem / 16.0,
        blank_rem / 16.0,
        live_rem / total_shells if total_shells > 0 else 0.5,
        is_live,
        is_blank,
        is_unknown,
        1.0 if my_upg.juice > 0 and my_hp < max_hp else 0.0,
        1.0 if my_upg.mag_glass > 0 and is_unknown == 1.0 else 0.0,
        1.0 if my_upg.hand_saw > 0 and session_no >= 1 else 0.0,
        1.0 if my_upg.pool_floatie > 0 and session_no >= 1 else 0.0,
        1.0 if my_upg.coffee > 0 else 0.0,
        saws_applied / 3.0,
        session_no / 5.0,
        len(others) / 3.0
    ]
    return features, known_shell, live_rem, blank_rem, others


def get_valid_actions(obs, me, known_shell, session_no, live_rem, blank_rem, others):
    valid = [5, 6]  # Shoot self, shoot opponent
    my_hp = obs.players[me].health
    max_hp = getattr(obs, "max_health", 5)
    my_upg = obs.players[me].upgrades
    rounds_rem = getattr(obs, "rounds_remaining", 1)

    if my_upg.juice > 0 and my_hp < max_hp:
        valid.append(0)
    if my_upg.mag_glass > 0 and known_shell is None and rounds_rem > 0:
        valid.append(1)
    if my_upg.hand_saw > 0 and session_no >= 1 and (known_shell == "LIVE" or live_rem > blank_rem):
        valid.append(2)
    if my_upg.pool_floatie > 0 and session_no >= 1:
        unskipped = [i for i in others if not obs.players[i].has_pool_floatie_on]
        if unskipped:
            valid.append(3)
    if my_upg.coffee > 0 and rounds_rem > 0 and (known_shell == "LIVE" or live_rem > blank_rem):
        valid.append(4)

    return valid


# -----------------------------------------------------------------------------
# 2. Neural Q-Network
# -----------------------------------------------------------------------------
class QNetwork:
    def __init__(self, input_dim=17, hidden_dim=48, output_dim=7):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden_dim)

        if HAS_NUMPY:
            self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
            self.b1 = np.zeros(hidden_dim, dtype=np.float32)
            self.W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * scale2
            self.b2 = np.zeros(output_dim, dtype=np.float32)
        else:
            self.W1 = [[random.gauss(0, scale1) for _ in range(hidden_dim)] for _ in range(input_dim)]
            self.b1 = [0.0] * hidden_dim
            self.W2 = [[random.gauss(0, scale2) for _ in range(output_dim)] for _ in range(hidden_dim)]
            self.b2 = [0.0] * output_dim

    def forward(self, x):
        if HAS_NUMPY:
            x_arr = np.array(x, dtype=np.float32)
            hidden = np.maximum(0.0, np.dot(x_arr, self.W1) + self.b1)
            q_vals = np.dot(hidden, self.W2) + self.b2
            return q_vals.tolist(), hidden
        else:
            hidden = []
            for j in range(self.hidden_dim):
                val = self.b1[j]
                for i in range(self.input_dim):
                    val += x[i] * self.W1[i][j]
                hidden.append(val if val > 0.0 else 0.0)

            q_vals = []
            for k in range(self.output_dim):
                val = self.b2[k]
                for j in range(self.hidden_dim):
                    val += hidden[j] * self.W2[j][k]
                q_vals.append(val)
            return q_vals, hidden

    def train_batch(self, batch, lr=0.003):
        n = len(batch)
        if n == 0:
            return 0.0

        if HAS_NUMPY:
            X = np.array([b[0] for b in batch], dtype=np.float32)
            actions = [b[1] for b in batch]
            targets = np.array([b[2] for b in batch], dtype=np.float32)

            hidden = np.maximum(0.0, np.dot(X, self.W1) + self.b1)
            Q = np.dot(hidden, self.W2) + self.b2

            pred = np.array([Q[i, actions[i]] for i in range(n)], dtype=np.float32)
            diff = pred - targets
            total_loss = float(np.mean(diff ** 2))

            dQ = np.zeros_like(Q)
            for i in range(n):
                dQ[i, actions[i]] = diff[i] / n

            dW2 = np.dot(hidden.T, dQ)
            db2 = np.sum(dQ, axis=0)

            dhidden = np.dot(dQ, self.W2.T)
            dhidden[hidden <= 0] = 0.0

            dW1 = np.dot(X.T, dhidden)
            db1 = np.sum(dhidden, axis=0)

            np.clip(dW1, -1.0, 1.0, out=dW1)
            np.clip(dW2, -1.0, 1.0, out=dW2)

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            return total_loss
        else:
            total_loss = 0.0
            for state, action_idx, target_val in batch:
                q_vals, hidden = self.forward(state)
                curr_q = q_vals[action_idx]
                delta = curr_q - target_val
                total_loss += delta ** 2

                d_hidden = [0.0] * self.hidden_dim
                for j in range(self.hidden_dim):
                    d_hidden[j] = delta * self.W2[j][action_idx]
                    self.W2[j][action_idx] -= lr * delta * hidden[j]
                self.b2[action_idx] -= lr * delta

                for j in range(self.hidden_dim):
                    if hidden[j] <= 0:
                        d_hidden[j] = 0.0

                for i in range(self.input_dim):
                    for j in range(self.hidden_dim):
                        self.W1[i][j] -= lr * d_hidden[j] * state[i]
                for j in range(self.hidden_dim):
                    self.b1[j] -= lr * d_hidden[j]

            return total_loss / n

    def copy_from(self, source_net):
        if HAS_NUMPY:
            self.W1 = np.copy(source_net.W1)
            self.b1 = np.copy(source_net.b1)
            self.W2 = np.copy(source_net.W2)
            self.b2 = np.copy(source_net.b2)
        else:
            self.W1 = [row[:] for row in source_net.W1]
            self.b1 = source_net.b1[:]
            self.W2 = [row[:] for row in source_net.W2]
            self.b2 = source_net.b2[:]

    def save(self, filepath):
        w1 = self.W1.tolist() if HAS_NUMPY else self.W1
        b1 = self.b1.tolist() if HAS_NUMPY else self.b1
        w2 = self.W2.tolist() if HAS_NUMPY else self.W2
        b2 = self.b2.tolist() if HAS_NUMPY else self.b2

        data = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "W1": w1,
            "b1": b1,
            "W2": w2,
            "b2": b2,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)


# -----------------------------------------------------------------------------
# 3. Experience Replay Buffer
# -----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done, next_valid):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done, next_valid)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)


# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------
def train(episodes=2500, lr=0.003, gamma=0.95):
    print("=" * 72)
    print("=== CSES ARCADE RL HACKATHON: DQN TRAINING PIPELINE ===")
    print("=" * 72)
    print(f"Episodes: {episodes} | LR: {lr} | Discount: {gamma} | NumPy: {HAS_NUMPY}")

    q_net = QNetwork(17, 48, 7)
    target_net = QNetwork(17, 48, 7)
    target_net.copy_from(q_net)

    replay_buffer = ReplayBuffer(20000)
    baseline_bot = RandomBot()

    wins = []
    losses = []
    start_time = time.time()

    for ep in range(1, episodes + 1):
        eps = 0.05 + 0.95 * math.exp(-1.0 * ep / (episodes * 0.4))
        seed = random.randint(1, 10000000)
        num_players = 2 if random.random() < 0.7 else 4

        state = api.initGame(seed, num_players=num_players)
        baseline_bot.init({"seed": seed})

        saws_applied = 0
        last_state = None
        last_action = None

        while not state.game_over:
            p_idx = state.turn

            if state.stage == Stage.PLAY:
                obs = api.play_observation(state)

                if p_idx == 0:
                    state_vec, known_shell, live_rem, blank_rem, others = extract_state_vector(obs, 0, saws_applied)
                    valid_actions = get_valid_actions(obs, 0, known_shell, state.session_no, live_rem, blank_rem, others)

                    # Action selection
                    if random.random() < eps:
                        action_idx = random.choice(valid_actions)
                    else:
                        q_vals, _ = q_net.forward(state_vec)
                        action_idx = max(valid_actions, key=lambda a: q_vals[a])

                    # Build action
                    action = Action_PLAY()
                    action.useUpgrade = None
                    action.target = None

                    step_reward = 0.0
                    target_opp = others[0] if others else 0
                    # Pick lowest HP opponent
                    if others:
                        target_opp = min(others, key=lambda i: state.players[i].health)

                    if action_idx == 0:
                        action.action = ActionType.USE_UPGRADE
                        action.useUpgrade = UpgradeType.JUICE
                        step_reward += 1.5

                    elif action_idx == 1:
                        action.action = ActionType.USE_UPGRADE
                        action.useUpgrade = UpgradeType.MAG_GLASS
                        step_reward += 0.8

                    elif action_idx == 2:
                        action.action = ActionType.USE_UPGRADE
                        action.useUpgrade = UpgradeType.HAND_SAW
                        saws_applied += 1
                        if known_shell == "LIVE":
                            step_reward += 2.0

                    elif action_idx == 3:
                        unskipped = [i for i in others if not state.players[i].has_pool_floatie_on]
                        t = max(unskipped, key=lambda i: state.players[i].health) if unskipped else others[0]
                        action.action = ActionType.USE_UPGRADE
                        action.useUpgrade = UpgradeType.POOL_FLOATIE
                        action.target = t
                        step_reward += 1.8

                    elif action_idx == 4:
                        action.action = ActionType.USE_UPGRADE
                        action.useUpgrade = UpgradeType.COFFEE
                        step_reward += 0.8
                        saws_applied = 0

                    elif action_idx == 5:
                        action.action = ActionType.SHOOT
                        action.target = 0
                        # Peek shell
                        shell_idx = len(state.round_contents) - state.rounds_remaining
                        is_live = state.round_contents[shell_idx]
                        if not is_live:
                            step_reward += 2.5  # Free bonus turn!
                        else:
                            step_reward -= 4.0  # Shot self with live shell!
                        saws_applied = 0

                    else:  # action_idx == 6
                        action.action = ActionType.SHOOT
                        action.target = target_opp
                        shell_idx = len(state.round_contents) - state.rounds_remaining
                        is_live = state.round_contents[shell_idx]
                        if is_live:
                            step_reward += 2.0 * state.damage
                            if state.players[target_opp].health <= state.damage:
                                step_reward += 5.0  # Elimination reward!
                        saws_applied = 0

                    if last_state is not None:
                        replay_buffer.push(last_state, last_action, step_reward, state_vec, False, valid_actions)

                    last_state = state_vec
                    last_action = action_idx

                    # Step
                    api.step_PLAY(state, action)

                else:
                    act = baseline_bot.play(obs)
                    api.step_PLAY(state, act)

            elif state.stage == Stage.PICK_UPGRADE:
                obs = api.upgrade_observation(state)
                if p_idx == 0:
                    action = Action_PICK_UPGRADES()
                    my_hp = obs.players[0].health
                    counts = obs.available_upgrades
                    if my_hp < 5 and counts.juice > 0:
                        action.chosen_upgrade = UpgradeType.JUICE
                    elif counts.mag_glass > 0:
                        action.chosen_upgrade = UpgradeType.MAG_GLASS
                    elif counts.hand_saw > 0 and state.session_no >= 1:
                        action.chosen_upgrade = UpgradeType.HAND_SAW
                    elif counts.pool_floatie > 0 and state.session_no >= 1:
                        action.chosen_upgrade = UpgradeType.POOL_FLOATIE
                    elif counts.coffee > 0:
                        action.chosen_upgrade = UpgradeType.COFFEE
                    else:
                        action.chosen_upgrade = UpgradeType.JUICE
                    api.step_PICK_UPGRADE(state, action)
                else:
                    act = baseline_bot.pick_upgrade(obs)
                    api.step_PICK_UPGRADE(state, act)

        # Match end
        won = (state.players[0].health > 0)
        final_reward = 12.0 if won else -12.0
        wins.append(1 if won else 0)
        if len(wins) > 100:
            wins.pop(0)

        if last_state is not None:
            replay_buffer.push(last_state, last_action, final_reward, last_state, True, [5, 6])

        # Bellman update
        if len(replay_buffer) >= 64:
            batch = replay_buffer.sample(48)
            train_items = []
            for b_s, b_a, b_r, b_ns, b_done, b_nvalid in batch:
                if b_done:
                    target_q = b_r
                else:
                    nq, _ = target_net.forward(b_ns)
                    best_nq = max(nq[a] for a in b_nvalid) if b_nvalid else 0.0
                    target_q = b_r + gamma * best_nq
                train_items.append((b_s, b_a, target_q))

            l = q_net.train_batch(train_items, lr=lr)
            losses.append(l)
            if len(losses) > 100:
                losses.pop(0)

        if ep % 25 == 0:
            target_net.copy_from(q_net)

        if ep % 200 == 0 or ep == episodes:
            w_rate = (sum(wins) / len(wins)) * 100
            avg_loss = (sum(losses) / len(losses)) if losses else 0.0
            el = time.time() - start_time
            print(f"Ep {ep:4d}/{episodes} | Win Rate: {w_rate:>5.1f}% | MSBE Loss: {avg_loss:>6.3f} | Eps: {eps:.3f} | {el:.1f}s")

    # Save weights
    out_path = ROOT / "my_bot" / "rl_weights.json"
    q_net.save(str(out_path))
    print(f"\n[DONE] Saved trained RL weights to {out_path} ({os.path.getsize(str(out_path))/1024:.2f} KB)")


if __name__ == "__main__":
    train(episodes=2500)
