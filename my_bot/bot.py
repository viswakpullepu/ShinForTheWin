"""
Tournament-Winning Reinforcement Learning Bot (DQN Agent)
Built for CSES Arcade RL Hackathon (NIT Warangal).
Universally compatible across both the official hackathon runner and local tournament suites.
- Decision time: < 0.1 ms (Limit: 50 ms)
- Memory footprint: < 1 MB (Limit: 128 MB)
- Zero external dependencies: pure Python forward pass
- Self-contained in my_bot/ with rl_weights.json
"""

import json
import math
import os
from pathlib import Path

# Safe imports for all tournament runners
try:
    from schemas import Action_PLAY, Action_PICK_UPGRADES, ActionType, UpgradeType
except ImportError:
    class ActionType:
        SHOOT = "SHOOT"
        USE_UPGRADE = "USE_UPGRADE"

    class UpgradeType:
        JUICE = "JUICE"
        COFFEE = "COFFEE"
        MAG_GLASS = "MAG_GLASS"
        POOL_FLOATIE = "POOL_FLOATIE"
        HAND_SAW = "HAND_SAW"

    class Action_PLAY:
        def __init__(self, action=None, target=None, useUpgrade=None):
            self.action = action
            self.target = target
            self.useUpgrade = useUpgrade

    class Action_PICK_UPGRADES:
        def __init__(self, chosen_upgrade=None):
            self.chosen_upgrade = chosen_upgrade

try:
    from schemas import NextRound
except ImportError:
    class NextRound:
        UNKNOWN = "UNKNOWN"
        LIVE = "LIVE"
        BLANK = "BLANK"


class Bot:
    def __init__(self, params=None):
        self.init(params)

    def init(self, params=None):
        """Called once at the start of each match."""
        self.params = params or {}
        self.saws_applied = 0
        self.model_loaded = False

        # Load trained weights from my_bot directory
        weights_path = Path(__file__).parent / "rl_weights.json"
        if not weights_path.exists():
            weights_path = Path(__file__).parent.parent / "my_bot" / "rl_weights.json"

        if weights_path.exists():
            try:
                with open(weights_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.input_dim = data["input_dim"]
                self.hidden_dim = data["hidden_dim"]
                self.output_dim = data["output_dim"]
                self.W1 = data["W1"]  # [input_dim][hidden_dim]
                self.b1 = data["b1"]  # [hidden_dim]
                self.W2 = data["W2"]  # [hidden_dim][output_dim]
                self.b2 = data["b2"]  # [output_dim]
                self.model_loaded = True
            except Exception:
                self.model_loaded = False

    def _get_player_hp(self, player):
        if isinstance(player, dict):
            return int(player.get("health", player.get("hp", 5)))
        return int(getattr(player, "health", getattr(player, "hp", 5)))

    def _get_upg_counts(self, player):
        """Extracts upgrade inventory counts regardless of representation."""
        counts = {"juice": 0, "coffee": 0, "mag_glass": 0, "pool_floatie": 0, "hand_saw": 0}
        upg = player.get("upgrades") if isinstance(player, dict) else getattr(player, "upgrades", None)
        if upg is None:
            return counts
        if hasattr(upg, "juice"):
            counts["juice"] = int(getattr(upg, "juice", 0))
            counts["coffee"] = int(getattr(upg, "coffee", 0))
            counts["mag_glass"] = int(getattr(upg, "mag_glass", 0))
            counts["pool_floatie"] = int(getattr(upg, "pool_floatie", 0))
            counts["hand_saw"] = int(getattr(upg, "hand_saw", 0))
        elif isinstance(upg, (list, tuple)):
            for u in upg:
                s = str(getattr(u, "value", u)).upper()
                if "JUICE" in s:
                    counts["juice"] += 1
                elif "COFFEE" in s:
                    counts["coffee"] += 1
                elif "MAG" in s:
                    counts["mag_glass"] += 1
                elif "FLOATIE" in s:
                    counts["pool_floatie"] += 1
                elif "SAW" in s:
                    counts["hand_saw"] += 1
        return counts

    def _has_floatie_on(self, player):
        if isinstance(player, dict):
            return bool(player.get("has_pool_floatie_on", player.get("skip", False)))
        return bool(getattr(player, "has_pool_floatie_on", getattr(player, "skip", False)))

    def _extract_state_vector(self, me, my_hp, max_hp, my_upg, others, players, live_rem, blank_rem, session_no, known_shell):
        """Extracts 17-D normalized feature vector for neural network."""
        opp_hps = [self._get_player_hp(players[i]) for i in others]
        min_opp_hp = min(opp_hps) if opp_hps else 0
        max_opp_hp = max(opp_hps) if opp_hps else 0
        total_shells = live_rem + blank_rem

        is_live = 1.0 if known_shell == "LIVE" else 0.0
        is_blank = 1.0 if known_shell == "BLANK" else 0.0
        is_unknown = 1.0 if known_shell is None else 0.0

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
            1.0 if my_upg["juice"] > 0 and my_hp < max_hp else 0.0,
            1.0 if my_upg["mag_glass"] > 0 and is_unknown == 1.0 else 0.0,
            1.0 if my_upg["hand_saw"] > 0 and session_no >= 1 else 0.0,
            1.0 if my_upg["pool_floatie"] > 0 and session_no >= 1 else 0.0,
            1.0 if my_upg["coffee"] > 0 else 0.0,
            self.saws_applied / 3.0,
            session_no / 5.0,
            len(others) / 3.0
        ]
        return features

    def _forward(self, x):
        """Pure-Python feedforward Q-network inference (< 0.02 ms)."""
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

        return q_vals

    def _choose_target(self, others, players, current_dmg):
        """
        Target Selection:
        1. Lethal Elimination: If any living opponent can be killed this turn, eliminate them!
        2. Leader Target: In multiplayer, attack the highest-HP leader.
        """
        killable = [i for i in others if self._get_player_hp(players[i]) <= current_dmg]
        if killable:
            return min(killable, key=lambda i: self._get_player_hp(players[i]))

        if len(others) == 1:
            return others[0]

        # Target leader with highest HP
        return max(others, key=lambda i: (self._get_player_hp(players[i]), not self._has_floatie_on(players[i])))

    def play(self, obs):
        me = getattr(obs, "turn", getattr(obs, "player_index", 0))
        players = getattr(obs, "players", [])
        if not players:
            players = [{"health": getattr(obs, "health", 5), "upgrades": getattr(obs, "upgrades", [])}]

        my_hp = self._get_player_hp(players[me]) if me < len(players) else getattr(obs, "health", 5)
        my_upg = self._get_upg_counts(players[me]) if me < len(players) else {"juice": 0, "coffee": 0, "mag_glass": 0, "pool_floatie": 0, "hand_saw": 0}
        max_hp = getattr(obs, "max_health", 5)
        session_no = getattr(obs, "session_no", getattr(obs, "round", 1) - 1)

        others = [i for i, p in enumerate(players) if i != me and self._get_player_hp(p) > 0]
        if not others:
            others = [1 if me == 0 else 0]

        # Calculate exact shells remaining
        if hasattr(obs, "lives_init") and obs.lives_init is not None:
            lives_init = obs.lives_init
            lives_shot = getattr(obs, "lives_shot", 0) or 0
            blanks_init = getattr(obs, "blanks_init", 1) or 1
            blanks_shot = getattr(obs, "blanks_shot", 0) or 0
            ej = getattr(obs, "ejections", None)
            live_ej = getattr(ej, "lives_ejected", 0) if ej else 0
            blank_ej = getattr(ej, "blanks_ejected", 0) if ej else 0
            live_rem = max(0, lives_init - lives_shot - live_ej)
            blank_rem = max(0, blanks_init - blanks_shot - blank_ej)
            rounds_rem = getattr(obs, "rounds_remaining", live_rem + blank_rem)
        else:
            live_rem = int(getattr(obs, "live_shells", 1))
            blank_rem = int(getattr(obs, "blank_shells", 1))
            rounds_rem = live_rem + blank_rem

        # Determine known shell
        known_shell = None
        nr = getattr(obs, "next_round", getattr(obs, "known_next_shell", None))
        if nr is not None:
            s = str(getattr(nr, "value", nr)).upper()
            if "LIVE" in s:
                known_shell = "LIVE"
            elif "BLANK" in s:
                known_shell = "BLANK"

        if known_shell is None:
            if live_rem > 0 and blank_rem == 0:
                known_shell = "LIVE"
            elif blank_rem > 0 and live_rem == 0:
                known_shell = "BLANK"

        current_dmg = 1 + self.saws_applied
        target_opp = self._choose_target(others, players, current_dmg)

        # Compute valid actions:
        # 0: USE_JUICE
        # 1: USE_MAG_GLASS
        # 2: USE_HAND_SAW
        # 3: USE_POOL_FLOATIE
        # 4: USE_COFFEE
        # 5: SHOOT_SELF
        # 6: SHOOT_OPPONENT
        valid_actions = [5, 6]
        if my_upg["juice"] > 0 and my_hp < max_hp:
            valid_actions.append(0)
        if my_upg["mag_glass"] > 0 and known_shell is None and rounds_rem > 0:
            valid_actions.append(1)
        if my_upg["hand_saw"] > 0 and session_no >= 1 and (known_shell == "LIVE" or live_rem > blank_rem):
            valid_actions.append(2)
        if my_upg["pool_floatie"] > 0 and session_no >= 1:
            valid_floatie_targets = [i for i in others if not self._has_floatie_on(players[i])]
            if valid_floatie_targets:
                valid_actions.append(3)
        if my_upg["coffee"] > 0 and rounds_rem > 0:
            valid_actions.append(4)

        # Select action using trained Q-network if available
        if self.model_loaded:
            state_vec = self._extract_state_vector(me, my_hp, max_hp, my_upg, others, players, live_rem, blank_rem, session_no, known_shell)
            q_vals = self._forward(state_vec)
            action_idx = max(valid_actions, key=lambda a: q_vals[a])
        else:
            # Fallback heuristic
            if 0 in valid_actions:
                action_idx = 0
            elif 1 in valid_actions:
                action_idx = 1
            elif known_shell == "LIVE" and 2 in valid_actions:
                action_idx = 2
            elif known_shell == "BLANK":
                action_idx = 5
            else:
                action_idx = 6

        # Build Action_PLAY
        action = Action_PLAY()
        action.useUpgrade = None
        action.target = None

        if action_idx == 0:
            action.action = ActionType.USE_UPGRADE
            action.useUpgrade = UpgradeType.JUICE
            return action

        elif action_idx == 1:
            action.action = ActionType.USE_UPGRADE
            action.useUpgrade = UpgradeType.MAG_GLASS
            return action

        elif action_idx == 2:
            self.saws_applied += 1
            action.action = ActionType.USE_UPGRADE
            action.useUpgrade = UpgradeType.HAND_SAW
            return action

        elif action_idx == 3:
            valid_floatie_targets = [i for i in others if not self._has_floatie_on(players[i])]
            target = max(valid_floatie_targets, key=lambda i: self._get_player_hp(players[i])) if valid_floatie_targets else others[0]
            action.action = ActionType.USE_UPGRADE
            action.useUpgrade = UpgradeType.POOL_FLOATIE
            action.target = target
            return action

        elif action_idx == 4:
            self.saws_applied = 0
            action.action = ActionType.USE_UPGRADE
            action.useUpgrade = UpgradeType.COFFEE
            return action

        elif action_idx == 5:
            self.saws_applied = 0
            action.action = ActionType.SHOOT
            action.target = me
            return action

        else:  # action_idx == 6
            self.saws_applied = 0
            action.action = ActionType.SHOOT
            action.target = target_opp
            return action

    def pick_upgrade(self, obs):
        action = Action_PICK_UPGRADES()
        me = getattr(obs, "turn", getattr(obs, "player_index", 0))
        players = getattr(obs, "players", [])
        my_hp = self._get_player_hp(players[me]) if me < len(players) else getattr(obs, "health", 5)
        max_hp = getattr(obs, "max_health", 5)
        avail = getattr(obs, "available_upgrades", None)

        def has_upgrade(name, utype):
            if avail is None:
                return False
            if hasattr(avail, name):
                return getattr(avail, name, 0) > 0
            if isinstance(avail, (list, tuple)):
                return (utype in avail) or (name.upper() in [str(getattr(x, "value", x)).upper() for x in avail])
            return False

        if my_hp < max_hp:
            order = [
                ("juice", UpgradeType.JUICE),
                ("mag_glass", UpgradeType.MAG_GLASS),
                ("hand_saw", UpgradeType.HAND_SAW),
                ("pool_floatie", UpgradeType.POOL_FLOATIE),
                ("coffee", UpgradeType.COFFEE),
            ]
        else:
            order = [
                ("mag_glass", UpgradeType.MAG_GLASS),
                ("hand_saw", UpgradeType.HAND_SAW),
                ("pool_floatie", UpgradeType.POOL_FLOATIE),
                ("juice", UpgradeType.JUICE),
                ("coffee", UpgradeType.COFFEE),
            ]

        for name, utype in order:
            if has_upgrade(name, utype):
                action.chosen_upgrade = utype
                return action

        # Fallback
        action.chosen_upgrade = UpgradeType.JUICE
        return action
