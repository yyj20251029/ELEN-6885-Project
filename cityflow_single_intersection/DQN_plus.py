import argparse
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from envs.cityflow_single_intersection_env import CityFlowSingleIntersectionEnv


# -------------------------
# Utils
# -------------------------
def flow_tag(flow_file: str) -> str:
    return os.path.splitext(os.path.basename(flow_file))[0]


def ensure_dirs():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)


# -------------------------
# Plus Feature Wrapper
# -------------------------
class PlusFeatureWrapper:
    """
    State:
      [phase, time_in_phase,
       Q_NS, Q_EW,
       dQ_NS, dQ_EW,
       W_NS, W_EW,
       A_NS, A_EW]

    Reward:
      - (Q_total(t+1) - Q_total(t))
      - beta * Var(Q_NS, Q_EW)
      - switch_penalty * 1[action != prev_action]
    """
    def __init__(
        self,
        env: CityFlowSingleIntersectionEnv,
        switch_penalty: float = 0.1,
        beta: float = 0.2,
        arrival_window: int = 5,
    ):
        self.env = env
        self.switch_penalty = switch_penalty
        self.beta = beta
        self.arrival_window = arrival_window

        self._wait_consec = {}
        self._prev_action = 0
        self._prev_Q_NS = 0.0
        self._prev_Q_EW = 0.0

        self._arrival_hist = {
            "NS": deque(maxlen=arrival_window),
            "EW": deque(maxlen=arrival_window),
        }

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self._wait_consec = {}
        self._prev_action = 0
        self._prev_Q_NS = 0.0
        self._prev_Q_EW = 0.0
        self._arrival_hist["NS"].clear()
        self._arrival_hist["EW"].clear()

        self._update_waits()
        Q_NS, Q_EW, _, _ = self._get_ns_ew_stats()
        self._prev_Q_NS = Q_NS
        self._prev_Q_EW = Q_EW
        return self._get_state()

    def step(self, action: int):
        _, _, done, info = self.env.step(action)
        self._update_waits()

        Q_NS, Q_EW, _, _ = self._get_ns_ew_stats()
        Q_total_prev = self._prev_Q_NS + self._prev_Q_EW
        Q_total_now = Q_NS + Q_EW

        reward = -(Q_total_now - Q_total_prev)
        reward -= self.beta * np.var([Q_NS, Q_EW])
        if action != self._prev_action:
            reward -= self.switch_penalty

        self._prev_action = action
        self._prev_Q_NS = Q_NS
        self._prev_Q_EW = Q_EW

        return self._get_state(), float(reward), done, info

    # -------------------------
    # helpers
    # -------------------------
    def _update_waits(self):
        if self.env.engine is None:
            return

        thr = self.env.queue_speed_thresh
        lane_vehicles = self.env.engine.get_lane_vehicles()
        vehicle_speeds = self.env.engine.get_vehicle_speed()
        veh_now = set(self.env.engine.get_vehicles())

        for vid in list(self._wait_consec.keys()):
            if vid not in veh_now:
                del self._wait_consec[vid]

        arrivals_NS, arrivals_EW = 0, 0

        for lane in self.env.IN_LANES:
            direction = "NS" if lane in self.env.IN_LANES[:2] else "EW"
            for vid in lane_vehicles.get(lane, []):
                if vehicle_speeds.get(vid, 0.0) < thr:
                    self._wait_consec[vid] = self._wait_consec.get(vid, 0) + 1
                else:
                    self._wait_consec[vid] = 0

                if self._wait_consec[vid] == 1:
                    if direction == "NS":
                        arrivals_NS += 1
                    else:
                        arrivals_EW += 1

        self._arrival_hist["NS"].append(arrivals_NS)
        self._arrival_hist["EW"].append(arrivals_EW)

    def _get_ns_ew_stats(self):
        thr = self.env.queue_speed_thresh
        lane_vehicles = self.env.engine.get_lane_vehicles()
        vehicle_speeds = self.env.engine.get_vehicle_speed()

        Q = {"NS": 0, "EW": 0}
        W = {"NS": [], "EW": []}

        for lane in self.env.IN_LANES:
            direction = "NS" if lane in self.env.IN_LANES[:2] else "EW"
            for vid in lane_vehicles.get(lane, []):
                if vehicle_speeds.get(vid, 0.0) < thr:
                    Q[direction] += 1
                    W[direction].append(self._wait_consec.get(vid, 0))

        W_NS = float(np.mean(W["NS"])) if W["NS"] else 0.0
        W_EW = float(np.mean(W["EW"])) if W["EW"] else 0.0
        return float(Q["NS"]), float(Q["EW"]), W_NS, W_EW

    def _get_state(self):
        phase = float(self.env.current_phase)
        phase_t = float(self.env.time_in_phase)

        Q_NS, Q_EW, W_NS, W_EW = self._get_ns_ew_stats()
        dQ_NS = Q_NS - self._prev_Q_NS
        dQ_EW = Q_EW - self._prev_Q_EW

        A_NS = np.mean(self._arrival_hist["NS"]) if self._arrival_hist["NS"] else 0.0
        A_EW = np.mean(self._arrival_hist["EW"]) if self._arrival_hist["EW"] else 0.0

        return np.array(
            [
                phase, phase_t,
                Q_NS, Q_EW,
                dQ_NS, dQ_EW,
                W_NS, W_EW,
                A_NS, A_EW,
            ],
            dtype=np.float32,
        )


# -------------------------
# DQN
# -------------------------
class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class DQNConfig:
    episodes: int = 400
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    warmup_steps: int = 10_000
    target_update_every: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 120_000
    max_grad_norm: float = 10.0
    device: str = "cpu"


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buf), size=batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


def epsilon_by_step(step: int, cfg: DQNConfig) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    frac = step / float(cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


# -------------------------
# Metrics (UNCHANGED)
# -------------------------
def run_one_episode_for_metrics(env: CityFlowSingleIntersectionEnv, policy_fn):
    obs = env.reset()
    done = False

    total_queue = 0.0
    total_wait_time = 0.0
    total_system_time = 0.0

    while not done:
        a = policy_fn(env)
        obs, _, done, info = env.step(a)

        q = float(np.sum(obs))
        total_queue += q
        total_wait_time += q
        total_system_time += len(env._prev_veh)

    throughput = info["throughput"]
    steps = env.t

    avg_queue = total_queue / steps if steps > 0 else 0.0
    avg_wait = total_wait_time / throughput if throughput > 0 else 0.0
    avg_travel = total_system_time / throughput if throughput > 0 else 0.0

    return throughput, avg_queue, avg_wait, avg_travel


# -------------------------
# Train / Test
# -------------------------
def train(flow_file: str, seed: int, cfg: DQNConfig):
    ensure_dirs()

    env = CityFlowSingleIntersectionEnv(
        base_config_path="cityflow_cfg/config.json",
        flow_path=os.path.join("cityflow_cfg", "flows", flow_file),
        seed=seed,
    )
    wenv = PlusFeatureWrapper(env)

    q = QNet(state_dim=10).to(cfg.device)
    q_tgt = QNet(state_dim=10).to(cfg.device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    opt = optim.Adam(q.parameters(), lr=cfg.lr)
    rb = ReplayBuffer(cfg.buffer_size)

    global_step = 0

    for ep in range(1, cfg.episodes + 1):
        s = wenv.reset(seed=seed)
        done = False
        ep_return = 0.0

        pbar = tqdm(
            total=env.episode_len,
            desc=f"[DQN_plus] Train ep {ep}/{cfg.episodes}",
            leave=False,
        )
        while not done:
            eps = epsilon_by_step(global_step, cfg)
            if np.random.rand() < eps:
                a = np.random.randint(2)
            else:
                with torch.no_grad():
                    st = torch.tensor(s, dtype=torch.float32, device=cfg.device).unsqueeze(0)
                    a = int(torch.argmax(q(st), dim=1).item())

            s2, r, done, _ = wenv.step(a)
            rb.push(s, a, r, s2, done)
            s = s2
            ep_return += r
            global_step += 1

            if len(rb) >= cfg.warmup_steps and len(rb) >= cfg.batch_size:
                bs, ba, br, bs2, bd = rb.sample(cfg.batch_size)
                bs_t = torch.tensor(bs, dtype=torch.float32, device=cfg.device)
                ba_t = torch.tensor(ba, dtype=torch.int64, device=cfg.device).unsqueeze(1)
                br_t = torch.tensor(br, dtype=torch.float32, device=cfg.device).unsqueeze(1)
                bs2_t = torch.tensor(bs2, dtype=torch.float32, device=cfg.device)
                bd_t = torch.tensor(bd.astype(np.float32), dtype=torch.float32, device=cfg.device).unsqueeze(1)

                q_sa = q(bs_t).gather(1, ba_t)
                with torch.no_grad():
                    q_next = q_tgt(bs2_t).max(dim=1, keepdim=True)[0]
                    target = br_t + cfg.gamma * (1.0 - bd_t) * q_next

                loss = nn.functional.smooth_l1_loss(q_sa, target)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), cfg.max_grad_norm)
                opt.step()

                if global_step % cfg.target_update_every == 0:
                    q_tgt.load_state_dict(q.state_dict())

            pbar.update(1)

        pbar.close()
        tqdm.write(f"[DQN_plus] ep={ep} return={ep_return:.2f}")

    torch.save(q.state_dict(), "models/DQN_plus.pt")
    print("[DQN_plus] Model saved to models/DQN_plus.pt")
    env.close()


def test(flow_file: str, seed: int):
    ensure_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = QNet(state_dim=10).to(device)
    q.load_state_dict(torch.load("models/DQN_plus.pt", map_location=device))
    q.eval()

    env = CityFlowSingleIntersectionEnv(
        base_config_path="cityflow_cfg/config.json",
        flow_path=os.path.join("cityflow_cfg", "flows", flow_file),
        seed=seed,
    )
    wenv = PlusFeatureWrapper(env, switch_penalty=0.0)

    def policy_fn(_env):
        s = wenv._get_state()
        with torch.no_grad():
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            return int(torch.argmax(q(st), dim=1).item())

    throughput, avg_queue, avg_wait, avg_travel = run_one_episode_for_metrics(
        env, policy_fn
    )

    print("====== DQN_plus Test ======")
    print(f"Flow file: {flow_file}")
    print(f"Throughput: {throughput}")
    print(f"Average queue length: {avg_queue:.3f}")
    print(f"Average waiting time: {avg_wait:.3f} s")
    print(f"Average travel time: {avg_travel:.3f} s")

    log = {
        "flow_file": flow_file,
        "throughput": int(throughput),
        "average_queue_length": float(avg_queue),
        "average_waiting_time": float(avg_wait),
        "average_travel_time": float(avg_travel),
    }

    out = os.path.join("logs", f"DQN_plus_{flow_tag(flow_file)}.json")
    with open(out, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[DQN_plus] Test log saved to {out}")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("flow_file")
    p_train.add_argument("--seed", type=int, default=0)

    p_test = sub.add_parser("test")
    p_test.add_argument("flow_file")
    p_test.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    cfg = DQNConfig(device="cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        train(args.flow_file, args.seed, cfg)
    else:
        test(args.flow_file, args.seed)


if __name__ == "__main__":
    main()
