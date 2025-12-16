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


def flow_tag(flow_file: str) -> str:
    return os.path.splitext(os.path.basename(flow_file))[0]


def ensure_dirs():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)


class WaitPressureWrapper:
    """
    State:
      [phase, phase_time,
       avg_wait_N, avg_wait_S, avg_wait_E, avg_wait_W,
       Q_NS, Q_EW]

    Reward (IMPORTANT):
      - sum of avg waiting times
      - CURRENT queue length at this time step (not cumulative)
      - optional switch penalty
    """
    def __init__(self, env: CityFlowSingleIntersectionEnv, switch_penalty: float = 0.1):
        self.env = env
        self.switch_penalty = switch_penalty
        self._wait_consec = {}
        self._prev_action = 0

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        self._wait_consec = {}
        self._prev_action = 0
        self._update_waits()
        return self._get_state(obs)

    def step(self, action: int):
        # obs is the CURRENT queue length vector [qN, qS, qE, qW]
        obs, _, done, info = self.env.step(action)
        self._update_waits()

        state = self._get_state(obs)
        avg_waits = state[2:6]

        # ---- KEY POINT: current (instantaneous) queue length ----
        current_queue = float(np.sum(obs))

        reward = -float(np.sum(avg_waits)) - current_queue
        if action != self._prev_action:
            reward -= self.switch_penalty
        self._prev_action = action

        return state, reward, done, info, obs

    def _update_waits(self):
        if self.env.engine is None:
            return

        thr = self.env.queue_speed_thresh
        lane_vehicles = self.env.engine.get_lane_vehicles()
        vehicle_speeds = self.env.engine.get_vehicle_speed()
        veh_now = set(self.env.engine.get_vehicles())

        # remove departed vehicles
        for vid in list(self._wait_consec.keys()):
            if vid not in veh_now:
                del self._wait_consec[vid]

        # update consecutive waiting time
        for lane_id in self.env.IN_LANES:
            for vid in lane_vehicles.get(lane_id, []):
                if vehicle_speeds.get(vid, 0.0) < thr:
                    self._wait_consec[vid] = self._wait_consec.get(vid, 0) + 1
                else:
                    self._wait_consec[vid] = 0

    def _avg_wait_per_lane(self) -> np.ndarray:
        thr = self.env.queue_speed_thresh
        lane_vehicles = self.env.engine.get_lane_vehicles()
        vehicle_speeds = self.env.engine.get_vehicle_speed()

        avgs = []
        for lane_id in self.env.IN_LANES:
            queued_vids = [
                vid for vid in lane_vehicles.get(lane_id, [])
                if vehicle_speeds.get(vid, 0.0) < thr
            ]

            if len(queued_vids) == 0:
                avgs.append(0.0)
            else:
                waits = [float(self._wait_consec.get(vid, 0)) for vid in queued_vids]
                avgs.append(float(np.mean(waits)))

        return np.array(avgs, dtype=np.float32)

    def _get_state(self, obs_queue: np.ndarray) -> np.ndarray:
        phase = float(self.env.current_phase)
        phase_t = float(self.env.time_in_phase)
        avg_waits = self._avg_wait_per_lane()

        q_ns = float(obs_queue[0] + obs_queue[1])
        q_ew = float(obs_queue[2] + obs_queue[3])

        return np.concatenate(
            [
                np.array([phase, phase_t], dtype=np.float32),
                avg_waits,
                np.array([q_ns, q_ew], dtype=np.float32),
            ],
            axis=0,
        )


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
    episodes: int = 200
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    warmup_steps: int = 5_000
    target_update_every: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
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


# =========================
# Training / Testing
# =========================





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


def train(train_flow_file: str, seed: int, switch_penalty: float, cfg: DQNConfig):
    ensure_dirs()

    base_config_path = "cityflow_cfg/config.json"
    flow_path = os.path.join("cityflow_cfg", "flows", train_flow_file)
    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Flow file not found: {flow_path}")

    env = CityFlowSingleIntersectionEnv(base_config_path=base_config_path, flow_path=flow_path, seed=seed)
    wenv = WaitPressureWrapper(env, switch_penalty=switch_penalty)

    state_dim = 2 + 4 + 2
    q = QNet(state_dim).to(cfg.device)
    q_tgt = QNet(state_dim).to(cfg.device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    opt = optim.Adam(q.parameters(), lr=cfg.lr)
    rb = ReplayBuffer(cfg.buffer_size)

    global_step = 0

    for ep in range(1, cfg.episodes + 1):
        s = wenv.reset(seed=seed)
        done = False
        ep_return = 0.0

        pbar = tqdm(total=env.episode_len, desc=f"Train ep {ep}/{cfg.episodes}", leave=False)
        while not done:
            eps = epsilon_by_step(global_step, cfg)
            if np.random.rand() < eps:
                a = np.random.randint(2)
            else:
                with torch.no_grad():
                    st = torch.tensor(s, dtype=torch.float32, device=cfg.device).unsqueeze(0)
                    a = int(torch.argmax(q(st), dim=1).item())

            s2, r, done, info, _obs_queue = wenv.step(a)
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
        tqdm.write(f"[MP_DQN] ep={ep} return={ep_return:.2f} eps={epsilon_by_step(global_step, cfg):.3f}")

    model_path = os.path.join("models", "MP_DQN.pt")
    torch.save(q.state_dict(), model_path)
    print(f"[MP_DQN] saved model to {model_path}")

    env.close()


def test(test_flow_file: str, seed: int):
    ensure_dirs()

    base_config_path = "cityflow_cfg/config.json"
    flow_path = os.path.join("cityflow_cfg", "flows", test_flow_file)
    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Flow file not found: {flow_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dim = 2 + 4 + 2
    q = QNet(state_dim).to(device)
    model_path = os.path.join("models", "MP_DQN.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train first.")
    q.load_state_dict(torch.load(model_path, map_location=device))
    q.eval()

    env = CityFlowSingleIntersectionEnv(base_config_path=base_config_path, flow_path=flow_path, seed=seed)
    wenv = WaitPressureWrapper(env, switch_penalty=0.0)

    def policy_fn(_env):
        # use wrapped state (need obs for Q_NS/Q_EW; reuse env._get_obs())
        obs_queue = env._get_obs()
        s = wenv._get_state(obs_queue)
        with torch.no_grad():
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            return int(torch.argmax(q(st), dim=1).item())

    throughput, avg_queue, avg_wait, avg_travel = run_one_episode_for_metrics(env, policy_fn)

    print("====== MP_DQN Test ======")
    print(f"Flow file: {test_flow_file}")
    print(f"Throughput: {throughput}")
    print(f"Average queue length: {avg_queue:.3f}")
    print(f"Average waiting time: {avg_wait:.3f} s")
    print(f"Average travel time: {avg_travel:.3f} s")

    log = {
        "flow_file": test_flow_file,
        "throughput": int(throughput),
        "average_queue_length": float(avg_queue),
        "average_waiting_time": float(avg_wait),
        "average_travel_time": float(avg_travel),
    }

    out = os.path.join("logs", f"MP_DQN_{flow_tag(test_flow_file)}.json")
    with open(out, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log saved to {out}")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("flow_file", type=str, help="e.g. flow_medium_train.json")
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--episodes", type=int, default=200)
    p_train.add_argument("--switch_penalty", type=float, default=0.1)

    p_test = sub.add_parser("test")
    p_test.add_argument("flow_file", type=str, help="e.g. flow_low0.json")
    p_test.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DQNConfig(device=device)

    if args.mode == "train":
        cfg.episodes = args.episodes
        train(args.flow_file, seed=args.seed, switch_penalty=args.switch_penalty, cfg=cfg)
    else:
        test(args.flow_file, seed=args.seed)


if __name__ == "__main__":
    main()
