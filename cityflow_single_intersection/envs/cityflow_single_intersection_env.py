import json
import os
import tempfile
from typing import Optional, Dict

import numpy as np

try:
    import cityflow
except ImportError as e:
    raise ImportError("cityflow is not installed.") from e


class CityFlowSingleIntersectionEnv:
    INTERSECTION_ID = "I_center"

    PHASE_NS_GREEN = 0
    PHASE_EW_GREEN = 1
    PHASE_ALL_RED = 2

    IN_LANES = [
        "R_N_in_0",
        "R_S_in_0",
        "R_E_in_0",
        "R_W_in_0",
    ]

    def __init__(
        self,
        base_config_path: str,
        flow_path: str,
        episode_len: int = 3600,
        min_green: int = 3,
        yellow: int = 1,
        seed: int = 0,
        queue_speed_thresh: float = 0.1,
    ):
        self.base_config_path = base_config_path
        self.flow_path = flow_path

        self.episode_len = episode_len
        self.min_green = min_green
        self.yellow = yellow
        self.seed = seed
        self.queue_speed_thresh = queue_speed_thresh

        self.engine: Optional[cityflow.Engine] = None
        self._tmp_config: Optional[str] = None

        self.current_phase = 0
        self.time_in_phase = 0
        self.in_yellow = False
        self.yellow_left = 0
        self.pending_phase: Optional[int] = None

        self.t = 0
        self._prev_veh = set()
        self._throughput = 0

    # --------------------------------------------------
    # Config handling (original working style)
    # --------------------------------------------------
    def _make_tmp_config(self) -> str:
        with open(self.base_config_path, "r") as f:
            cfg = json.load(f)

        cfg["seed"] = self.seed
        cfg["dir"] = ""
        cfg["roadnetFile"] = "cityflow_cfg/roadnet.json"
        cfg["flowFile"] = os.path.relpath(self.flow_path)

        fd, path = tempfile.mkstemp(suffix="_cityflow.json")
        os.close(fd)

        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)

        self._tmp_config = path
        return path

    def _cleanup(self):
        if self._tmp_config and os.path.exists(self._tmp_config):
            os.remove(self._tmp_config)
        self._tmp_config = None

    # --------------------------------------------------
    # API
    # --------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed

        self._cleanup()
        cfg_path = self._make_tmp_config()

        self.engine = cityflow.Engine(cfg_path, thread_num=1)

        self.t = 0
        self.current_phase = 0
        self.time_in_phase = 0
        self.in_yellow = False
        self.yellow_left = 0
        self.pending_phase = None

        self.engine.set_tl_phase(self.INTERSECTION_ID, self.PHASE_NS_GREEN)

        self._prev_veh = set(self.engine.get_vehicles())
        self._throughput = 0

        return self._get_obs()

    def step(self, action: int):
        assert action in (0, 1)

        self._update_signal(action)

        self.engine.next_step()
        self.t += 1

        veh_now = set(self.engine.get_vehicles())
        left = self._prev_veh - veh_now
        self._throughput += len(left)
        self._prev_veh = veh_now

        obs = self._get_obs()
        reward = -float(obs.sum())
        done = self.t >= self.episode_len

        info = {
            "t": self.t,
            "queues": obs.tolist(),
            "queue_sum": float(obs.sum()),
            "throughput": int(self._throughput),
            "phase": int(self.current_phase),
            "in_yellow": bool(self.in_yellow),
        }

        return obs, reward, done, info

    # --------------------------------------------------
    # Signal logic
    # --------------------------------------------------
    def _update_signal(self, action: int):
        if self.in_yellow:
            self.yellow_left -= 1
            if self.yellow_left <= 0:
                self.current_phase = self.pending_phase
                self.pending_phase = None
                self.in_yellow = False
                self.time_in_phase = 0
                self.engine.set_tl_phase(self.INTERSECTION_ID, self.current_phase)
            return

        if action == self.current_phase:
            self.time_in_phase += 1
            return

        if self.time_in_phase >= self.min_green:
            self.in_yellow = True
            self.yellow_left = self.yellow
            self.pending_phase = action
            self.engine.set_tl_phase(self.INTERSECTION_ID, self.PHASE_ALL_RED)
        else:
            self.time_in_phase += 1

    # --------------------------------------------------
    # Observation (CityFlow version-safe)
    # --------------------------------------------------
    def _get_obs(self):
        obs = []
        thr = self.queue_speed_thresh

        lane_vehicles: Dict[str, list] = self.engine.get_lane_vehicles()
        vehicle_speeds: Dict[str, float] = self.engine.get_vehicle_speed()

        for lane_id in self.IN_LANES:
            q = 0
            for vid in lane_vehicles.get(lane_id, []):
                if vehicle_speeds.get(vid, 0.0) < thr:
                    q += 1
            obs.append(q)

        return np.array(obs, dtype=np.float32)

    def close(self):
        self._cleanup()
        self.engine = None
