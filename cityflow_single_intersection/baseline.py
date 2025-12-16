import argparse
import json
import os
import numpy as np

from envs.cityflow_single_intersection_env import CityFlowSingleIntersectionEnv


# --------------------------------------------------
# Fixed-time controller: 60s NS, 60s EW
# --------------------------------------------------
def fixed_time_action(t, g_ns=60, g_ew=60):
    cycle = g_ns + g_ew
    return 0 if (t % cycle) < g_ns else 1   # 0=NS, 1=EW


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-time baseline for CityFlow single intersection"
    )
    parser.add_argument(
        "flow",
        type=str,
        help="flow file name, e.g. flow_low1.json"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="CityFlow engine seed (for reproducibility)"
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    base_config_path = "cityflow_cfg/config.json"
    flow_path = os.path.join("cityflow_cfg", "flows", args.flow)

    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Flow file not found: {flow_path}")

    # --------------------------------------------------
    # Environment
    # --------------------------------------------------
    env = CityFlowSingleIntersectionEnv(
        base_config_path=base_config_path,
        flow_path=flow_path,
        seed=args.seed,
    )

    obs = env.reset()

    # --------------------------------------------------
    # Metrics accumulators
    # --------------------------------------------------
    total_queue = 0.0          # sum_t queue length
    total_wait_time = 0.0      # sum_t queue(t) * 1s
    total_system_time = 0.0    # sum_t vehicles_in_system * 1s

    done = False

    while not done:
        action = fixed_time_action(env.t, g_ns=30, g_ew=30)
        obs, _, done, info = env.step(action)

        q = float(np.sum(obs))
        total_queue += q
        total_wait_time += q
        total_system_time += len(env._prev_veh)

    # --------------------------------------------------
    # Final statistics
    # --------------------------------------------------
    throughput = info["throughput"]
    steps = env.t

    avg_queue = total_queue / steps if steps > 0 else 0.0
    avg_wait = total_wait_time / throughput if throughput > 0 else 0.0
    avg_travel = total_system_time / throughput if throughput > 0 else 0.0

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------
    print("====== Fixed-Time Baseline (60s / 60s) ======")
    print(f"Flow file: {args.flow}")
    print(f"Throughput: {throughput}")
    print(f"Average queue length: {avg_queue:.3f}")
    print(f"Average waiting time: {avg_wait:.3f} s")
    print(f"Average travel time: {avg_travel:.3f} s")

    # --------------------------------------------------
    # Save log
    # --------------------------------------------------
    os.makedirs("logs", exist_ok=True)

    flow_tag = os.path.splitext(args.flow)[0]   # e.g. flow_low1
    log_path = f"logs/baseline_{flow_tag}.json"

    log = {
        "flow_file": args.flow,
        "seed": args.seed,
        "g_ns": 60,
        "g_ew": 60,
        "throughput": throughput,
        "average_queue_length": avg_queue,
        "average_waiting_time": avg_wait,
        "average_travel_time": avg_travel,
        "episode_length": steps,
    }

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"Log saved to {log_path}")

    env.close()


if __name__ == "__main__":
    main()
