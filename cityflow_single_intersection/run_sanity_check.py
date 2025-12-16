import argparse
import os
import numpy as np

from envs.cityflow_single_intersection_env import CityFlowSingleIntersectionEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument(
        "--policy",
        type=str,
        default="switch30",
        choices=["random", "fixed0", "fixed1", "switch30", "switch60"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=60)
    args = parser.parse_args()

    base_config = "cityflow_cfg/config.json"
    flow_path = f"cityflow_cfg/flows/flow_{args.flow}.json"

    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"{flow_path} not found.")

    env = CityFlowSingleIntersectionEnv(
        base_config_path=base_config,
        flow_path=flow_path,
        episode_len=3600,
        min_green=3,
        yellow=1,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)

    obs = env.reset()
    queue_sums = []

    # throughput tracking (using env.engine, not _engine)
    prev_veh = set(env.engine.get_vehicles())
    throughput = 0
    last_throughput = 0

    for t in range(3600):
        if args.policy == "random":
            action = int(rng.integers(0, 2))
        elif args.policy == "fixed0":
            action = 0
        elif args.policy == "fixed1":
            action = 1
        elif args.policy == "switch30":
            action = 0 if (t // 30) % 2 == 0 else 1
        elif args.policy == "switch60":
            action = 0 if (t // 60) % 2 == 0 else 1
        else:
            action = 0

        obs, reward, done, info = env.step(action)
        queue_sums.append(info["queue_sum"])

        veh_now = set(env.engine.get_vehicles())
        left = prev_veh - veh_now
        throughput += len(left)
        prev_veh = veh_now

        if (t + 1) % args.print_every == 0:
            delta = throughput - last_throughput
            last_throughput = throughput
            print(
                f"[t={t+1:4d}] phase={info['phase']} yellow={info['in_yellow']} "
                f"queues={info['queues']} sum={info['queue_sum']:.1f} "
                f"throughput={throughput} (+{delta})"
            )

        if done:
            break

    env.close()

    avg_q = float(np.mean(queue_sums))
    print("\n===== SANITY CHECK SUMMARY =====")
    print(f"flow: {args.flow}, policy: {args.policy}")
    print(f"avg_queue_sum: {avg_q:.3f}")
    print(f"final_throughput: {throughput}")
    print("================================")


if __name__ == "__main__":
    main()
