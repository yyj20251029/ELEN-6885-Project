import argparse
import json
import os
import numpy as np

# --------------------------------------------------
# Straight-only routes (MUST match roadnet.json)
# --------------------------------------------------
ROUTES = {
    "N": ["R_N_in", "R_S_out"],
    "S": ["R_S_in", "R_N_out"],
    "E": ["R_E_in", "R_W_out"],
    "W": ["R_W_in", "R_E_out"],
}

DIRECTIONS = ["N", "S", "E", "W"]

# --------------------------------------------------
# Vehicle template (match CityFlow example exactly)
# --------------------------------------------------
VEHICLE_TEMPLATE = {
    "length": 5.0,
    "width": 2.0,
    "maxSpeed": 29.0,
    "maxPosAcc": 2.0,
    "maxNegAcc": 4.5,
    "usualPosAcc": 1.0,
    "usualNegAcc": 2.0,
    "minGap": 2.5,
    "headwayTime": 1.5
}


def generate_flow(p: float, seed: int, steps: int):
    rng = np.random.default_rng(seed)
    flows = []

    for t in range(steps):
        for d in DIRECTIONS:
            if rng.random() < p:
                flows.append({
                    "vehicle": VEHICLE_TEMPLATE,
                    "route": ROUTES[d],
                    "interval": 1,
                    "startTime": int(t),
                    "endTime": int(t)
                })

    return flows


def main():
    parser = argparse.ArgumentParser(
        description="Generate CityFlow flow file (flow-style schema)"
    )
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    flows = generate_flow(args.p, args.seed, args.steps)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(flows, f, indent=2)

    print(f"[gen_flow] wrote {len(flows)} flow entries to {args.out}")
    print(f"[gen_flow] p={args.p}, seed={args.seed}, steps={args.steps}")


if __name__ == "__main__":
    main()
