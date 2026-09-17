import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.harness import ALL, measure

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def grid():
    pts = []
    for t in [2, 4, 8, 16, 32]:
        pts.append((64, t))
    for N in [16, 32, 64, 128, 256, 512]:
        if (N, 8) not in pts:
            pts.append((N, 8))
    pts.append((100, 50))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(RESULTS, "bench.json"))
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    if os.path.exists(args.out):
        rows = json.load(open(args.out))
    done = {(r["scheme"], r["N"], r["t"]) for r in rows}
    for (N, t) in grid():
        for ad in ALL:
            if args.only and ad.name != args.only:
                continue
            if (ad.name, N, t) in done:
                continue
            t0 = time.time()
            reps = args.reps if N <= 128 else 1
            r = measure(ad, N, t, reps=reps, seed=1234)
            r["wall_s"] = time.time() - t0
            rows.append(r)
            print(json.dumps(r), flush=True)
            json.dump(rows, open(args.out, "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
