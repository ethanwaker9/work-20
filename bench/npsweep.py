import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mqtrs.params import best_params
from mqtrs.scheme import MQuorum

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    rows = []
    for npar in [128, 256, 512, 1024, 2048]:
        for (N, t) in [(64, 8), (100, 50)]:
            par = best_params(N, t, n_party=npar, mq_n=88, mq_m=88)
            sch = MQuorum(par)
            rng = np.random.default_rng(11)
            sks, Y = sch.setup_ring(rng, N)
            idx = sorted(rng.choice(N, size=t, replace=False).tolist())
            ts, tv = [], []
            for _ in range(2):
                t0 = time.perf_counter()
                sig = sch.sign(b"np sweep", Y, idx, sks)
                t1 = time.perf_counter()
                assert sch.verify(b"np sweep", Y, sig)
                t2 = time.perf_counter()
                ts.append(t1 - t0); tv.append(t2 - t1)
            rows.append(dict(n_party=npar, N=N, t=t, tau=par.tau,
                             sig_bytes=len(sig), sign_ms=1000 * float(np.median(ts)),
                             verify_ms=1000 * float(np.median(tv))))
            print(rows[-1], flush=True)
    json.dump(rows, open(os.path.join(RESULTS, "npsweep.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
