import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bench.harness import ALL, measure


def test_all_schemes_small():
    for ad in ALL:
        r = measure(ad, N=12, t=3, reps=1, seed=1)
        assert r["sig_bytes"] > 0
        print(r)


def test_reject_wrong_message():
    from mqtrs.params import best_params
    from mqtrs.scheme import MQuorum
    par = best_params(16, 3, mq_n=32, mq_m=32)
    sch = MQuorum(par)
    rng = np.random.default_rng(0)
    sks, Y = sch.setup_ring(rng, 16)
    sig = sch.sign(b"a", Y, [1, 5, 9], sks)
    assert sch.verify(b"a", Y, sig)
    assert not sch.verify(b"b", Y, sig)


def test_reject_repeated_signer():
    from mqtrs.params import best_params
    from mqtrs.relation import ThresholdMQRelation
    from mqtrs.scheme import MQuorum
    from mqtrs import field as F
    par = best_params(16, 3, mq_n=32, mq_m=32)
    sch = MQuorum(par)
    rng = np.random.default_rng(0)
    sks, Y = sch.setup_ring(rng, 16)
    rel = ThresholdMQRelation(par, sch.mq, Y)
    w = rel.build_witness([sks[2], sks[2], sks[7]], [2, 2, 7])
    ctx = rel.precompute(b"seed")
    assert int(rel.eval(ctx, F.GF4[w.astype(np.int64)])) != 0


if __name__ == "__main__":
    test_all_schemes_small()
    test_reject_wrong_message()
    test_reject_repeated_signer()
    print("all tests passed")
