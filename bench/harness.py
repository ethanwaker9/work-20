import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mqtrs.params import best_params
from mqtrs.scheme import MQuorum
from mqtrs.baselines.concat import ConcatRS
from mqtrs.baselines.multivariate_trs import PBB13, MTRS5
from mqtrs.baselines.cds import CDSMQ
from mqtrs.baselines.hs20 import HS20
from bench.complexity import pk_bytes

MQ_N = 88
MQ_M = 88


class Adapter:
    def __init__(self, name):
        self.name = name

    def setup(self, N, t, rng):
        raise NotImplementedError


class OursAdapter(Adapter):
    def __init__(self, n_party=256):
        super().__init__("MQuorum")
        self.n_party = n_party

    def setup(self, N, t, rng):
        par = best_params(N, t, n_party=self.n_party, mq_n=MQ_N, mq_m=MQ_M)
        sch = MQuorum(par)
        sks, Y = sch.setup_ring(rng, N)
        idx = sorted(rng.choice(N, size=t, replace=False).tolist())
        return dict(sch=sch, sks=sks, Y=Y, idx=idx, par=par)

    def sign(self, st, msg):
        return st["sch"].sign(msg, st["Y"], st["idx"], st["sks"])

    def verify(self, st, msg, sig):
        return st["sch"].verify(msg, st["Y"], sig)

    def pk(self, st):
        return pk_bytes("MQuorum")


class ConcatAdapter(Adapter):
    def __init__(self):
        super().__init__("CONCAT")

    def setup(self, N, t, rng):
        par = best_params(N, t, mq_n=MQ_N, mq_m=MQ_M)
        sch = ConcatRS(par, d=best_params(N, 1, mq_n=MQ_N, mq_m=MQ_M).d)
        sks, Y = sch.setup_ring(rng, N)
        idx = sorted(rng.choice(N, size=t, replace=False).tolist())
        return dict(sch=sch, sks=sks, Y=Y, idx=idx, t=t)

    def sign(self, st, msg):
        return st["sch"].sign(msg, st["Y"], st["idx"], st["sks"])

    def verify(self, st, msg, sig):
        return st["sch"].verify(msg, st["Y"], sig, st["t"])

    def pk(self, st):
        return pk_bytes("CONCAT")


class MVAdapter(Adapter):
    def __init__(self, cls):
        super().__init__(cls.name)
        self.cls = cls

    def setup(self, N, t, rng):
        sch = self.cls(n=MQ_N, m=MQ_M)
        roots, maps = sch.setup_ring(rng, N)
        idx = set(rng.choice(N, size=t, replace=False).tolist())
        return dict(sch=sch, roots=roots, maps=maps, idx=idx, t=t)

    def sign(self, st, msg):
        return st["sch"].sign(msg, st["maps"], st["roots"], st["idx"], st["t"])

    def verify(self, st, msg, sig):
        return st["sch"].verify(msg, st["maps"], sig, st["t"])

    def pk(self, st):
        return pk_bytes(self.name)


class CDSAdapter(Adapter):
    def __init__(self):
        super().__init__("CDS-MQ")

    def setup(self, N, t, rng):
        sch = CDSMQ(n=MQ_N, m=MQ_M)
        sks, pks = sch.setup_ring(rng, N)
        idx = set(rng.choice(N, size=t, replace=False).tolist())
        return dict(sch=sch, sks=sks, pks=pks, idx=idx, t=t)

    def sign(self, st, msg):
        return st["sch"].sign(msg, st["pks"], st["sks"], st["idx"], st["t"])

    def verify(self, st, msg, sig):
        return st["sch"].verify(msg, st["pks"], sig, st["t"])

    def pk(self, st):
        return pk_bytes("CDS-MQ")


class HSAdapter(Adapter):
    def __init__(self):
        super().__init__("HS20")

    def setup(self, N, t, rng):
        sch = HS20()
        _, alphas = sch.setup_ring(rng, N)
        idx = set(rng.choice(N, size=t, replace=False).tolist())
        return dict(sch=sch, alphas=alphas, idx=idx, t=t, N=N)

    def sign(self, st, msg):
        return st["sch"].sign(msg, st["alphas"], st["idx"], st["N"], st["t"])

    def verify(self, st, msg, sig):
        return st["sch"].verify(msg, st["alphas"], sig, st["N"], st["t"])

    def pk(self, st):
        return pk_bytes("HS20")


ALL = [MVAdapter(PBB13), MVAdapter(MTRS5), CDSAdapter(), HSAdapter(),
       ConcatAdapter(), OursAdapter()]


def measure(ad, N, t, reps=1, seed=0, msg=b"threshold ring signature benchmark"):
    rng = np.random.default_rng(seed)
    st = ad.setup(N, t, rng)
    ts, tv, sz = [], [], 0
    for _ in range(reps):
        t0 = time.perf_counter()
        sig = ad.sign(st, msg)
        t1 = time.perf_counter()
        ok = ad.verify(st, msg, sig)
        t2 = time.perf_counter()
        if not ok:
            raise RuntimeError("verification failed for " + ad.name)
        ts.append(t1 - t0); tv.append(t2 - t1); sz = len(sig)
    tracemalloc.start()
    tracemalloc.reset_peak()
    sig = ad.sign(st, msg)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return dict(scheme=ad.name, N=N, t=t, sign_ms=1000 * float(np.median(ts)),
                verify_ms=1000 * float(np.median(tv)), sig_bytes=sz,
                pk_bytes=ad.pk(st), peak_kib=peak / 1024.0)
