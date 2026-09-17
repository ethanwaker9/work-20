import os
import numpy as np
from . import field as F
from .params import Params, best_params
from .mq import MQInstance
from .relation import ThresholdMQRelation
from .tcith import Engine, prove, verify
from .prg import h


class MQuorum:
    name = "MQuorum"

    def __init__(self, par, mq_seed=b"\x00" * 16):
        self.par = par
        self.mq = MQInstance(mq_seed, par.mq_n, par.mq_m)

    def keygen(self, rng):
        x = rng.integers(0, 4, size=self.par.mq_n, dtype=np.uint8)
        y = self.mq.eval_gf4_idx(x)
        return x, y

    def setup_ring(self, rng, N):
        sks, pks = [], []
        for _ in range(N):
            x, y = self.keygen(rng)
            sks.append(x); pks.append(y)
        return sks, np.array(pks, dtype=np.uint16)

    def _engine(self, Y):
        rel = ThresholdMQRelation(self.par, self.mq, Y)
        return Engine(self.par, rel), rel

    def sign(self, msg, Y, signer_idx, sks):
        eng, rel = self._engine(Y)
        w = rel.build_witness([sks[i] for i in signer_idx], list(signer_idx))
        ctxb = h(Y, self.par.t, self.mq.seed, dsep=b"ring")
        sig, _ = prove(eng, rel, w, msg, ctxb)
        return sig

    def verify(self, msg, Y, sig):
        eng, rel = self._engine(Y)
        ctxb = h(Y, self.par.t, self.mq.seed, dsep=b"ring")
        return verify(eng, rel, sig, msg, ctxb)
