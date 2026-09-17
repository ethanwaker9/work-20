import numpy as np
from ..params import Params
from ..mq import MQInstance
from ..relation import ThresholdMQRelation
from ..tcith import Engine, prove, verify
from ..prg import h
from .. import field as F


class ConcatRS:

    name = "CONCAT"

    def __init__(self, par, mq_seed=b"\x00" * 16, d=2):
        self.base = par
        self.d = d
        self.mq = MQInstance(mq_seed, par.mq_n, par.mq_m)
        self.tagmq = MQInstance(mq_seed + b"tag", par.mq_n, par.mq_n)
        self.unit = Params(ring_N=par.ring_N, t=1, d=d, n_party=par.n_party,
                           mq_n=par.mq_n, mq_m=par.mq_m, lam=par.lam)

    def keygen(self, rng):
        x = rng.integers(0, 4, size=self.unit.mq_n, dtype=np.uint8)
        return x, self.mq.eval_gf4_idx(x)

    def setup_ring(self, rng, N):
        sks, pks = [], []
        for _ in range(N):
            x, y = self.keygen(rng)
            sks.append(x); pks.append(y)
        return sks, np.array(pks, dtype=np.uint16)

    def _rel(self, Y, tag):
        return ThresholdMQRelation(self.unit, self.mq, Y, tag_mq=self.tagmq,
                                   tags=np.asarray(tag, dtype=np.uint16).reshape(1, -1))

    def sign(self, msg, Y, signer_idx, sks):
        ctxb = h(Y, self.base.t, dsep=b"cring")
        out = []
        for i in signer_idx:
            tag = self.tagmq.eval_gf4_idx(sks[i])
            rel = self._rel(Y, tag)
            eng = Engine(self.unit, rel)
            w = rel.build_witness([sks[i]], [int(i)])
            pi, _ = prove(eng, rel, w, msg, ctxb)
            out.append(_tagbytes(tag) + pi)
        return b"".join(out)

    def verify(self, msg, Y, sig, t):
        ctxb = h(Y, self.base.t, dsep=b"cring")
        rel0 = self._rel(Y, np.zeros(self.tagmq.m, dtype=np.uint16))
        eng = Engine(self.unit, rel0)
        tl = (2 * self.tagmq.m + 7) // 8
        unit = tl + _proof_len(eng, rel0)
        if len(sig) != t * unit:
            return False
        seen = set()
        for k in range(t):
            blk = sig[k * unit:(k + 1) * unit]
            tag = _tagparse(blk[:tl], self.tagmq.m)
            if bytes(blk[:tl]) in seen:
                return False
            seen.add(bytes(blk[:tl]))
            rel = self._rel(Y, tag)
            if not verify(eng, rel, blk[tl:], msg, ctxb):
                return False
        return True


def _tagbytes(tag):
    idx = np.array([int(np.where(F.GF4 == v)[0][0]) for v in np.asarray(tag)], dtype=np.uint8)
    bits = np.stack([idx & 1, (idx >> 1) & 1], axis=1).reshape(-1)
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="little").tobytes()


def _tagparse(buf, m):
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")[:2 * m]
    idx = bits.reshape(m, 2)
    return F.GF4[(idx[:, 0] + 2 * idx[:, 1]).astype(np.int64)]


def _proof_len(eng, rel):
    return (2 * 16 + 32 + eng.tau * (eng.dwlen + eng.pathlen * 16 + 32
                                     + eng.rho * (eng.D - 1) * 2))
