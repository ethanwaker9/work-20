import numpy as np
from . import field as F
from .prg import xof


class MQInstance:
    def __init__(self, seed, n, m):
        self.n = n
        self.m = m
        self.seed = seed
        self.iu = np.triu_indices(n)
        self.L = len(self.iu[0])
        raw = xof(seed, m * (self.L + n), b"mqinst")
        buf = (np.frombuffer(raw, dtype=np.uint8) & 3).reshape(m, self.L + n)
        self.A_idx = np.ascontiguousarray(buf[:, :self.L])
        self.b_idx = np.ascontiguousarray(buf[:, self.L:])
        self.A = F.GF4[self.A_idx.astype(np.int64)]
        self.B = F.GF4[self.b_idx.astype(np.int64)]
        self.nb = (m + 7) // 8
        self.PA = [_bitpack(self.A_idx, c, self.nb) for c in (0, 1)]
        self.PB = [_bitpack(self.b_idx, c, self.nb) for c in (0, 1)]

    def tri(self, x):
        return F.mul(x[..., self.iu[0]], x[..., self.iu[1]])

    def eval(self, x):
        x = np.asarray(x, dtype=np.uint16)
        o = self.tri(x)
        q = np.bitwise_xor.reduce(F.mul(self.A, o[None, :]), axis=1)
        l = np.bitwise_xor.reduce(F.mul(self.B, x[None, :]), axis=1)
        return q ^ l

    def eval_gf4_idx(self, x_idx):
        return self.eval(F.GF4[np.asarray(x_idx, dtype=np.int64)])

    def combine(self, gamma):
        g = [np.asarray(gamma, dtype=np.uint16), F.smul(int(F.GF4[2]), gamma)]
        Ag = np.zeros(self.L, dtype=np.uint16)
        bg = np.zeros(self.n, dtype=np.uint16)
        for c in (0, 1):
            T = _xor_tables(g[c], self.m, self.nb)
            for b in range(self.nb):
                Ag ^= T[b][self.PA[c][b]]
                bg ^= T[b][self.PB[c][b]]
        return Ag, bg


def _bitpack(idx, c, nb):
    bits = ((idx >> c) & 1).astype(np.uint8)
    m, L = bits.shape
    out = np.zeros((nb, L), dtype=np.uint8)
    for j in range(m):
        out[j // 8] |= bits[j] << (j % 8)
    return out


def _xor_tables(g, m, nb):
    T = np.zeros((nb, 256), dtype=np.uint16)
    gg = np.zeros(8 * nb, dtype=np.uint16)
    gg[:m] = np.asarray(g, dtype=np.uint16)
    gg = gg.reshape(nb, 8)
    for r in range(8):
        step = 1 << r
        T[:, step:2 * step] = T[:, :step] ^ gg[:, r][:, None]
    return T
