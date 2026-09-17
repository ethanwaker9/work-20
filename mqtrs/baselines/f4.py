import numpy as np
from ..prg import xof


def pack(bits):
    L = len(bits)
    pad = (-L) % 64
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="little").view(np.uint64)


def parity(M):
    red = np.bitwise_xor.reduce(M, axis=1)
    return (np.bitwise_count(red) & 1).astype(np.uint8)


class F4MQ:
    def __init__(self, seed, n, m, tri=True):
        self.n, self.m = n, m
        self.iu = np.triu_indices(n)
        self.L = len(self.iu[0])
        raw = xof(seed, m * self.L, b"f4mq")
        vals = (np.frombuffer(raw, dtype=np.uint8) & 3).reshape(m, self.L)
        self.vals = vals.copy()
        self._repack()

    def _repack(self):
        self.A0 = np.stack([pack(self.vals[j] & 1) for j in range(self.m)])
        self.A1 = np.stack([pack((self.vals[j] >> 1) & 1) for j in range(self.m)])

    def plant_root(self, s0, s1):
        pos = int(np.nonzero(s0 | s1)[0][0])
        k = int(np.nonzero((self.iu[0] == pos) & (self.iu[1] == pos))[0][0])
        self.vals[:, k] = 0
        self._repack()
        c0, c1 = self.eval(s0, s1)
        a0, a1 = s0[pos], s1[pos]
        p0 = (a0 & a0) ^ (a1 & a1)
        p1 = (a0 & a1) ^ (a1 & a0) ^ (a1 & a1)
        d0, d1 = _f4_inv(p0, p1)
        for j in range(self.m):
            q0, q1 = _f4_mul(int(c0[j]), int(c1[j]), d0, d1)
            self.vals[j, k] = q0 | (q1 << 1)
        self._repack()

    def _outer(self, a0, a1, b0, b1):
        o0 = (a0[:, None] & b0[None, :]) ^ (a1[:, None] & b1[None, :])
        o1 = (a0[:, None] & b1[None, :]) ^ (a1[:, None] & b0[None, :]) ^ (a1[:, None] & b1[None, :])
        return o0[self.iu], o1[self.iu]

    def eval(self, x0, x1):
        o0, o1 = self._outer(x0, x1, x0, x1)
        return self._apply(o0, o1)

    def polar(self, a0, a1, b0, b1):
        p0, p1 = self._outer(a0, a1, b0, b1)
        q0, q1 = self._outer(b0, b1, a0, a1)
        return self._apply(p0 ^ q0, p1 ^ q1)

    def _apply(self, o0, o1):
        w0 = pack(o0.astype(np.uint8))
        w1 = pack(o1.astype(np.uint8))
        c0 = parity(self.A0 & w0[None, :]) ^ parity(self.A1 & w1[None, :])
        c1 = (parity(self.A0 & w1[None, :]) ^ parity(self.A1 & w0[None, :])
              ^ parity(self.A1 & w1[None, :]))
        return c0, c1


def _f4_mul(a0, a1, b0, b1):
    return (a0 & b0) ^ (a1 & b1), (a0 & b1) ^ (a1 & b0) ^ (a1 & b1)


def _f4_inv(a0, a1):
    for c0 in (0, 1):
        for c1 in (0, 1):
            if _f4_mul(a0, a1, c0, c1) == (1, 0):
                return c0, c1
    raise ZeroDivisionError


def rand_f4(rng, n):
    v = rng.integers(0, 4, size=n, dtype=np.uint8)
    return (v & 1).astype(np.uint8), ((v >> 1) & 1).astype(np.uint8)


def f4_bytes(*vecs):
    bits = np.concatenate([np.stack([v0, v1], axis=1).reshape(-1) for v0, v1 in vecs])
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="little").tobytes()
