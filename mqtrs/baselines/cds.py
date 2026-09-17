import math
import os
import numpy as np
from .f4 import F4MQ, rand_f4
from .multivariate_trs import enc, dec, enc_len, zero, xo, rounds_for
from ..prg import h, xof

CB = 32
FB = 16
MOD = (1 << 128) | (1 << 7) | (1 << 2) | (1 << 1) | 1


def gmul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> 128:
            a ^= MOD
    return r


def gpow(a, e):
    r = 1
    while e:
        if e & 1:
            r = gmul(r, a)
        a = gmul(a, a)
        e >>= 1
    return r


def ginv(a):
    return gpow(a, (1 << 128) - 2)


class Interp128:
    def __init__(self, xs):
        self.xs = list(xs)
        L = len(self.xs)
        w = []
        for i in range(L):
            d = 1
            for j in range(L):
                if i != j:
                    d = gmul(d, self.xs[i] ^ self.xs[j])
            w.append(d)
        self.w = batch_inv(w)

    def at(self, ys, x0):
        d = [x ^ x0 for x in self.xs]
        for i, v in enumerate(d):
            if v == 0:
                return ys[i]
        allp = 1
        for v in d:
            allp = gmul(allp, v)
        di = batch_inv(d)
        acc = 0
        for i in range(len(d)):
            acc ^= gmul(ys[i], gmul(self.w[i], di[i]))
        return gmul(allp, acc)


def batch_inv(vals):
    pref = [1]
    for v in vals:
        pref.append(gmul(pref[-1], v))
    inv = ginv(pref[-1])
    out = [0] * len(vals)
    for i in range(len(vals) - 1, -1, -1):
        out[i] = gmul(inv, pref[i])
        inv = gmul(inv, vals[i])
    return out


def interp_at(xs, ys, x0):
    return Interp128(xs).at(ys, x0)


class CDSMQ:

    name = "CDS-MQ"

    def __init__(self, n=88, m=88, lam=128):
        self.n, self.m, self.lam = n, m, lam
        self.rounds = rounds_for(2.0 / 3.0, lam)
        self.Q = F4MQ(b"cds-common-instance", n, m)

    def setup_ring(self, rng, N):
        sks, pks = [], []
        for _ in range(N):
            x = rand_f4(rng, self.n)
            sks.append(x)
            pks.append(self.Q.eval(*x))
        return sks, pks

    def pk_bytes(self, _):
        return (2 * self.m + 7) // 8

    def _chal(self, ci):
        raw = xof(ci.to_bytes(16, "little"), self.rounds, b"cdsch")
        return [b % 3 for b in raw]

    def _real(self, x, seed):
        rng = np.random.default_rng(seed)
        n, m = self.n, self.m
        cs, data = [], []
        for rd in range(self.rounds):
            r0 = rand_f4(rng, n); t0 = rand_f4(rng, n); e0 = rand_f4(rng, m)
            r1 = xo(x, r0)
            t1 = xo(r0, t0)
            e1 = xo(self.Q.eval(*r0), e0)
            g = self.Q.polar(*t0, *r1)
            c0 = h(enc(r1, xo(g, e0)), dsep=b"k0")
            c1 = h(enc(t0, e0), dsep=b"k1")
            c2 = h(enc(t1, e1), dsep=b"k2")
            cs.append(c0 + c1 + c2)
            data.append((c0, c1, c2, r0, r1, t0, t1, e0, e1))
        return h(b"".join(cs), dsep=b"hi"), data

    def _respond(self, data, ch):
        out = []
        for rd, (c0, c1, c2, r0, r1, t0, t1, e0, e1) in enumerate(data):
            if ch[rd] == 0:
                out.append(c0 + enc(r0, t1, e1))
            elif ch[rd] == 1:
                out.append(c1 + enc(r1, t1, e1))
            else:
                out.append(c2 + enc(r1, t0, e0))
        return b"".join(out)

    def _sim(self, v, ch, rng):
        n, m = self.n, self.m
        blocks, cs = [], []
        for rd in range(self.rounds):
            if ch[rd] == 0:
                r0 = rand_f4(rng, n); t1 = rand_f4(rng, n); e1 = rand_f4(rng, m)
                c1 = h(enc(xo(r0, t1), xo(self.Q.eval(*r0), e1)), dsep=b"k1")
                c2 = h(enc(t1, e1), dsep=b"k2")
                c0 = os.urandom(CB)
                blocks.append(c0 + enc(r0, t1, e1))
            elif ch[rd] == 1:
                r1 = rand_f4(rng, n); t1 = rand_f4(rng, n); e1 = rand_f4(rng, m)
                ev = self.Q.eval(*r1)
                val = xo(xo((v[0] ^ ev[0], v[1] ^ ev[1]), self.Q.polar(*t1, *r1)), e1)
                c0 = h(enc(r1, val), dsep=b"k0")
                c2 = h(enc(t1, e1), dsep=b"k2")
                c1 = os.urandom(CB)
                blocks.append(c1 + enc(r1, t1, e1))
            else:
                r1 = rand_f4(rng, n); t0 = rand_f4(rng, n); e0 = rand_f4(rng, m)
                c0 = h(enc(r1, xo(self.Q.polar(*t0, *r1), e0)), dsep=b"k0")
                c1 = h(enc(t0, e0), dsep=b"k1")
                c2 = os.urandom(CB)
                blocks.append(c2 + enc(r1, t0, e0))
            cs.append(c0 + c1 + c2)
        return h(b"".join(cs), dsep=b"hi"), b"".join(blocks)

    def sign(self, msg, pks, sks, signers, t):
        N = len(pks)
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
        nonsign = [i for i in range(N) if i not in signers]
        cvals = [0] * N
        hs = [None] * N
        blocks = [None] * N
        data = {}
        for i in nonsign:
            cvals[i] = int.from_bytes(os.urandom(16), "little")
            hs[i], blocks[i] = self._sim(pks[i], self._chal(cvals[i]), rng)
        for i in signers:
            seed = int.from_bytes(os.urandom(8), "little")
            hs[i], data[i] = self._real(sks[i], seed)
        cstar = int.from_bytes(h(msg, b"".join(hs), dsep=b"cds"), "little") % (1 << 128)
        itp = Interp128([0] + [i + 1 for i in nonsign])
        ys = [cstar] + [cvals[i] for i in nonsign]
        for i in signers:
            cvals[i] = itp.at(ys, i + 1)
            blocks[i] = self._respond(data[i], self._chal(cvals[i]))
        out = [b"".join(c.to_bytes(16, "little") for c in cvals)] + blocks
        return b"".join(out)

    def unit_len(self):
        return self.rounds * (CB + enc_len(self.n, self.n, self.m))

    def verify(self, msg, pks, sig, t):
        try:
            return self._verify(msg, pks, sig, t)
        except Exception:
            return False

    def _verify(self, msg, pks, sig, t):
        N = len(pks)
        ul = self.unit_len()
        if len(sig) != N * FB + N * ul:
            return False
        cvals = [int.from_bytes(sig[i * FB:(i + 1) * FB], "little") for i in range(N)]
        base = N * FB
        n, m = self.n, self.m
        vl = enc_len(n, n, m)
        hs = []
        for i in range(N):
            ch = self._chal(cvals[i])
            off = base + i * ul
            cs = []
            for rd in range(self.rounds):
                miss = sig[off:off + CB]; off += CB
                a, b, c = dec(sig[off:off + vl], [n, n, m]); off += vl
                if ch[rd] == 0:
                    c1 = h(enc(xo(a, b), xo(self.Q.eval(*a), c)), dsep=b"k1")
                    c2 = h(enc(b, c), dsep=b"k2")
                    cs.append(miss + c1 + c2)
                elif ch[rd] == 1:
                    ev = self.Q.eval(*a)
                    val = xo(xo((pks[i][0] ^ ev[0], pks[i][1] ^ ev[1]),
                                self.Q.polar(*b, *a)), c)
                    c0 = h(enc(a, val), dsep=b"k0")
                    c2 = h(enc(b, c), dsep=b"k2")
                    cs.append(c0 + miss + c2)
                else:
                    c0 = h(enc(a, xo(self.Q.polar(*b, *a), c)), dsep=b"k0")
                    c1 = h(enc(b, c), dsep=b"k1")
                    cs.append(c0 + c1 + miss)
            hs.append(h(b"".join(cs), dsep=b"hi"))
        cstar = int.from_bytes(h(msg, b"".join(hs), dsep=b"cds"), "little") % (1 << 128)
        itp = Interp128([i + 1 for i in range(N - t + 1)])
        ys = [cvals[i] for i in range(N - t + 1)]
        if itp.at(ys, 0) != cstar:
            return False
        for i in range(N - t + 1, N):
            if itp.at(ys, i + 1) != cvals[i]:
                return False
        return True
