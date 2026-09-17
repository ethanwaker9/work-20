import os
import numpy as np
from .. import field as F
from ..prg import h, xof

CB = 32
OPB = 16
RB = 16


class Interpolator:
    def __init__(self, xs):
        self.xs = np.asarray(xs, dtype=np.uint16)
        L = len(self.xs)
        diff = np.bitwise_xor(self.xs[:, None], self.xs[None, :])
        np.fill_diagonal(diff, 1)
        prod = np.ones(L, dtype=np.uint16)
        for j in range(L):
            prod = F.mul(prod, diff[:, j])
        self.w = np.array([F.inv(int(v)) for v in prod], dtype=np.uint16)

    def at(self, ys, x0):
        d = np.bitwise_xor(self.xs, np.uint16(x0))
        if np.any(d == 0):
            return int(ys[int(np.argmin(d))])
        allp = np.uint16(1)
        for v in d:
            allp = F.mul(allp, v)
        inv = np.array([F.inv(int(v)) for v in d], dtype=np.uint16)
        terms = F.mul(F.mul(np.asarray(ys, dtype=np.uint16), self.w), inv)
        return int(F.mul(allp, np.bitwise_xor.reduce(terms)))


def lagrange_at(xs, ys, x0):
    return Interpolator(xs).at(ys, x0)


class HS20:

    name = "HS20"

    def __init__(self, lam=128, m_open=2):
        self.lam = lam
        self.m = m_open
        self.lines = lam

    def setup_ring(self, rng, N):
        alphas = np.arange(1, N + 1, dtype=np.uint16)
        return [os.urandom(16) for _ in range(N)], alphas

    def pk_bytes(self, _):
        return CB + 2

    def sign(self, msg, alphas, signers, N, t):
        L, M = self.lines, self.m
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
        nonsign = [i for i in range(N) if i not in signers]
        parts = []
        for i in range(L):
            ys = rng.integers(0, 65536, size=(M, N), dtype=np.uint16)
            ops = rng.integers(0, 256, size=(N, OPB), dtype=np.uint8)
            rs = rng.integers(0, 256, size=(M, N, RB), dtype=np.uint8)
            coms = [h(bytes(ops[s]), dsep=b"tc") for s in range(N)]
            comblk = b"".join(coms)
            z = np.frombuffer(xof(h(msg, comblk, i, dsep=b"z"), 2 * M, b"zz"), dtype=np.uint16)
            xsn = [int(alphas[s]) for s in nonsign]
            itp = Interpolator([0] + xsn)
            for j in range(M):
                pts_y = [int(z[j])] + [int(ys[j, s]) for s in nonsign]
                for s in signers:
                    ys[j, s] = itp.at(pts_y, int(alphas[s]))
            g = np.zeros((M, N, CB), dtype=np.uint8)
            for j in range(M):
                for s in range(N):
                    d = h(int(ys[j, s]), bytes(ops[s]), bytes(rs[j, s]), dsep=b"G")
                    g[j, s] = np.frombuffer(d, dtype=np.uint8)
            J = int(np.frombuffer(xof(h(msg, comblk, g.tobytes(), dsep=b"J"), 4, b"jj"),
                                  dtype=np.uint32)[0]) % M
            parts.append(comblk)
            parts.append(g.tobytes())
            parts.append(ys[J].tobytes())
            parts.append(ops.tobytes())
            parts.append(rs[J].tobytes())
        return b"".join(parts)

    def verify(self, msg, alphas, sig, N, t):
        try:
            return self._verify(msg, alphas, sig, N, t)
        except Exception:
            return False

    def _verify(self, msg, alphas, sig, N, t):
        L, M = self.lines, self.m
        self._itp = None
        off = 0
        for i in range(L):
            comblk = sig[off:off + N * CB]; off += N * CB
            g = np.frombuffer(sig[off:off + M * N * CB], dtype=np.uint8).reshape(M, N, CB)
            off += M * N * CB
            yv = np.frombuffer(sig[off:off + 2 * N], dtype=np.uint16); off += 2 * N
            ops = np.frombuffer(sig[off:off + N * OPB], dtype=np.uint8).reshape(N, OPB)
            off += N * OPB
            rs = np.frombuffer(sig[off:off + N * RB], dtype=np.uint8).reshape(N, RB)
            off += N * RB
            z = np.frombuffer(xof(h(msg, comblk, i, dsep=b"z"), 2 * M, b"zz"), dtype=np.uint16)
            J = int(np.frombuffer(xof(h(msg, comblk, g.tobytes(), dsep=b"J"), 4, b"jj"),
                                  dtype=np.uint32)[0]) % M
            for s in range(N):
                if h(bytes(ops[s]), dsep=b"tc") != comblk[s * CB:(s + 1) * CB]:
                    return False
                d = h(int(yv[s]), bytes(ops[s]), bytes(rs[s]), dsep=b"G")
                if not np.array_equal(np.frombuffer(d, dtype=np.uint8), g[J, s]):
                    return False
            if self._itp is None:
                self._itp = Interpolator([0] + [int(alphas[s]) for s in range(N - t)])
            pts_y = [int(z[J])] + [int(yv[s]) for s in range(N - t)]
            for s in range(N - t, N):
                if self._itp.at(pts_y, int(alphas[s])) != int(yv[s]):
                    return False
        return off == len(sig)

    def sig_size(self, N):
        return self.lines * (N * CB + self.m * N * CB + 2 * N + N * OPB + N * RB)
