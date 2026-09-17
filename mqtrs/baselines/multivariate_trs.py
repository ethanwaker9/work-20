import math
import os
import numpy as np
from .f4 import F4MQ, rand_f4
from ..prg import h, xof

CB = 32


def rounds_for(eps, lam=128):
    return int(math.ceil(lam / -math.log2(eps)))


def enc(*vecs):
    bits = np.concatenate([np.stack([v0, v1], axis=1).reshape(-1) for v0, v1 in vecs])
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="little").tobytes()


def enc_len(*lens):
    return (2 * sum(lens) + 7) // 8


def dec(buf, lens):
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")
    out, p = [], 0
    for L in lens:
        chunk = bits[p:p + 2 * L].reshape(L, 2)
        out.append((chunk[:, 0].copy(), chunk[:, 1].copy()))
        p += 2 * L
    return out


def perm_enc(perm):
    return np.asarray(perm, dtype=np.uint32).tobytes()


def perm_len(N):
    return (N * max(1, (N - 1).bit_length()) + 7) // 8


def scal(a, v):
    a0, a1 = a & 1, (a >> 1) & 1
    if (a0, a1) == (0, 0):
        return (np.zeros_like(v[0]), np.zeros_like(v[1]))
    if (a0, a1) == (1, 0):
        return v
    if (a0, a1) == (0, 1):
        return (v[1], v[0] ^ v[1])
    return (v[0] ^ v[1], v[0])


def zero(n):
    return (np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8))


def xo(a, b):
    return (a[0] ^ b[0], a[1] ^ b[1])


class PBB13:
    name = "PBB13"
    eps = 0.75

    def __init__(self, n=88, m=88, lam=128):
        self.n, self.m, self.lam = n, m, lam
        self.rounds = rounds_for(self.eps, lam)

    def setup_ring(self, rng, N):
        maps, roots = [], []
        for i in range(N):
            P = F4MQ(bytes([i & 255, (i >> 8) & 255]) + b"pbb", self.n, self.m)
            s = rand_f4(rng, self.n)
            P.plant_root(*s)
            maps.append(P)
            roots.append(s)
        return roots, maps

    def pk_bytes(self, maps):
        return maps[0].m * maps[0].L * 2 // 8 + 0

    def sign(self, msg, maps, roots, signers, t):
        N, n, m, R = len(maps), self.n, self.m, self.rounds
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
        store, allC = [], []
        for rd in range(R):
            perm = rng.permutation(N)
            cs = [[None] * N for _ in range(5)]
            keep = []
            for i in range(N):
                P = maps[i]
                r0 = rand_f4(rng, n); t0 = rand_f4(rng, n); e0 = rand_f4(rng, m)
                x = roots[i] if i in signers else zero(n)
                r1 = xo(x, r0)
                t1 = xo(r0, t0)
                e1 = xo(P.eval(*r0), e0)
                g = P.polar(*t0, *r1)
                cs[0][i] = h(enc(r1, xo(g, e0)), dsep=b"c0")
                cs[1][i] = h(enc(t0, e0), dsep=b"c1")
                cs[2][i] = h(enc(t1, e1), dsep=b"c2")
                cs[3][i] = h(enc(r0), dsep=b"c34")
                cs[4][i] = h(enc(r1), dsep=b"c34")
                keep.append((r0, r1, t0, t1, e0, e1))
            C = self._master(cs, perm, N)
            allC.append(C)
            store.append((keep, perm, cs))
        COM = h(b"".join(b"".join(c) for c in allC), dsep=b"COM")
        chal = xof(h(msg, COM, dsep=b"ch"), R, b"chal")
        parts = [COM]
        for rd in range(R):
            ch = chal[rd] & 3
            keep, perm, cs = store[rd]
            C = allC[rd]
            if ch == 0:
                parts += [C[0], C[4], perm_enc(perm)]
                parts.append(b"".join(enc(k[0], k[3], k[5]) for k in keep))
            elif ch == 1:
                parts += [C[1], C[3], C[4]]
                parts.append(b"".join(enc(k[1], k[3], k[5]) for k in keep))
            elif ch == 2:
                parts += [C[2], C[3], perm_enc(perm)]
                parts.append(b"".join(enc(k[1], k[2], k[4]) for k in keep))
            else:
                parts += [C[0], C[1], C[2]]
                parts.append(b"".join(cs[3][perm[i]] for i in range(N)))
                parts.append(b"".join(cs[4][perm[i]] for i in range(N)))
        return b"".join(parts)

    def _master(self, cs, perm, N):
        pb = perm_enc(perm)
        return [h(b"".join(cs[0]), dsep=b"C0"),
                h(pb, b"".join(cs[1]), dsep=b"C1"),
                h(b"".join(cs[2]), dsep=b"C2"),
                h(b"".join([cs[3][perm[i]] for i in range(N)]), dsep=b"C3"),
                h(b"".join([cs[4][perm[i]] for i in range(N)]), dsep=b"C4")]

    def sig_size(self, N):
        R = self.rounds
        n, m = self.n, self.m
        body = 0
        for ch in range(4):
            if ch == 0:
                s = 2 * CB + 4 * N + N * enc_len(n, n, m)
            elif ch == 1:
                s = 3 * CB + N * enc_len(n, n, m)
            elif ch == 2:
                s = 2 * CB + 4 * N + N * enc_len(n, n, m)
            else:
                s = 3 * CB + 2 * N * CB
            body += s
        return CB + R * body // 4

    def verify(self, msg, maps, sig, t):
        try:
            return self._verify(msg, maps, sig, t)
        except Exception:
            return False

    def _verify(self, msg, maps, sig, t):
        N, n, m, R = len(maps), self.n, self.m, self.rounds
        COM = sig[:CB]
        off = CB
        chal = xof(h(msg, COM, dsep=b"ch"), R, b"chal")
        allC = []
        vlen = enc_len(n, n, m)
        for rd in range(R):
            ch = chal[rd] & 3
            cs = [[None] * N for _ in range(5)]
            if ch == 0:
                C0 = sig[off:off + CB]; off += CB
                C4 = sig[off:off + CB]; off += CB
                perm = np.frombuffer(sig[off:off + 4 * N], dtype=np.uint32); off += 4 * N
                if perm.size != N or int(perm.max()) >= N:
                    return False
                for i in range(N):
                    r0, t1, e1 = dec(sig[off:off + vlen], [n, n, m]); off += vlen
                    P = maps[i]
                    cs[1][i] = h(enc(xo(r0, t1), xo(P.eval(*r0), e1)), dsep=b"c1")
                    cs[2][i] = h(enc(t1, e1), dsep=b"c2")
                    cs[3][i] = h(enc(r0), dsep=b"c34")
                pb = perm_enc(perm)
                C = [C0, h(pb, b"".join(cs[1]), dsep=b"C1"), h(b"".join(cs[2]), dsep=b"C2"),
                     h(b"".join([cs[3][perm[i]] for i in range(N)]), dsep=b"C3"), C4]
            elif ch == 1:
                C1 = sig[off:off + CB]; off += CB
                C3 = sig[off:off + CB]; off += CB
                C4 = sig[off:off + CB]; off += CB
                for i in range(N):
                    r1, t1, e1 = dec(sig[off:off + vlen], [n, n, m]); off += vlen
                    P = maps[i]
                    val = xo(xo(P.eval(*r1), P.polar(*t1, *r1)), e1)
                    cs[0][i] = h(enc(r1, val), dsep=b"c0")
                    cs[2][i] = h(enc(t1, e1), dsep=b"c2")
                C = [h(b"".join(cs[0]), dsep=b"C0"), C1, h(b"".join(cs[2]), dsep=b"C2"), C3, C4]
            elif ch == 2:
                C2 = sig[off:off + CB]; off += CB
                C3 = sig[off:off + CB]; off += CB
                perm = np.frombuffer(sig[off:off + 4 * N], dtype=np.uint32); off += 4 * N
                if perm.size != N or int(perm.max()) >= N:
                    return False
                for i in range(N):
                    r1, t0, e0 = dec(sig[off:off + vlen], [n, n, m]); off += vlen
                    P = maps[i]
                    cs[0][i] = h(enc(r1, xo(P.polar(*t0, *r1), e0)), dsep=b"c0")
                    cs[1][i] = h(enc(t0, e0), dsep=b"c1")
                    cs[4][i] = h(enc(r1), dsep=b"c34")
                pb = perm_enc(perm)
                C = [h(b"".join(cs[0]), dsep=b"C0"), h(pb, b"".join(cs[1]), dsep=b"C1"), C2, C3,
                     h(b"".join([cs[4][perm[i]] for i in range(N)]), dsep=b"C4")]
            else:
                C0 = sig[off:off + CB]; off += CB
                C1 = sig[off:off + CB]; off += CB
                C2 = sig[off:off + CB]; off += CB
                l3 = [sig[off + k * CB:off + (k + 1) * CB] for k in range(N)]; off += N * CB
                l4 = [sig[off + k * CB:off + (k + 1) * CB] for k in range(N)]; off += N * CB
                if sum(1 for k in range(N) if l3[k] != l4[k]) < t:
                    return False
                C = [C0, C1, C2, h(b"".join(l3), dsep=b"C3"), h(b"".join(l4), dsep=b"C4")]
            allC.append(C)
        return h(b"".join(b"".join(c) for c in allC), dsep=b"COM") == COM and off == len(sig)


class MTRS5(PBB13):

    name = "MTRS5"

    def sign(self, msg, maps, roots, signers, t):
        N, n, m, R = len(maps), self.n, self.m, self.rounds
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
        store, allC = [], []
        for rd in range(R):
            perm = rng.permutation(N)
            cs = [[None] * N for _ in range(4)]
            keep = []
            for i in range(N):
                P = maps[i]
                r0 = rand_f4(rng, n); t0 = rand_f4(rng, n); e0 = rand_f4(rng, m)
                x = roots[i] if i in signers else zero(n)
                r1 = xo(x, r0)
                g = P.polar(*t0, *r1)
                cs[0][i] = h(enc(r0, t0, e0), dsep=b"d0")
                cs[1][i] = h(enc(r1, xo(g, e0)), dsep=b"d1")
                cs[2][i] = h(enc(r0), dsep=b"d23")
                cs[3][i] = h(enc(r1), dsep=b"d23")
                keep.append((r0, r1, t0, e0, P.eval(*r0)))
            C = [h(b"".join(cs[0]), dsep=b"E0"), h(b"".join(cs[1]), dsep=b"E1"),
                 h(b"".join([cs[2][perm[i]] for i in range(N)]), dsep=b"E2"),
                 h(b"".join([cs[3][perm[i]] for i in range(N)]), dsep=b"E3")]
            allC.append(C)
            store.append((keep, perm, cs))
        COM = h(b"".join(b"".join(c) for c in allC), dsep=b"COMb")
        alpha = xof(h(msg, COM, dsep=b"al"), R, b"alpha")
        te = []
        for rd in range(R):
            a = alpha[rd] & 3
            blk = [enc(xo(scal(a, k[0]), k[2]), xo(scal(a, k[4]), k[3])) for k in store[rd][0]]
            te.append(b"".join(blk))
        chal = xof(h(msg, COM, b"".join(te), dsep=b"ch2"), R, b"chal2")
        parts = [COM] + te
        for rd in range(R):
            ch = chal[rd] % 3
            keep, perm, cs = store[rd]
            C = allC[rd]
            if ch == 0:
                parts += [C[1], C[2], C[3]]
                parts.append(b"".join(enc(k[0]) for k in keep))
            elif ch == 1:
                parts += [C[0], C[2], C[3]]
                parts.append(b"".join(enc(k[1]) for k in keep))
            else:
                parts += [C[0], C[1]]
                parts.append(b"".join(cs[2][perm[i]] for i in range(N)))
                parts.append(b"".join(cs[3][perm[i]] for i in range(N)))
        return b"".join(parts)

    def verify(self, msg, maps, sig, t):
        try:
            return self._verify(msg, maps, sig, t)
        except Exception:
            return False

    def _verify(self, msg, maps, sig, t):
        N, n, m, R = len(maps), self.n, self.m, self.rounds
        COM = sig[:CB]
        off = CB
        alpha = xof(h(msg, COM, dsep=b"al"), R, b"alpha")
        telen = N * enc_len(n, m)
        tes = []
        for rd in range(R):
            tes.append(sig[off:off + telen]); off += telen
        chal = xof(h(msg, COM, b"".join(tes), dsep=b"ch2"), R, b"chal2")
        allC = []
        vlen = enc_len(n)
        ulen = enc_len(n, m)
        for rd in range(R):
            a = alpha[rd] & 3
            teblk = tes[rd]
            ch = chal[rd] % 3
            cs = [[None] * N for _ in range(4)]
            if ch == 0:
                C1 = sig[off:off + CB]; off += CB
                C2 = sig[off:off + CB]; off += CB
                C3 = sig[off:off + CB]; off += CB
                for i in range(N):
                    t1, e1 = dec(teblk[i * ulen:(i + 1) * ulen], [n, m])
                    (r0,) = dec(sig[off:off + vlen], [n]); off += vlen
                    P = maps[i]
                    cs[0][i] = h(enc(r0, xo(scal(a, r0), t1),
                                     xo(scal(a, P.eval(*r0)), e1)), dsep=b"d0")
                C = [h(b"".join(cs[0]), dsep=b"E0"), C1, C2, C3]
            elif ch == 1:
                C0 = sig[off:off + CB]; off += CB
                C2 = sig[off:off + CB]; off += CB
                C3 = sig[off:off + CB]; off += CB
                for i in range(N):
                    t1, e1 = dec(teblk[i * ulen:(i + 1) * ulen], [n, m])
                    (r1,) = dec(sig[off:off + vlen], [n]); off += vlen
                    P = maps[i]
                    val = xo(xo(scal(a, P.eval(*r1)), P.polar(*t1, *r1)), e1)
                    cs[1][i] = h(enc(r1, val), dsep=b"d1")
                C = [C0, h(b"".join(cs[1]), dsep=b"E1"), C2, C3]
            else:
                C0 = sig[off:off + CB]; off += CB
                C1 = sig[off:off + CB]; off += CB
                l2 = [sig[off + k * CB:off + (k + 1) * CB] for k in range(N)]; off += N * CB
                l3 = [sig[off + k * CB:off + (k + 1) * CB] for k in range(N)]; off += N * CB
                if sum(1 for k in range(N) if l2[k] != l3[k]) < t:
                    return False
                C = [C0, C1, h(b"".join(l2), dsep=b"E2"), h(b"".join(l3), dsep=b"E3")]
            allC.append(C)
        return h(b"".join(b"".join(c) for c in allC), dsep=b"COMb") == COM and off == len(sig)
