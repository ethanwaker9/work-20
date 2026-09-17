import numpy as np
from . import field as F
from .prg import xof


def prefix_xor_excl(v, axis=-1):
    c = np.bitwise_xor.accumulate(v, axis=axis)
    out = np.zeros_like(v)
    sl_dst = [slice(None)] * v.ndim
    sl_src = [slice(None)] * v.ndim
    sl_dst[axis] = slice(1, None)
    sl_src[axis] = slice(0, -1)
    out[tuple(sl_dst)] = c[tuple(sl_src)]
    return out


def digits_of(idx, d, b):
    out = np.zeros(d, dtype=np.int64)
    for i in range(d - 1, -1, -1):
        out[i] = idx % b
        idx //= b
    return out


class ThresholdMQRelation:
    def __init__(self, par, mq, Y, tag_mq=None, tags=None):
        self.par = par
        self.mq = mq
        self.Y = Y
        self.tag_mq = tag_mq
        self.tags = tags
        self.t = par.t
        self.d = par.d
        self.b = par.b
        self.n = par.mq_n
        self.m = par.mq_m
        self.N = par.ring_N
        self.full = self.b ** self.d
        self.bounded = self.full > self.N
        self.n_pairs = (self.t - 1) + (1 if self.bounded else 0)
        self.blk = 3 * self.d - 1
        self.off_x = 0
        self.off_u = self.t * self.n
        self.off_l = self.off_u + self.t * self.d * self.b
        self.wlen = self.off_l + self.n_pairs * self.blk
        self.sub = np.ones(self.wlen, dtype=np.uint8)
        self.sub[: self.off_u] = 2
        if self.bounded:
            self.bound_digits = digits_of(self.N, self.d, self.b)
            self.bound_onehot = np.zeros((self.d, self.b), dtype=np.uint16)
            for i in range(self.d):
                self.bound_onehot[i, self.bound_digits[i]] = 1
        self.D = par.D

    def n_gamma(self):
        extra = (self.tag_mq.m + self.t) if self.tag_mq is not None else 0
        return (extra + self.m + self.t
                + self.t * self.d
                + self.t * self.d * self.b
                + self.n_pairs * (2 * self.d + (self.d - 1) + 1))

    def build_witness(self, xs, idxs):
        w = np.zeros(self.wlen, dtype=np.uint8)
        order = np.argsort(idxs)
        idxs = [int(idxs[i]) for i in order]
        xs = [xs[i] for i in order]
        for k in range(self.t):
            w[self.off_x + k * self.n: self.off_x + (k + 1) * self.n] = xs[k]
            dg = digits_of(idxs[k], self.d, self.b)
            for i in range(self.d):
                w[self.off_u + (k * self.d + i) * self.b + dg[i]] = 1
        chain = [digits_of(idxs[k], self.d, self.b) for k in range(self.t)]
        if self.bounded:
            chain.append(self.bound_digits)
        for p in range(self.n_pairs):
            lo, hi = chain[p], chain[p + 1]
            eq = np.array([1 if lo[i] == hi[i] else 0 for i in range(self.d)], dtype=np.uint8)
            gt = np.array([1 if hi[i] > lo[i] else 0 for i in range(self.d)], dtype=np.uint8)
            E = np.zeros(self.d - 1, dtype=np.uint8)
            acc = 1
            for i in range(self.d - 1):
                acc = acc & int(eq[i])
                E[i] = acc
            base = self.off_l + p * self.blk
            w[base: base + self.d] = eq
            w[base + self.d: base + 2 * self.d] = gt
            w[base + 2 * self.d: base + self.blk] = E
        return w

    def precompute(self, seed):
        need = self.n_gamma()
        raw = xof(seed, 2 * need + 16, b"mpcrand")
        g = np.frombuffer(raw[: 2 * need], dtype=np.uint16).copy()
        g[g == 0] = 1
        p = 0
        gam = g[p: p + self.m]; p += self.m
        theta = g[p: p + self.t]; p += self.t
        mu = g[p: p + self.t * self.d].reshape(self.t, self.d); p += self.t * self.d
        nu = g[p: p + self.t * self.d * self.b].reshape(self.t, self.d, self.b)
        p += self.t * self.d * self.b
        nl = self.n_pairs
        ca = g[p: p + nl * self.d].reshape(nl, self.d); p += nl * self.d
        cb = g[p: p + nl * self.d].reshape(nl, self.d); p += nl * self.d
        cc = g[p: p + nl * (self.d - 1)].reshape(nl, self.d - 1); p += nl * (self.d - 1)
        cf = g[p: p + nl]; p += nl
        tagctx = None
        if self.tag_mq is not None:
            mt = self.tag_mq.m
            delta = g[p: p + mt]; p += mt
            theta2 = g[p: p + self.t]; p += self.t
            Bg, cg = self.tag_mq.combine(delta)
            tconst = np.bitwise_xor.reduce(F.mul(self.tags, delta[None, :]), axis=1)
            tagctx = (Bg, cg, theta2, tconst)
        Ag, bg = self.mq.combine(gam)
        ch = np.zeros(self.full, dtype=np.uint16)
        ch[: self.N] = np.bitwise_xor.reduce(F.mul(self.Y, gam[None, :]), axis=1)
        return dict(tag=tagctx, gam=gam, theta=theta, mu=mu, nu=nu, ca=ca, cb=cb, cc=cc, cf=cf,
                    Ag=Ag, bg=bg, C=ch.reshape((self.b,) * self.d))

    def _select(self, ctx, U):
        T = ctx["C"]
        acc = np.broadcast_to(T, (self.t,) + T.shape).copy()
        for i in range(self.d - 1, -1, -1):
            uu = U[:, i, :]
            shp = (self.t,) + (1,) * (acc.ndim - 2) + (self.b,)
            acc = np.bitwise_xor.reduce(F.mul(acc, uu.reshape(shp)), axis=-1)
        return acc

    def eval(self, ctx, ws):
        t, d, b, n = self.t, self.d, self.b, self.n
        X = ws[self.off_x: self.off_u].reshape(t, n)
        U = ws[self.off_u: self.off_l].reshape(t, d, b)
        L = ws[self.off_l:].reshape(self.n_pairs, self.blk) if self.n_pairs else None
        O = self.mq.tri(X)
        quad = np.bitwise_xor.reduce(F.mul(ctx["Ag"][None, :], O), axis=1)
        lin = np.bitwise_xor.reduce(F.mul(X, ctx["bg"][None, :]), axis=1)
        sel = self._select(ctx, U)
        total = int(np.bitwise_xor.reduce(F.mul(ctx["theta"], quad ^ lin ^ sel)))
        if ctx["tag"] is not None:
            Bg, cg, theta2, tconst = ctx["tag"]
            tq = np.bitwise_xor.reduce(F.mul(Bg[None, :], O), axis=1)
            tl = np.bitwise_xor.reduce(F.mul(X, cg[None, :]), axis=1)
            total ^= int(np.bitwise_xor.reduce(F.mul(theta2, tq ^ tl ^ tconst)))
        ones = np.bitwise_xor.reduce(U, axis=2) ^ np.uint16(1)
        total ^= int(np.bitwise_xor.reduce(F.mul(ctx["mu"], ones).reshape(-1)))
        P = prefix_xor_excl(U, axis=2)
        total ^= int(np.bitwise_xor.reduce(F.mul(ctx["nu"], F.mul(U, P)).reshape(-1)))
        if self.n_pairs:
            eq = L[:, :d]
            gt = L[:, d: 2 * d]
            E = L[:, 2 * d:]
            Ulo = U[: self.n_pairs]
            if self.bounded:
                Uhi = np.concatenate([U[1:], self.bound_onehot[None, :, :]], axis=0)
            else:
                Uhi = U[1:]
            Plo = prefix_xor_excl(Ulo, axis=2)
            eq_t = np.bitwise_xor.reduce(F.mul(Ulo, Uhi), axis=2)
            gt_t = np.bitwise_xor.reduce(F.mul(Uhi, Plo), axis=2)
            total ^= int(np.bitwise_xor.reduce(F.mul(ctx["ca"], eq ^ eq_t).reshape(-1)))
            total ^= int(np.bitwise_xor.reduce(F.mul(ctx["cb"], gt ^ gt_t).reshape(-1)))
            if d >= 2:
                pred = np.concatenate([np.ones((self.n_pairs, 1), dtype=np.uint16), E[:, :-1]], axis=1)
                Eexp = F.mul(pred, eq[:, : d - 1])
                total ^= int(np.bitwise_xor.reduce(F.mul(ctx["cc"], E ^ Eexp).reshape(-1)))
            Efull = np.concatenate([np.ones((self.n_pairs, 1), dtype=np.uint16), E], axis=1)
            lex = np.bitwise_xor.reduce(F.mul(gt, Efull), axis=1) ^ np.uint16(1)
            total ^= int(np.bitwise_xor.reduce(F.mul(ctx["cf"], lex)))
        return np.uint16(total)
