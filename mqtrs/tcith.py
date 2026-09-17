import os
import numpy as np
from . import field as F
from .prg import SeedTree, h, xof, LAMBDA_BYTES, DIGEST_BYTES
from .pack import pack_bits, unpack_bits, packed_len


class Engine:
    def __init__(self, par, rel):
        self.par = par
        self.rel = rel
        self.Np = par.n_party
        self.D = par.D
        self.tau = par.tau
        self.rho = par.rho
        self.e = np.arange(1, self.Np + 1, dtype=np.uint16)
        self.einv = np.array([F.inv(int(v)) for v in self.e], dtype=np.uint16)
        self.widths = rel.sub.copy()
        self.dwlen = packed_len(self.widths)
        self.pathlen = int(np.log2(self.Np))
        self.xi = np.arange(1, self.D + 1, dtype=np.uint16)
        V = np.zeros((self.D, self.D), dtype=np.uint16)
        for a in range(self.D):
            for c in range(self.D):
                V[a, c] = F.pow_(int(self.xi[a]), c + 1)
        self.Vinv = gf_inverse(V)
        self.lut_div = np.stack([F.smul(int(self.einv[j]), F.GF4) for j in range(self.Np)])

    def leaf_shares(self, leaves, salt, e_idx):
        wl = self.rel.wlen
        need = wl + 2 * (self.D - 1) * self.rho
        out = np.zeros((self.Np, wl), dtype=np.uint8)
        hints = np.zeros((self.Np, self.rho, self.D - 1), dtype=np.uint16)
        mask = (np.uint8(1) << self.widths) - np.uint8(1)
        for j in range(self.Np):
            raw = xof(leaves[j] + salt + bytes([e_idx & 255, (e_idx >> 8) & 255]), need, b"lf")
            out[j] = np.frombuffer(raw[:wl], dtype=np.uint8) & mask
            hb = np.frombuffer(raw[wl:], dtype=np.uint16).reshape(self.rho, self.D - 1)
            hints[j] = hb
        return out, hints

    def hint_coeffs(self, hints):
        S = np.zeros((self.rho, self.D - 1), dtype=np.uint16)
        T = np.zeros((self.rho, self.D - 1), dtype=np.uint16)
        for j in range(self.Np):
            S ^= hints[j]
            T ^= F.smul(int(self.e[j]), hints[j])
        coef = np.zeros((self.rho, self.D + 1), dtype=np.uint16)
        for c in range(1, self.D):
            coef[:, c + 1] ^= S[:, c - 1]
            coef[:, c] ^= T[:, c - 1]
        return coef

    def hint_share(self, hints, istar):
        ei = int(self.e[istar])
        acc = np.zeros((self.rho, self.D - 1), dtype=np.uint16)
        for j in range(self.Np):
            if j == istar:
                continue
            acc ^= F.smul(ei ^ int(self.e[j]), hints[j])
        tot = np.zeros(self.rho, dtype=np.uint16)
        for c in range(1, self.D):
            tot ^= F.smul(F.pow_(ei, c), acc[:, c - 1])
        return tot

    def commit_leaves(self, leaves, salt, e_idx):
        return [h(salt, e_idx, j, leaves[j], dsep=b"leaf") for j in range(self.Np)]


def gf_inverse(M):
    n = M.shape[0]
    A = np.concatenate([M.copy(), np.eye(n, dtype=np.uint16)], axis=1)
    for c in range(n):
        piv = c
        while A[piv, c] == 0:
            piv += 1
        A[[c, piv]] = A[[piv, c]]
        A[c] = F.smul(F.inv(int(A[c, c])), A[c])
        for r2 in range(n):
            if r2 != c and A[r2, c]:
                A[r2] ^= F.smul(int(A[r2, c]), A[c])
    return A[:, n:]


def _poly_eval(coef, x):
    acc = np.zeros(coef.shape[0], dtype=np.uint16)
    for c in range(coef.shape[1] - 1, -1, -1):
        acc = F.smul(int(x), acc) ^ coef[:, c]
    return acc


def _solve(Vinv, vals):
    out = np.zeros(len(vals), dtype=np.uint16)
    for i in range(len(vals)):
        out[i] = int(np.bitwise_xor.reduce(F.mul(Vinv[i], vals)))
    return out


def prove(eng, rel, w_idx, msg, ctxbytes):
    par = eng.par
    salt = os.urandom(2 * LAMBDA_BYTES)
    Np, tau, rho, D = eng.Np, eng.tau, eng.rho, eng.D
    trees, Ss, Hs, dws, coms = [], [], [], [], []
    for e in range(tau):
        rseed = os.urandom(LAMBDA_BYTES)
        tree = SeedTree(rseed, salt, e, Np)
        leaves = tree.leaves()
        S, hints = eng.leaf_shares(leaves, salt, e)
        acc = np.bitwise_xor.reduce(S, axis=0)
        dw = np.bitwise_xor(w_idx, acc)
        trees.append(tree); Ss.append(S); Hs.append(hints); dws.append(dw)
        coms.append(eng.commit_leaves(leaves, salt, e))
    dwp = [pack_bits(dw, eng.widths) for dw in dws]
    h1 = h(salt, msg, ctxbytes, b"".join(b"".join(c) for c in coms), b"".join(dwp), dsep=b"h1")
    wgf = F.GF4[w_idx.astype(np.int64)]
    A = np.zeros((tau, rho, D + 1), dtype=np.uint16)
    for e in range(tau):
        S = Ss[e]
        rows = np.arange(Np)[:, None]
        rvec = np.bitwise_xor.reduce(eng.lut_div[rows, S], axis=0)
        rvec ^= eng.lut_div[0][dws[e]]
        hc = eng.hint_coeffs(Hs[e])
        shares = [wgf ^ F.smul(int(eng.xi[c]), rvec) for c in range(D)]
        for r in range(rho):
            ctx = rel.precompute(h1 + bytes([e & 255, (e >> 8) & 255, r & 255]))
            vals = np.zeros(D, dtype=np.uint16)
            for c in range(D):
                g = rel.eval(ctx, shares[c])
                pv = _poly_eval(hc[r:r + 1], int(eng.xi[c]))[0]
                vals[c] = g ^ pv
            A[e, r, 1:] = _solve(eng.Vinv, vals)
    h2 = h(h1, A, dsep=b"h2")
    idx_raw = xof(h2, 4 * tau, b"open")
    istars = [int(np.frombuffer(idx_raw[4 * e:4 * e + 4], dtype=np.uint32)[0]) % Np for e in range(tau)]
    parts = [salt, h2]
    for e in range(tau):
        parts.append(dwp[e])
        for nd in trees[e].path(istars[e]):
            parts.append(nd)
        parts.append(coms[e][istars[e]])
        parts.append(A[e, :, 2:].tobytes())
    return b"".join(parts), dict(h1=h1, istars=istars)


def verify(eng, rel, sig, msg, ctxbytes):
    par = eng.par
    Np, tau, rho, D = eng.Np, eng.tau, eng.rho, eng.D
    off = 0
    salt = sig[off:off + 2 * LAMBDA_BYTES]; off += 2 * LAMBDA_BYTES
    h2 = sig[off:off + DIGEST_BYTES]; off += DIGEST_BYTES
    idx_raw = xof(h2, 4 * tau, b"open")
    istars = [int(np.frombuffer(idx_raw[4 * e:4 * e + 4], dtype=np.uint32)[0]) % Np for e in range(tau)]
    dwp, paths, comI, Atop = [], [], [], []
    for e in range(tau):
        dwp.append(sig[off:off + eng.dwlen]); off += eng.dwlen
        p = []
        for _ in range(eng.pathlen):
            p.append(sig[off:off + LAMBDA_BYTES]); off += LAMBDA_BYTES
        paths.append(p)
        comI.append(sig[off:off + DIGEST_BYTES]); off += DIGEST_BYTES
        nb = rho * (D - 1) * 2
        Atop.append(np.frombuffer(sig[off:off + nb], dtype=np.uint16).reshape(rho, D - 1).copy())
        off += nb
    if off != len(sig):
        return False
    coms, leafsets, hintsets = [], [], []
    for e in range(tau):
        leaves = SeedTree.reconstruct(paths[e], istars[e], salt, Np)
        leaves = [lv if lv is not None else b"\x00" * LAMBDA_BYTES for lv in leaves]
        cl = eng.commit_leaves(leaves, salt, e)
        cl[istars[e]] = comI[e]
        coms.append(cl)
        S, hints = eng.leaf_shares(leaves, salt, e)
        leafsets.append(S); hintsets.append(hints)
    h1 = h(salt, msg, ctxbytes, b"".join(b"".join(c) for c in coms), b"".join(dwp), dsep=b"h1")
    A = np.zeros((tau, rho, D + 1), dtype=np.uint16)
    for e in range(tau):
        ist = istars[e]
        dw = unpack_bits(dwp[e], eng.widths)
        ei = int(eng.e[ist])
        share = eng.lut_div[0][dw]
        share = F.smul(ei, share) ^ F.GF4[dw.astype(np.int64)]
        S = leafsets[e]
        kco = np.bitwise_xor(np.uint16(1), F.smul(ei, eng.einv))
        kco[ist] = 0
        lut = F.mul(kco[:, None], np.broadcast_to(F.GF4, (Np, 4)))
        rows = np.arange(Np)[:, None]
        share ^= np.bitwise_xor.reduce(lut[rows, S], axis=0)
        hs = eng.hint_share(hintsets[e], ist)
        A[e, :, 2:] = Atop[e]
        for r in range(rho):
            ctx = rel.precompute(h1 + bytes([e & 255, (e >> 8) & 255, r & 255]))
            g = int(rel.eval(ctx, share)) ^ int(hs[r])
            rest = 0
            for c in range(2, D + 1):
                rest ^= int(F.smul(F.pow_(ei, c), np.uint16(A[e, r, c])))
            A[e, r, 1] = F.smul(F.inv(ei), np.uint16(g ^ rest))
    return h(h1, A, dsep=b"h2") == h2
