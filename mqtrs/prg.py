import hashlib
import numpy as np

LAMBDA_BYTES = 16
DIGEST_BYTES = 32


def h(*chunks, dsep=b"", outlen=DIGEST_BYTES):
    s = hashlib.shake_128()
    s.update(dsep)
    for c in chunks:
        if isinstance(c, int):
            c = c.to_bytes(8, "little")
        elif isinstance(c, np.ndarray):
            c = c.tobytes()
        s.update(len(c).to_bytes(4, "little"))
        s.update(c)
    return s.digest(outlen)


def xof(seed, outlen, dsep=b""):
    s = hashlib.shake_128()
    s.update(dsep)
    s.update(seed)
    return s.digest(outlen)


class SeedTree:
    def __init__(self, root, salt, rep, n_leaves):
        self.n = n_leaves
        self.depth = (n_leaves - 1).bit_length()
        self.salt = salt
        self.rep = rep
        self.nodes = [b""] * (2 * n_leaves)
        self.nodes[1] = root
        for i in range(1, n_leaves):
            out = xof(self.nodes[i] + salt + i.to_bytes(4, "little"), 2 * LAMBDA_BYTES, b"tree")
            self.nodes[2 * i] = out[:LAMBDA_BYTES]
            self.nodes[2 * i + 1] = out[LAMBDA_BYTES:]

    def leaves(self):
        return self.nodes[self.n:2 * self.n]

    def path(self, hidden):
        idx = self.n + hidden
        out = []
        while idx > 1:
            out.append(self.nodes[idx ^ 1])
            idx //= 2
        return out

    @staticmethod
    def reconstruct(path, hidden, salt, n_leaves):
        depth = (n_leaves - 1).bit_length()
        nodes = [None] * (2 * n_leaves)
        idx = n_leaves + hidden
        lvl = 0
        while idx > 1:
            nodes[idx ^ 1] = path[lvl]
            idx //= 2
            lvl += 1
        for i in range(1, n_leaves):
            if nodes[i] is not None:
                out = xof(nodes[i] + salt + i.to_bytes(4, "little"), 2 * LAMBDA_BYTES, b"tree")
                if nodes[2 * i] is None:
                    nodes[2 * i] = out[:LAMBDA_BYTES]
                if nodes[2 * i + 1] is None:
                    nodes[2 * i + 1] = out[LAMBDA_BYTES:]
        return nodes[n_leaves:2 * n_leaves]


def expand_subfield(seed, salt, tag, n_gf4, n_gf2, n_k):
    need = n_gf4 + n_gf2 + 2 * n_k + 8
    raw = xof(seed + salt + tag, need, b"share")
    a = np.frombuffer(raw[:n_gf4], dtype=np.uint8) & 3
    b = np.frombuffer(raw[n_gf4:n_gf4 + n_gf2], dtype=np.uint8) & 1
    off = n_gf4 + n_gf2
    c = np.frombuffer(raw[off:off + 2 * n_k], dtype=np.uint16).copy()
    return a, b, c
