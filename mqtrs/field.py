import numpy as np

GF16_MODULUS = 0x1100B
GF16_ORDER = 1 << 16
GF16_MAX = GF16_ORDER - 1


def _build_tables():
    exp = np.zeros(2 * GF16_MAX, dtype=np.uint16)
    log = np.zeros(GF16_ORDER, dtype=np.uint16)
    x = 1
    for i in range(GF16_MAX):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & GF16_ORDER:
            x ^= GF16_MODULUS
    exp[GF16_MAX:] = exp[:GF16_MAX]
    return exp, log


EXP, LOG = _build_tables()
_EXP32 = EXP.astype(np.uint32)
_LOG32 = LOG.astype(np.uint32)

_ZLOG = np.uint32(1 << 20)
LOGX = LOG.astype(np.uint32).copy()
LOGX[0] = _ZLOG
EXPX = np.zeros((1 << 21) + 4, dtype=np.uint16)
EXPX[: 2 * GF16_MAX] = EXP[: 2 * GF16_MAX]


def mul(a, b):
    return EXPX[LOGX[a] + LOGX[b]]


def scalar_table(c):
    c = int(c) & GF16_MAX
    if c == 0:
        return np.zeros(GF16_ORDER, dtype=np.uint16)
    idx = _LOG32[np.arange(GF16_ORDER, dtype=np.uint32).astype(np.uint16)] + np.uint32(LOG[c])
    tbl = _EXP32[idx].astype(np.uint16)
    tbl[0] = 0
    return tbl


def smul(c, a):
    c = int(c) & GF16_MAX
    a = np.asarray(a, dtype=np.uint16)
    if c == 0:
        return np.zeros_like(a)
    if c == 1:
        return a.copy()
    return EXPX[LOGX[a] + np.uint32(LOG[c])]


def add(a, b):
    return np.bitwise_xor(np.asarray(a, dtype=np.uint16), np.asarray(b, dtype=np.uint16))


def inv(a):
    a = int(a) & GF16_MAX
    if a == 0:
        raise ZeroDivisionError
    return int(EXP[(GF16_MAX - int(LOG[a])) % GF16_MAX])


def pow_(a, e):
    a = int(a) & GF16_MAX
    if a == 0:
        return 0
    return int(EXP[(int(LOG[a]) * int(e)) % GF16_MAX])


def dot(a, b):
    return int(np.bitwise_xor.reduce(mul(a, b))) if len(a) else 0


GF4_GEN = int(EXP[GF16_MAX // 3])
GF4 = np.array([0, 1, GF4_GEN, int(EXP[(2 * (GF16_MAX // 3)) % GF16_MAX])], dtype=np.uint16)
GF4_INDEX = {int(v): i for i, v in enumerate(GF4)}
GF2 = np.array([0, 1], dtype=np.uint16)


def lift_gf4(idx):
    return GF4[np.asarray(idx, dtype=np.int64)]


def gf4_index(vals):
    out = np.zeros(len(vals), dtype=np.uint8)
    for i, v in enumerate(np.asarray(vals, dtype=np.uint16)):
        out[i] = GF4_INDEX[int(v)]
    return out


def matvec(mat, vec):
    prod = mul(mat, vec[None, :])
    return np.bitwise_xor.reduce(prod, axis=1)


def matmul_tall(mat, mats):
    out = np.zeros((mats.shape[0], mat.shape[0]), dtype=np.uint16)
    for i in range(mats.shape[0]):
        out[i] = matvec(mat, mats[i])
    return out
