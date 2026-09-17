import math

LAM = 128
QBITS = 2
CB = 32
SEEDB = 16


def ceil_div(a, b):
    return -(-a // b)


def ours_size(N, t, n=88, m=88, Np=256, lam=LAM, dmax=4):
    best = None
    for d in range(1, dmax + 1):
        b = 2
        while b ** d < N:
            b += 1
        D = max(2, d)
        tau = ceil_div(lam, int(math.floor(math.log2(Np / D))))
        rho = ceil_div(lam, 15)
        pairs = (t - 1) + (1 if b ** d > N else 0)
        wbits = t * n * QBITS + t * d * b + pairs * (3 * d - 1)
        per = wbits + lam * int(math.log2(Np)) + 2 * lam + rho * (D - 1) * 16
        sz = 4 * lam + tau * per
        cand = (ceil_div(sz, 8), d, b, tau, rho, D, wbits)
        if best is None or cand[0] < best[0]:
            best = cand
    return best


def concat_size(N, t, **kw):
    unit = ours_size(N, 1, **kw)
    tag = ceil_div(88 * QBITS, 8)
    return t * (unit[0] + tag)


def pbb13_size(N, t, n=88, m=88, lam=LAM):
    R = ceil_div(lam, 1) / -math.log2(0.75)
    R = int(math.ceil(lam / -math.log2(0.75)))
    resp = ceil_div(QBITS * (2 * n + m), 8)
    avg = ((2 * CB + 4 * N + N * resp) + (3 * CB + N * resp)
           + (2 * CB + 4 * N + N * resp) + (3 * CB + 2 * N * CB)) / 4.0
    return int(CB + R * avg)


def mtrs5_size(N, t, n=88, m=88, lam=LAM):
    R = int(math.ceil(lam / -math.log2(0.75)))
    te = N * ceil_div(QBITS * (n + m), 8)
    r_b = N * ceil_div(QBITS * n, 8)
    avg = ((3 * CB + r_b) + (3 * CB + r_b) + (2 * CB + 2 * N * CB)) / 3.0
    return int(CB + R * (te + avg))


def cds_size(N, t, n=88, m=88, lam=LAM):
    R = int(math.ceil(lam / -math.log2(2.0 / 3.0)))
    unit = R * (CB + ceil_div(QBITS * (2 * n + m), 8))
    return N * 16 + N * unit


def hs20_size(N, t, lam=LAM, m_open=2):
    lines = lam
    return lines * (N * CB + m_open * N * CB + 2 * N + N * 16 + N * 16)


def pk_bytes(scheme, n=88, m=88):
    if scheme in ("PBB13", "MTRS5"):
        return m * (n * (n + 1) // 2) * QBITS // 8
    if scheme == "HS20":
        return CB + 2
    return ceil_div(m * QBITS, 8)


SIZE_MODELS = {
    "PBB13": pbb13_size,
    "MTRS5": mtrs5_size,
    "CDS-MQ": cds_size,
    "HS20": hs20_size,
    "CONCAT": concat_size,
    "MQuorum": lambda N, t, **kw: ours_size(N, t, **kw)[0],
}
