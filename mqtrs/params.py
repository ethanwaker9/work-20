import math
from dataclasses import dataclass, field

K_BITS = 16
LAMBDA = 128
MPC_FP_NUMERATOR = 2


def _ceil_div(a, b):
    return -(-a // b)


@dataclass
class Params:
    ring_N: int
    t: int
    d: int = 2
    n_party: int = 256
    ell: int = 1
    mq_n: int = 88
    mq_m: int = 88
    lam: int = LAMBDA
    b: int = field(init=False)
    D: int = field(init=False)
    tau: int = field(init=False)
    rho: int = field(init=False)

    def __post_init__(self):
        self.b = max(2, math.ceil(self.ring_N ** (1.0 / self.d) - 1e-9))
        while self.b ** self.d < self.ring_N:
            self.b += 1
        self.D = max(2, self.d)
        self.rho = _ceil_div(self.lam, K_BITS - (MPC_FP_NUMERATOR - 1).bit_length())
        soundness = math.log2(self.n_party / self.D)
        self.tau = _ceil_div(self.lam, int(math.floor(soundness)))

    @property
    def witness_gf4(self):
        return self.t * self.mq_n

    @property
    def witness_gf2(self):
        return self.t * self.d * self.b + max(0, self.t - 1) * (3 * self.d - 1)

    @property
    def witness_len(self):
        return self.witness_gf4 + self.witness_gf2

    @property
    def witness_bits(self):
        return 2 * self.witness_gf4 + self.witness_gf2

    def signature_bits(self):
        per_rep = (self.witness_bits
                   + self.lam * int(math.log2(self.n_party))
                   + 2 * self.lam
                   + self.rho * (self.D - 1) * K_BITS)
        return 4 * self.lam + self.tau * per_rep

    def signature_bytes(self):
        return _ceil_div(self.signature_bits(), 8)


def best_params(ring_N, t, n_party=256, dmax=4, **kw):
    best = None
    for d in range(1, dmax + 1):
        p = Params(ring_N=ring_N, t=t, d=d, n_party=n_party, **kw)
        if best is None or p.signature_bits() < best.signature_bits():
            best = p
    return best
