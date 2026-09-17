# Compact Threshold Ring Signatures with Soundness Bounds in the Quantum Random Oracle Model

The repository contains implementation of our proposed scheme
`MQuorum` together with measurements of five threshold ring
signatures, all placed on the same multivariate instance and the same set of
symmetric primitives so that the measurements compare constructions.

## Files and Contents
```
mqtrs/
  field.py        arithmetic in GF(2^16), GF(4) and GF(2)
  prg.py          SHAKE128 wrappers, GGM seed trees, sibling paths
  mq.py           multivariate quadratic instances and batched combination
  params.py       parameter selection (tau, rho, degree, encoding dimension)
  relation.py     the threshold constraint system C_{t,N} of our research
  tcith.py        the five round proof system, prover and verifier
  pack.py         bit packing of witness offsets over mixed subfields
  scheme.py       MQuorum key generation, signing and verification
  baselines/
    f4.py                 bit-sliced GF(4) quadratic map evaluation
    multivariate_trs.py   PBB13 and the five pass variant MTRS5
    cds.py                partial knowledge composition over the MQ scheme
    hs20.py               trapdoor commitment template
    concat.py             concatenation of linkable ring signatures
bench/
  complexity.py   analytic size models used for the asymptotic tables
  harness.py      uniform adapters and the measurement routine
  run_bench.py    the benchmark grid
  npsweep.py      the size and time trade-off in the number of parties
  make_figures.py emits the EPS 
tests/
  test_all.py     correctness tests for all six schemes
results/          measurement output in JSON
```

## Running Experiments

Run the correctness tests, which sign and verify with every scheme on a small
ring:

```
python3 tests/test_all.py
```

Sign and verify one message with `MQuorum` for a ring of 100 members and a
threshold of 50:

```
python3 - <<'PY'
import numpy as np
from mqtrs.params import best_params
from mqtrs.scheme import MQuorum
par = best_params(ring_N=100, t=50, mq_n=88, mq_m=88)
sch = MQuorum(par)
rng = np.random.default_rng(0)
sks, Y = sch.setup_ring(rng, 100)
signers = sorted(rng.choice(100, size=50, replace=False).tolist())
sig = sch.sign(b"hello", Y, signers, sks)
print(len(sig), "bytes,", sch.verify(b"hello", Y, sig))
PY
```

To check the measurements:
```
python3 bench/run_bench.py --reps 2      # writes results/bench.json
python3 bench/npsweep.py                 # writes results/npsweep.json
python3 bench/make_tables.py             
python3 bench/make_figures.py            
```

The grid of `run_bench.py` varies the threshold at a fixed ring of 64 members,
then the ring size at a fixed threshold of 8, and ends with the setting of 100
members and 50 signers. A full pass takes about twenty minutes on a laptop, and
most of that time is spent in the four baselines whose cost is proportional to
the product of the number of identification rounds and the ring size.

## Parameters
The default instance is the multivariate quadratic problem over GF(4) with 88
variables and 88 equations, which is the parameter set usually used in
the 128-bit level. The proof system uses 256 parties, privacy threshold 1,
9 multiparty repetitions and 19 parallel repetitions, and it emulates the
computation over GF(2^16). All of these are set in `mqtrs/params.py` and can be
changed from the command line of the benchmark scripts through `best_params`.

## Notes on the baselines
* `PBB13` and `MTRS5` follow the threshold ring identification structure in which
  every ring member contributes commitments in every round, with the three-pass
  and the five-pass multivariate identification schemes respectively. Each ring
  member owns a full quadratic map, so the public key is 86152 bytes.
* `CDS-MQ` is the partial knowledge composition applied to the three-pass
  multivariate identification scheme, with challenge shares on a polynomial of
  degree N-t over GF(2^128).
* `HS20` is the trapdoor commitment structure with 128 lines and 2 candidate rows.
  Its trapdoor commitment is modeled by a single hash call and an opening of
  16 bytes, which is a lower bound on the cost of any realization and therefore
  favors that scheme in the comparison.
* `CONCAT` is the concatenation of linkable ring signatures with deterministic
  tags, instantiated with the most compact multivariate ring signature available,
  which makes it smaller than its original block cipher instantiation.
