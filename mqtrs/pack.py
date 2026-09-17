import numpy as np


def pack_bits(vals, widths):
    bits = []
    for v, w in zip(vals.tolist(), widths.tolist()):
        for k in range(w):
            bits.append((v >> k) & 1)
    arr = np.array(bits, dtype=np.uint8)
    pad = (-len(arr)) % 8
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(arr, bitorder="little").tobytes()


def unpack_bits(data, widths):
    total = int(widths.sum())
    arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")[:total]
    out = np.zeros(len(widths), dtype=np.uint8)
    p = 0
    for i, w in enumerate(widths.tolist()):
        v = 0
        for k in range(w):
            v |= int(arr[p]) << k
            p += 1
        out[i] = v
    return out


def packed_len(widths):
    return (int(widths.sum()) + 7) // 8
