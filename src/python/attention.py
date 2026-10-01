"""Scaled dot-product attention, forward + backward — Python port of src/rust/src/attention.rs (pure Python, no dependencies).

Same mathematics, layout and algorithm as the Rust fused scalar path:
  * tensors are flat row-major lists of floats, x[i * D + d], n rows by D columns; single head; Q, K, V supplied by the caller
  * forward : S = Q·Kᵀ/√D (causal: j > i masked), P = softmax_row(S), O = P·V, and L_i = logsumexp_j S[i][j];
              computed with the online-softmax recurrence over key blocks of BK = 16 (never materialises S or P)
  * backward: Δ_i = dO_i·O_i;  dQ_i = scale·Σ_j dS_ij k_j  (pass 1, by query row);  dV_j = Σ_i P_ij dO_i,
              dK_j = scale·Σ_i dS_ij q_i  (pass 2, by key row), with P_ij = exp(scale·q_i·k_j − L_i) and dS_ij = P_ij (dO_i·v_j − Δ_i)
  * errors (shape mismatch, non-finite input, D < 1) raise ValueError; nothing is silently coerced
Not ported: the AVX2 path and threading (Rust only). Run this file to execute the self-tests, including a comparison against
the Rust-generated golden vectors in bench/shared/attention_golden.txt:   python3 attention.py
"""
import math
import os
import sys

BK = 16
M64 = (1 << 64) - 1
TOL = 1e-11


class Rng:
    """xorshift64 → uniform in [-1, 1); bit-identical to attention::Rng in Rust."""
    def __init__(self, seed):
        self.x = seed & M64

    def next(self):
        x = self.x
        x ^= (x << 13) & M64
        x ^= x >> 7
        x ^= (x << 17) & M64
        self.x = x
        return (x >> 11) / float(1 << 53) * 2.0 - 1.0

    def fill(self, length):
        return [self.next() for _ in range(length)]


def _check(name, x, length):
    if len(x) != length:
        raise ValueError(f"attention: bad shape: {name} has {len(x)} elements, expected {length}")
    if not all(math.isfinite(v) for v in x):
        raise ValueError(f"attention: non-finite value in {name}")


def _dot(a, ao, b, bo, D):
    s = 0.0
    for d in range(D):
        s += a[ao + d] * b[bo + d]
    return s


def forward(q, k, v, n, D, causal):
    """Fused forward. Returns (o, lse)."""
    if D < 1:
        raise ValueError("attention: bad shape: D must be > 0")
    for name, x in (("q", q), ("k", k), ("v", v)):
        _check(name, x, n * D)
    scale = 1.0 / math.sqrt(D)
    o = [0.0] * (n * D)
    lse = [0.0] * n
    for i in range(n):
        kmax = i + 1 if causal else n
        m = -math.inf
        l = 0.0
        acc = [0.0] * D
        j0 = 0
        while j0 < kmax:
            bl = min(BK, kmax - j0)
            s = [0.0] * BK
            bm = m
            for b in range(bl):
                x = _dot(q, i * D, k, (j0 + b) * D, D) * scale
                s[b] = x
                if x > bm:
                    bm = x
            corr = 0.0 if m == -math.inf else math.exp(m - bm)
            l *= corr
            for d in range(D):
                acc[d] *= corr
            for b in range(bl):
                p = math.exp(s[b] - bm)
                l += p
                vo = (j0 + b) * D
                for d in range(D):
                    acc[d] += p * v[vo + d]
            m = bm
            j0 += bl
        inv = 1.0 / l
        for d in range(D):
            o[i * D + d] = acc[d] * inv
        lse[i] = m + math.log(l)
    return o, lse


def backward(q, k, v, o, lse, do, n, D, causal):
    """Fused two-pass backward. Returns (dq, dk, dv)."""
    if D < 1:
        raise ValueError("attention: bad shape: D must be > 0")
    for name, x in (("q", q), ("k", k), ("v", v), ("o", o), ("do", do)):
        _check(name, x, n * D)
    _check("lse", lse, n)
    scale = 1.0 / math.sqrt(D)
    delta = [_dot(do, i * D, o, i * D, D) for i in range(n)]
    dq = [0.0] * (n * D)
    dk = [0.0] * (n * D)
    dv = [0.0] * (n * D)
    for i in range(n):                                   # pass 1: dQ by query row
        kmax = i + 1 if causal else n
        acc = [0.0] * D
        for j in range(kmax):
            p = math.exp(_dot(q, i * D, k, j * D, D) * scale - lse[i])
            ds = p * (_dot(do, i * D, v, j * D, D) - delta[i])
            for d in range(D):
                acc[d] += ds * k[j * D + d]
        for d in range(D):
            dq[i * D + d] = acc[d] * scale
    for j in range(n):                                   # pass 2: dK, dV by key row
        acck = [0.0] * D
        accv = [0.0] * D
        for i in range(j if causal else 0, n):
            p = math.exp(_dot(q, i * D, k, j * D, D) * scale - lse[i])
            for d in range(D):
                accv[d] += p * do[i * D + d]
            ds = p * (_dot(do, i * D, v, j * D, D) - delta[i])
            for d in range(D):
                acck[d] += ds * q[i * D + d]
        for d in range(D):
            dk[j * D + d] = acck[d] * scale
            dv[j * D + d] = accv[d]
    return dq, dk, dv


def forward_reference(q, k, v, n, D, causal):
    """Textbook forward: materialises S and normalises explicitly. Returns (o, lse)."""
    for name, x in (("q", q), ("k", k), ("v", v)):
        _check(name, x, n * D)
    scale = 1.0 / math.sqrt(D)
    o = [0.0] * (n * D)
    lse = [0.0] * n
    for i in range(n):
        kmax = i + 1 if causal else n
        s = [_dot(q, i * D, k, j * D, D) * scale for j in range(kmax)]
        m = max(s)
        e = [math.exp(x - m) for x in s]
        l = sum(e)
        for d in range(D):
            o[i * D + d] = sum(e[j] * v[j * D + d] for j in range(kmax)) / l
        lse[i] = m + math.log(l)
    return o, lse


def backward_reference(q, k, v, do, n, D, causal):
    """Textbook backward from the formulas (Δ = Σ P·dP, independent of dO·O). Returns (dq, dk, dv)."""
    for name, x in (("q", q), ("k", k), ("v", v), ("do", do)):
        _check(name, x, n * D)
    scale = 1.0 / math.sqrt(D)
    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        kmax = i + 1 if causal else n
        s = [_dot(q, i * D, k, j * D, D) * scale for j in range(kmax)]
        m = max(s)
        e = [math.exp(x - m) for x in s]
        l = sum(e)
        for j in range(kmax):
            P[i][j] = e[j] / l
    dP = [[_dot(do, i * D, v, j * D, D) for j in range(n)] for i in range(n)]
    dS = [[0.0] * n for _ in range(n)]
    for i in range(n):
        delta = sum(P[i][j] * dP[i][j] for j in range(n))
        for j in range(n):
            dS[i][j] = P[i][j] * (dP[i][j] - delta)
    dv = [sum(P[i][j] * do[i * D + d] for i in range(n)) for j in range(n) for d in range(D)]
    dq = [sum(dS[i][j] * k[j * D + d] for j in range(n)) * scale for i in range(n) for d in range(D)]
    dk = [sum(dS[i][j] * q[i * D + d] for i in range(n)) * scale for j in range(n) for d in range(D)]
    return dq, dk, dv


# ─────────────────────────── self-tests ───────────────────────────

GOLDEN_CASES = [(1, 4, False, 11), (5, 8, False, 12), (5, 8, True, 13), (12, 16, False, 14), (12, 16, True, 15), (33, 64, True, 16)]


def _maxd(a, b):
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def _inputs(n, D, seed):
    r = Rng(seed)
    return r.fill(n * D), r.fill(n * D), r.fill(n * D), r.fill(n * D)


def _parse_golden(path):
    cases, cur = [], None
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts[0] == "case":
                cur = {"n": int(parts[1]), "D": int(parts[2]), "causal": parts[3] == "1", "seed": int(parts[4])}
                cases.append(cur)
            else:
                cur[parts[0]] = [float(x) for x in parts[1:]]
    return cases


def test_golden(path):
    cases = _parse_golden(path)
    assert [(c["n"], c["D"], c["causal"], c["seed"]) for c in cases] == GOLDEN_CASES, "golden file does not list the expected cases"
    for c in cases:
        n, D, causal = c["n"], c["D"], c["causal"]
        q, k, v, do = _inputs(n, D, c["seed"])
        o, lse = forward(q, k, v, n, D, causal)
        dq, dk, dv = backward(q, k, v, o, lse, do, n, D, causal)
        ro, rl = forward_reference(q, k, v, n, D, causal)
        rdq, rdk, rdv = backward_reference(q, k, v, do, n, D, causal)
        for name, got in (("o", o), ("lse", lse), ("dq", dq), ("dk", dk), ("dv", dv), ("ref_o", ro), ("ref_lse", rl), ("ref_dq", rdq), ("ref_dk", rdk), ("ref_dv", rdv)):
            key = name[4:] if name.startswith("ref_") else name
            d = _maxd(got, c[key])
            assert d < TOL, f"golden mismatch {name} n={n} D={D} causal={causal}: {d:e}"
    print(f"golden vectors: {len(cases)} cases match the Rust reference within {TOL:e}")


def test_fused_matches_reference():
    for D in (1, 4, 6, 8):
        for n in (1, 2, 5, 17, 20):
            for causal in (False, True):
                q, k, v, do = _inputs(n, D, 1000 + n + D)
                o, lse = forward(q, k, v, n, D, causal)
                ro, rl = forward_reference(q, k, v, n, D, causal)
                assert _maxd(o, ro) < TOL and _maxd(lse, rl) < TOL, (D, n, causal)
                g, rg = backward(q, k, v, o, lse, do, n, D, causal), backward_reference(q, k, v, do, n, D, causal)
                for a, b in zip(g, rg):
                    assert _maxd(a, b) < TOL, (D, n, causal)
    print("fused == reference for D in {1,4,6,8}, n up to 20, causal and bidirectional")


def test_finite_differences():
    h = 1e-6
    for n, D, causal in ((1, 4, False), (4, 4, False), (5, 4, True), (3, 6, True)):
        q, k, v, do = _inputs(n, D, 77 + n)
        o, lse = forward(q, k, v, n, D, causal)
        grads = backward(q, k, v, o, lse, do, n, D, causal)

        def loss(q_, k_, v_):
            return sum(a * b for a, b in zip(forward_reference(q_, k_, v_, n, D, causal)[0], do))
        for which, g in enumerate(grads):
            for idx in range(n * D):
                args = [list(q), list(k), list(v)]
                args[which][idx] += h
                lp = loss(*args)
                args[which][idx] -= 2 * h
                lm = loss(*args)
                fd = (lp - lm) / (2 * h)
                assert abs(fd - g[idx]) / max(1.0, abs(g[idx])) < 1e-7, (n, D, causal, which, idx, fd, g[idx])
    print("finite-difference gradients (dq, dk, dv) match")


def test_edges():
    assert forward([], [], [], 0, 4, False) == ([], [])
    assert backward([], [], [], [], [], [], 0, 4, True) == ([], [], [])
    q, k, v, do = _inputs(1, 4, 3)
    o, lse = forward(q, k, v, 1, 4, True)
    assert _maxd(o, v) < 1e-15                              # softmax over one key
    dq, dk, dv = backward(q, k, v, o, lse, do, 1, 4, True)
    assert max(abs(x) for x in dq + dk) < 1e-15 and _maxd(dv, do) < 1e-15
    big = [x * 300.0 for x in _inputs(9, 8, 5)[0]]          # large logits stay finite
    _, k9, v9, _ = _inputs(9, 8, 5)
    o9, l9 = forward(big, k9, v9, 9, 8, False)
    assert all(math.isfinite(x) for x in o9 + l9)
    for bad in (lambda: forward(q[1:], k, v, 1, 4, False), lambda: forward([float("nan")] * 4, k, v, 1, 4, False),
                lambda: forward([], [], [], 0, 0, False), lambda: backward(q, k, v, o, lse[1:], do, 1, 4, False)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    print("edge cases: n=0, n=1, large logits, explicit errors")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    golden = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "bench", "shared", "attention_golden.txt")
    test_edges()
    test_fused_matches_reference()
    test_finite_differences()
    test_golden(golden)
    print("all attention.py self-tests passed")
