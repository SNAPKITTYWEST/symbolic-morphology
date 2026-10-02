# Self-tests for mojo/attention.mojo. Build and run from the mojo/ directory:
#   mojo build attention_test.mojo -o /tmp/attention_test && /tmp/attention_test
# Exits non-zero (unhandled error) on the first failure.
from std.math import isfinite, abs
from attention import (
    Rng,
    attn_forward,
    attn_backward,
    attn_forward_reference,
    attn_backward_reference,
)

comptime TOL = 1e-11


def maxd(a: List[Float64], b: List[Float64]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        var d = abs(a[i] - b[i])
        if d > m:
            m = d
    return m


def expect_close(a: List[Float64], b: List[Float64], tol: Float64, what: String) raises:
    var d = maxd(a, b)
    if not (d < tol):
        raise Error("FAIL " + what + ": max diff " + String(d))


def zeros(count: Int) -> List[Float64]:
    return List[Float64](length=count, fill=0.0)


def fused_vs_reference[D: Int](vec: Bool, n: Int, causal: Bool, seed: UInt64) raises:
    var r = Rng(seed)
    var q = r.fill(n * D)
    var k = r.fill(n * D)
    var v = r.fill(n * D)
    var dout = r.fill(n * D)
    var o = zeros(n * D)
    var lse = zeros(n)
    var dq = zeros(n * D)
    var dk = zeros(n * D)
    var dv = zeros(n * D)
    var delta = zeros(n)
    attn_forward[D](vec, q, k, v, o, lse, n, causal)
    attn_backward[D](vec, q, k, v, o, lse, dout, dq, dk, dv, delta, n, causal)
    var ro = zeros(n * D)
    var rl = zeros(n)
    var s = zeros(n * n)
    var rdq = zeros(n * D)
    var rdk = zeros(n * D)
    var rdv = zeros(n * D)
    var p = zeros(n * n)
    var dp = zeros(n * n)
    attn_forward_reference[D](q, k, v, ro, rl, s, n, causal)
    attn_backward_reference[D](q, k, v, dout, rdq, rdk, rdv, p, dp, n, causal)
    var tag = " D=" + String(D) + " n=" + String(n) + " causal=" + String(causal) + " vec=" + String(vec)
    expect_close(o, ro, TOL, "o" + tag)
    expect_close(lse, rl, TOL, "lse" + tag)
    expect_close(dq, rdq, TOL, "dq" + tag)
    expect_close(dk, rdk, TOL, "dk" + tag)
    expect_close(dv, rdv, TOL, "dv" + tag)


def sweep[D: Int](simd_ok: Bool) raises:
    var sizes = List[Int]()
    sizes.append(1)
    sizes.append(2)
    sizes.append(3)
    sizes.append(7)
    sizes.append(16)
    sizes.append(17)
    sizes.append(33)
    sizes.append(64)
    for idx in range(len(sizes)):
        for c in range(2):
            fused_vs_reference[D](False, sizes[idx], c == 1, UInt64(1000 + sizes[idx]))
            if simd_ok:
                fused_vs_reference[D](True, sizes[idx], c == 1, UInt64(1000 + sizes[idx]))


def loss_via_reference[D: Int](q: List[Float64], k: List[Float64], v: List[Float64], dout: List[Float64], n: Int, causal: Bool) raises -> Float64:
    var o = zeros(n * D)
    var lse = zeros(n)
    var s = zeros(n * n)
    attn_forward_reference[D](q, k, v, o, lse, s, n, causal)
    var t = 0.0
    for i in range(n * D):
        t += o[i] * dout[i]
    return t


def finite_differences[D: Int](vec: Bool, n: Int, causal: Bool) raises:
    var r = Rng(UInt64(77 + n))
    var q = r.fill(n * D)
    var k = r.fill(n * D)
    var v = r.fill(n * D)
    var dout = r.fill(n * D)
    var o = zeros(n * D)
    var lse = zeros(n)
    var dq = zeros(n * D)
    var dk = zeros(n * D)
    var dv = zeros(n * D)
    var delta = zeros(n)
    attn_forward[D](vec, q, k, v, o, lse, n, causal)
    attn_backward[D](vec, q, k, v, o, lse, dout, dq, dk, dv, delta, n, causal)
    var h = 1e-6
    for which in range(3):
        for idx in range(n * D):
            var qp = q.copy()
            var kp = k.copy()
            var vp = v.copy()
            var qm = q.copy()
            var km = k.copy()
            var vm = v.copy()
            var g: Float64
            if which == 0:
                qp[idx] += h
                qm[idx] -= h
                g = dq[idx]
            elif which == 1:
                kp[idx] += h
                km[idx] -= h
                g = dk[idx]
            else:
                vp[idx] += h
                vm[idx] -= h
                g = dv[idx]
            var fd = (loss_via_reference[D](qp, kp, vp, dout, n, causal) - loss_via_reference[D](qm, km, vm, dout, n, causal)) / (2.0 * h)
            var scale = max(1.0, abs(g))
            if not (abs(fd - g) / scale < 1e-7):
                raise Error("FAIL finite difference tensor " + String(which) + " idx " + String(idx) + " n=" + String(n) + " causal=" + String(causal) + ": fd=" + String(fd) + " analytic=" + String(g))


def edge_cases() raises:
    var empty = List[Float64]()
    var o0 = List[Float64]()
    var l0 = List[Float64]()
    attn_forward[8](False, empty, empty, empty, o0, l0, 0, True)
    var d0 = List[Float64]()
    var dq0 = List[Float64]()
    var dk0 = List[Float64]()
    var dv0 = List[Float64]()
    attn_backward[8](False, empty, empty, empty, empty, empty, empty, dq0, dk0, dv0, d0, 0, False)

    # n = 1: softmax over one key is 1, so O = V, dQ = dK = 0, dV = dO
    var r = Rng(3)
    var q = r.fill(8)
    var k = r.fill(8)
    var v = r.fill(8)
    var dout = r.fill(8)
    var o = zeros(8)
    var lse = zeros(1)
    var dq = zeros(8)
    var dk = zeros(8)
    var dv = zeros(8)
    var delta = zeros(1)
    attn_forward[8](True, q, k, v, o, lse, 1, True)
    attn_backward[8](True, q, k, v, o, lse, dout, dq, dk, dv, delta, 1, True)
    expect_close(o, v, 1e-15, "n=1 o == v")
    expect_close(dq, zeros(8), 1e-15, "n=1 dq == 0")
    expect_close(dk, zeros(8), 1e-15, "n=1 dk == 0")
    expect_close(dv, dout, 1e-15, "n=1 dv == dout")

    # constant V gives constant O
    var n = 13
    var r2 = Rng(21)
    var q2 = r2.fill(n * 8)
    var k2 = r2.fill(n * 8)
    var vc = List[Float64](length=n * 8, fill=0.25)
    var o2 = zeros(n * 8)
    var l2 = zeros(n)
    attn_forward[8](True, q2, k2, vc, o2, l2, n, False)
    expect_close(o2, vc, 1e-14, "constant V")

    # large logits stay finite and match the reference
    var r3 = Rng(9)
    var qb = r3.fill(20 * 16)
    for i in range(len(qb)):
        qb[i] *= 300.0
    var kb = r3.fill(20 * 16)
    var vb = r3.fill(20 * 16)
    var ob = zeros(20 * 16)
    var lb = zeros(20)
    var sb = zeros(400)
    var orf = zeros(20 * 16)
    var lrf = zeros(20)
    attn_forward[16](True, qb, kb, vb, ob, lb, 20, False)
    attn_forward_reference[16](qb, kb, vb, orf, lrf, sb, 20, False)
    for i in range(len(ob)):
        if not isfinite(ob[i]):
            raise Error("FAIL large logits: non-finite output")
    expect_close(ob, orf, 1e-8, "large logits")

    # causal: changing the last key/value must not change earlier output rows
    var r4 = Rng(4)
    var q4 = r4.fill(10 * 8)
    var k4 = r4.fill(10 * 8)
    var v4 = r4.fill(10 * 8)
    var oa = zeros(80)
    var la = zeros(10)
    attn_forward[8](False, q4, k4, v4, oa, la, 10, True)
    var k5 = k4.copy()
    var v5 = v4.copy()
    for d in range(8):
        k5[9 * 8 + d] -= 3.0
        v5[9 * 8 + d] += 5.0
    var ob2 = zeros(80)
    var lb2 = zeros(10)
    attn_forward[8](False, q4, k5, v5, ob2, lb2, 10, True)
    for i in range(72):
        if oa[i] != ob2[i]:
            raise Error("FAIL causal: last key/value leaked into row " + String(i // 8))


def expect_error(label: String, raised: Bool) raises:
    if not raised:
        raise Error("FAIL expected an error: " + label)


def error_paths() raises:
    var r = Rng(1)
    var q = r.fill(32)
    var k = r.fill(32)
    var v = r.fill(32)
    var o = zeros(32)
    var lse = zeros(4)
    var raised = False
    try:
        var short = zeros(31)
        attn_forward[8](False, short, k, v, o, lse, 4, False)
    except:
        raised = True
    expect_error("shape mismatch", raised)

    raised = False
    try:
        var bad = q.copy()
        bad[3] = bad[3] / 0.0
        attn_forward[8](False, bad, k, v, o, lse, 4, False)
    except:
        raised = True
    expect_error("non-finite input", raised)

    raised = False
    try:
        var q6 = zeros(24)
        var o6 = zeros(24)
        attn_forward[6](True, q6, q6, q6, o6, lse, 4, False)
    except:
        raised = True
    expect_error("SIMD with D % 4 != 0", raised)

    raised = False
    try:
        var e = List[Float64]()
        var oe = List[Float64]()
        var le = List[Float64]()
        attn_forward[0](False, e, e, e, oe, le, 0, False)
    except:
        raised = True
    expect_error("D = 0", raised)


def golden_case[D: Int](vec: Bool, n: Int, causal: Bool, seed: UInt64, eo: List[Float64], el: List[Float64], edq: List[Float64], edk: List[Float64], edv: List[Float64]) raises:
    var r = Rng(seed)
    var q = r.fill(n * D)
    var k = r.fill(n * D)
    var v = r.fill(n * D)
    var dout = r.fill(n * D)
    var o = zeros(n * D)
    var lse = zeros(n)
    var dq = zeros(n * D)
    var dk = zeros(n * D)
    var dv = zeros(n * D)
    var delta = zeros(n)
    attn_forward[D](vec, q, k, v, o, lse, n, causal)
    attn_backward[D](vec, q, k, v, o, lse, dout, dq, dk, dv, delta, n, causal)
    var tag = " golden D=" + String(D) + " n=" + String(n) + " causal=" + String(causal) + " vec=" + String(vec)
    expect_close(o, eo, TOL, "o" + tag)
    expect_close(lse, el, TOL, "lse" + tag)
    expect_close(dq, edq, TOL, "dq" + tag)
    expect_close(dk, edk, TOL, "dk" + tag)
    expect_close(dv, edv, TOL, "dv" + tag)


def parse_row(line: String) raises -> List[Float64]:
    var parts = line.split(" ")
    var out = List[Float64]()
    for j in range(1, len(parts)):
        out.append(atof(parts[j]))
    return out^


def golden(path: String) raises:
    var lines = open(path, "r").read().split("\n")
    var idx = 0
    var cases = 0
    while idx < len(lines) and lines[idx].byte_length() > 0:
        var head = lines[idx].split(" ")
        var n = atol(head[1])
        var d = atol(head[2])
        var causal = head[3] == "1"
        var seed = UInt64(atol(head[4]))
        var eo = parse_row(String(lines[idx + 1]))
        var el = parse_row(String(lines[idx + 2]))
        var edq = parse_row(String(lines[idx + 3]))
        var edk = parse_row(String(lines[idx + 4]))
        var edv = parse_row(String(lines[idx + 5]))
        for vec in range(2):
            var use_vec = vec == 1
            if d == 4:
                golden_case[4](use_vec, n, causal, seed, eo, el, edq, edk, edv)
            elif d == 8:
                golden_case[8](use_vec, n, causal, seed, eo, el, edq, edk, edv)
            elif d == 16:
                golden_case[16](use_vec, n, causal, seed, eo, el, edq, edk, edv)
            elif d == 64:
                golden_case[64](use_vec, n, causal, seed, eo, el, edq, edk, edv)
            else:
                raise Error("golden case with unsupported D " + String(d))
        cases += 1
        idx += 6
    if cases != 6:
        raise Error("expected 6 golden cases, found " + String(cases))
    print("golden vectors:", cases, "cases match the Rust reference (scalar and SIMD) within 1e-11")


def main() raises:
    edge_cases()
    print("edge cases: n=0, n=1, constant V, large logits, causal no-leak")
    error_paths()
    print("error paths: shape, non-finite, SIMD with D % 4 != 0, D = 0")
    sweep[4](True)
    sweep[8](True)
    sweep[16](True)
    sweep[64](True)
    sweep[6](False)
    sweep[1](False)
    print("fused == reference: D in {1, 4, 6, 8, 16, 64}, n in {1..64}, causal and bidirectional, scalar and SIMD")
    for c in range(2):
        for nn in range(1, 6):
            if nn == 1 or nn == 2 or nn == 5:
                finite_differences[4](False, nn, c == 1)
                finite_differences[4](True, nn, c == 1)
                finite_differences[8](True, nn, c == 1)
    print("finite-difference gradients (dq, dk, dv) match")
    golden("../bench/shared/attention_golden.txt")
    print("all attention tests passed")
