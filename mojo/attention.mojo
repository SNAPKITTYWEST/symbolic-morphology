# Scaled dot-product attention, forward + backward — Mojo port of src/rust/src/attention.rs.
#
# Layout: row-major Float64 lists, x[i * D + d], n rows by D columns, single head, scale = 1/sqrt(D).
# Forward   S = scale * Q K^T (causal: j > i masked), P = softmax_row(S), O = P V, L_i = logsumexp_j S[i][j]
# Backward  Delta_i = dO_i . O_i
#           pass 1 (query rows):  dQ_i = scale * sum_j dS_ij k_j
#           pass 2 (key rows):    dV_j = sum_i P_ij dO_i,  dK_j = scale * sum_i dS_ij q_i
#           with P_ij = exp(scale q_i.k_j - L_i) and dS_ij = P_ij (dO_i . v_j - Delta_i)
# vec=False: scalar loops.  vec=True: SIMD[float64, 4] with fused multiply-add (needs D % 4 == 0).
# Errors are raised, never coerced: shape mismatch (lengths must be exactly n*D / n), non-finite input, D < 1,
# vec=True with D % 4 != 0.
from std.ffi import external_call
from std.math import sqrt, isfinite, inf

comptime BK = 16


# std.math.exp / log in Mojo 1.1 are not correctly rounded (exp(1.0) is off in the 13th digit, log(0.3) by 8e-11), which
# breaks agreement with the Rust and Python kernels. The kernel calls the C library's exp / log instead.
@always_inline
def exp(x: Float64) -> Float64:
    return external_call["exp", Float64](x)


@always_inline
def log(x: Float64) -> Float64:
    return external_call["log", Float64](x)


struct Rng:
    """xorshift64 -> uniform in [-1, 1). Bit-identical to attention::Rng in Rust and Rng in attention.py."""
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Float64:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return Float64(self.s >> 11) / Float64(UInt64(1) << 53) * 2.0 - 1.0

    def fill(mut self, count: Int) -> List[Float64]:
        var out = List[Float64](length=count, fill=0.0)
        for i in range(count):
            out[i] = self.next()
        return out^


@always_inline
def dot[D: Int, VEC: Bool](a: Pointer[Float64, _], b: Pointer[Float64, _]) -> Float64:
    comptime if VEC:
        var acc = SIMD[DType.float64, 4](0.0)
        comptime for d in range(0, D, 4):
            acc = a.unsafe_load[width=4](d).fma(b.unsafe_load[width=4](d), acc)
        return acc.reduce_add()
    else:
        var s = 0.0
        for d in range(D):
            s += a[unsafe_offset=d] * b[unsafe_offset=d]
        return s


@always_inline
def axpy[D: Int, VEC: Bool, o: MutOrigin](y: Pointer[Float64, o], a: Float64, x: Pointer[Float64, _]):
    comptime if VEC:
        var va = SIMD[DType.float64, 4](a)
        comptime for d in range(0, D, 4):
            y.unsafe_store(d, va.fma(x.unsafe_load[width=4](d), y.unsafe_load[width=4](d)))
    else:
        for d in range(D):
            y[unsafe_offset=d] += a * x[unsafe_offset=d]


@always_inline
def scale_by[D: Int, VEC: Bool, o: MutOrigin](y: Pointer[Float64, o], a: Float64):
    comptime if VEC:
        var va = SIMD[DType.float64, 4](a)
        comptime for d in range(0, D, 4):
            y.unsafe_store(d, va * y.unsafe_load[width=4](d))
    else:
        for d in range(D):
            y[unsafe_offset=d] *= a


def fwd_rows[D: Int, VEC: Bool](q: List[Float64], k: List[Float64], v: List[Float64], mut o: List[Float64], mut lse: List[Float64], n: Int, causal: Bool):
    var sc = 1.0 / sqrt(Float64(D))
    var pq = q.unsafe_ptr()
    var pk = k.unsafe_ptr()
    var pv = v.unsafe_ptr()
    var neg_inf = -inf[DType.float64]()
    var acc = List[Float64](length=D, fill=0.0)
    var s = List[Float64](length=BK, fill=0.0)
    var pacc = acc.unsafe_ptr()
    var ps = s.unsafe_ptr()
    for i in range(n):
        var qi = pq.unsafe_offset(i * D)
        var kmax = i + 1 if causal else n
        var m = neg_inf
        var l = 0.0
        for d in range(D):
            acc[d] = 0.0
        var j0 = 0
        while j0 < kmax:
            var bl = min(BK, kmax - j0)
            var bm = m
            for b in range(bl):
                var x = dot[D, VEC](qi, pk.unsafe_offset((j0 + b) * D)) * sc
                ps[unsafe_offset=b] = x
                if x > bm:
                    bm = x
            var corr = 0.0 if m == neg_inf else exp(m - bm)
            l *= corr
            scale_by[D, VEC](pacc, corr)
            for b in range(bl):
                var p = exp(ps[unsafe_offset=b] - bm)
                l += p
                axpy[D, VEC](pacc, p, pv.unsafe_offset((j0 + b) * D))
            m = bm
            j0 += bl
        scale_by[D, VEC](pacc, 1.0 / l)
        for d in range(D):
            o[i * D + d] = acc[d]
        lse[i] = m + log(l)


def dq_rows[D: Int, VEC: Bool](q: List[Float64], k: List[Float64], v: List[Float64], do_: List[Float64], lse: List[Float64], delta: List[Float64], mut dq: List[Float64], n: Int, causal: Bool):
    var sc = 1.0 / sqrt(Float64(D))
    var pq = q.unsafe_ptr()
    var pk = k.unsafe_ptr()
    var pv = v.unsafe_ptr()
    var pdo = do_.unsafe_ptr()
    var acc = List[Float64](length=D, fill=0.0)
    var pacc = acc.unsafe_ptr()
    for i in range(n):
        var qi = pq.unsafe_offset(i * D)
        var doi = pdo.unsafe_offset(i * D)
        var kmax = i + 1 if causal else n
        for d in range(D):
            acc[d] = 0.0
        for j in range(kmax):
            var kj = pk.unsafe_offset(j * D)
            var p = exp(dot[D, VEC](qi, kj) * sc - lse[i])
            var ds = p * (dot[D, VEC](doi, pv.unsafe_offset(j * D)) - delta[i])
            axpy[D, VEC](pacc, ds, kj)
        scale_by[D, VEC](pacc, sc)
        for d in range(D):
            dq[i * D + d] = acc[d]


def dkv_rows[D: Int, VEC: Bool](q: List[Float64], k: List[Float64], v: List[Float64], do_: List[Float64], lse: List[Float64], delta: List[Float64], mut dk: List[Float64], mut dv: List[Float64], n: Int, causal: Bool):
    var sc = 1.0 / sqrt(Float64(D))
    var pq = q.unsafe_ptr()
    var pk = k.unsafe_ptr()
    var pv = v.unsafe_ptr()
    var pdo = do_.unsafe_ptr()
    var acck = List[Float64](length=D, fill=0.0)
    var accv = List[Float64](length=D, fill=0.0)
    var pacck = acck.unsafe_ptr()
    var paccv = accv.unsafe_ptr()
    for j in range(n):
        var kj = pk.unsafe_offset(j * D)
        var vj = pv.unsafe_offset(j * D)
        for d in range(D):
            acck[d] = 0.0
            accv[d] = 0.0
        for i in range(j if causal else 0, n):
            var qi = pq.unsafe_offset(i * D)
            var doi = pdo.unsafe_offset(i * D)
            var p = exp(dot[D, VEC](qi, kj) * sc - lse[i])
            axpy[D, VEC](paccv, p, doi)
            var ds = p * (dot[D, VEC](doi, vj) - delta[i])
            axpy[D, VEC](pacck, ds, qi)
        scale_by[D, VEC](pacck, sc)
        for d in range(D):
            dk[j * D + d] = acck[d]
            dv[j * D + d] = accv[d]


def delta_rows[D: Int, VEC: Bool](o: List[Float64], do_: List[Float64], mut delta: List[Float64], n: Int):
    var po = o.unsafe_ptr()
    var pdo = do_.unsafe_ptr()
    for i in range(n):
        delta[i] = dot[D, VEC](pdo.unsafe_offset(i * D), po.unsafe_offset(i * D))


def check_len(x: List[Float64], want: Int, what: String) raises:
    if len(x) != want:
        raise Error("attention: bad shape: " + what)


def check_finite(x: List[Float64], what: String) raises:
    for i in range(len(x)):
        if not isfinite(x[i]):
            raise Error("attention: non-finite value in " + what)


def check_cfg[D: Int](vec: Bool, n: Int) raises:
    if D < 1:
        raise Error("attention: bad shape: D must be > 0")
    if n < 0:
        raise Error("attention: bad shape: n must be >= 0")
    if vec and D % 4 != 0:
        raise Error("attention: SIMD path needs D % 4 == 0")


def attn_forward[D: Int](vec: Bool, q: List[Float64], k: List[Float64], v: List[Float64], mut o: List[Float64], mut lse: List[Float64], n: Int, causal: Bool) raises:
    """Fused forward. Writes o (n*D) and lse (n); both are fully overwritten."""
    check_cfg[D](vec, n)
    check_len(q, n * D, "q")
    check_len(k, n * D, "k")
    check_len(v, n * D, "v")
    check_len(o, n * D, "o")
    check_len(lse, n, "lse")
    check_finite(q, "q")
    check_finite(k, "k")
    check_finite(v, "v")
    if n == 0:
        return
    if vec:
        comptime if D % 4 == 0:
            fwd_rows[D, True](q, k, v, o, lse, n, causal)
    else:
        fwd_rows[D, False](q, k, v, o, lse, n, causal)


def attn_backward[D: Int](vec: Bool, q: List[Float64], k: List[Float64], v: List[Float64], o: List[Float64], lse: List[Float64], do_: List[Float64],
                          mut dq: List[Float64], mut dk: List[Float64], mut dv: List[Float64], mut delta: List[Float64], n: Int, causal: Bool) raises:
    """Fused two-pass backward. dq, dk, dv (n*D) and delta (n) are fully overwritten, not accumulated."""
    check_cfg[D](vec, n)
    check_len(q, n * D, "q")
    check_len(k, n * D, "k")
    check_len(v, n * D, "v")
    check_len(o, n * D, "o")
    check_len(do_, n * D, "do")
    check_len(lse, n, "lse")
    check_len(dq, n * D, "dq")
    check_len(dk, n * D, "dk")
    check_len(dv, n * D, "dv")
    check_len(delta, n, "delta")
    check_finite(q, "q")
    check_finite(k, "k")
    check_finite(v, "v")
    check_finite(o, "o")
    check_finite(do_, "do")
    check_finite(lse, "lse")
    if n == 0:
        return
    if vec:
        comptime if D % 4 == 0:
            delta_rows[D, True](o, do_, delta, n)
            dq_rows[D, True](q, k, v, do_, lse, delta, dq, n, causal)
            dkv_rows[D, True](q, k, v, do_, lse, delta, dk, dv, n, causal)
    else:
        delta_rows[D, False](o, do_, delta, n)
        dq_rows[D, False](q, k, v, do_, lse, delta, dq, n, causal)
        dkv_rows[D, False](q, k, v, do_, lse, delta, dk, dv, n, causal)


def attn_forward_reference[D: Int](q: List[Float64], k: List[Float64], v: List[Float64], mut o: List[Float64], mut lse: List[Float64], mut s: List[Float64], n: Int, causal: Bool) raises:
    """Textbook forward: materialises S (n*n scratch) and normalises it explicitly."""
    check_cfg[D](False, n)
    check_len(q, n * D, "q")
    check_len(k, n * D, "k")
    check_len(v, n * D, "v")
    check_len(o, n * D, "o")
    check_len(lse, n, "lse")
    check_len(s, n * n, "scratch s (n*n)")
    check_finite(q, "q")
    check_finite(k, "k")
    check_finite(v, "v")
    var sc = 1.0 / sqrt(Float64(D))
    for i in range(n):
        var kmax = i + 1 if causal else n
        var m = -inf[DType.float64]()
        for j in range(kmax):
            var d_ = 0.0
            for d in range(D):
                d_ += q[i * D + d] * k[j * D + d]
            var x = d_ * sc
            s[i * n + j] = x
            if x > m:
                m = x
        var l = 0.0
        for j in range(kmax):
            var e = exp(s[i * n + j] - m)
            s[i * n + j] = e
            l += e
        for d in range(D):
            var a = 0.0
            for j in range(kmax):
                a += s[i * n + j] * v[j * D + d]
            o[i * D + d] = a / l
        lse[i] = m + log(l)


def attn_backward_reference[D: Int](q: List[Float64], k: List[Float64], v: List[Float64], do_: List[Float64],
                                     mut dq: List[Float64], mut dk: List[Float64], mut dv: List[Float64], mut p: List[Float64], mut dp: List[Float64], n: Int, causal: Bool) raises:
    """Textbook backward from the formulas (Delta = sum_j P dP, independent of dO.O). p and dp are n*n scratch."""
    check_cfg[D](False, n)
    check_len(q, n * D, "q")
    check_len(k, n * D, "k")
    check_len(v, n * D, "v")
    check_len(do_, n * D, "do")
    check_len(dq, n * D, "dq")
    check_len(dk, n * D, "dk")
    check_len(dv, n * D, "dv")
    check_len(p, n * n, "scratch p (n*n)")
    check_len(dp, n * n, "scratch dp (n*n)")
    check_finite(q, "q")
    check_finite(k, "k")
    check_finite(v, "v")
    check_finite(do_, "do")
    var sc = 1.0 / sqrt(Float64(D))
    for i in range(n):
        var kmax = i + 1 if causal else n
        var m = -inf[DType.float64]()
        for j in range(n):
            if j >= kmax:
                p[i * n + j] = 0.0
                continue
            var d_ = 0.0
            for d in range(D):
                d_ += q[i * D + d] * k[j * D + d]
            var x = d_ * sc
            p[i * n + j] = x
            if x > m:
                m = x
        var l = 0.0
        for j in range(kmax):
            var e = exp(p[i * n + j] - m)
            p[i * n + j] = e
            l += e
        for j in range(kmax):
            p[i * n + j] = p[i * n + j] / l
    for j in range(n):
        for d in range(D):
            var a = 0.0
            for i in range(n):
                a += p[i * n + j] * do_[i * D + d]
            dv[j * D + d] = a
    for i in range(n):
        for j in range(n):
            var a = 0.0
            for d in range(D):
                a += do_[i * D + d] * v[j * D + d]
            dp[i * n + j] = a
    for i in range(n):
        var delta = 0.0
        for j in range(n):
            delta += p[i * n + j] * dp[i * n + j]
        for j in range(n):
            dp[i * n + j] = p[i * n + j] * (dp[i * n + j] - delta)
    for i in range(n):
        for d in range(D):
            var a = 0.0
            for j in range(n):
                a += dp[i * n + j] * k[j * D + d]
            dq[i * D + d] = a * sc
    for j in range(n):
        for d in range(D):
            var a = 0.0
            for i in range(n):
                a += dp[i * n + j] * q[i * D + d]
            dk[j * D + d] = a * sc
