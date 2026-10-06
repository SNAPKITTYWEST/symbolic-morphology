# Attention kernel benchmark for mojo/attention.mojo.
#   mojo build -O3 attention_bench.mojo -o /tmp/attention_bench && /tmp/attention_bench
# D = 64, causal. GF/s counts executed flops of the fused algorithm (mul and add separately, exp excluded):
# forward 4*D per (query, key) pair; backward 14*D per pair + 2*n*D.
from std.time import perf_counter_ns
from attention import (
    Rng,
    attn_forward,
    attn_backward,
    attn_forward_reference,
    attn_backward_reference,
)

comptime D = 64


def zeros(count: Int) -> List[Float64]:
    return List[Float64](length=count, fill=0.0)


struct Bench:
    var n: Int
    var q: List[Float64]
    var k: List[Float64]
    var v: List[Float64]
    var dout: List[Float64]
    var o: List[Float64]
    var lse: List[Float64]
    var delta: List[Float64]
    var dq: List[Float64]
    var dk: List[Float64]
    var dv: List[Float64]
    var s: List[Float64]
    var dp: List[Float64]

    def __init__(out self, n: Int):
        var r = Rng(42)
        self.n = n
        self.q = r.fill(n * D)
        self.k = r.fill(n * D)
        self.v = r.fill(n * D)
        self.dout = r.fill(n * D)
        self.o = zeros(n * D)
        self.lse = zeros(n)
        self.delta = zeros(n)
        self.dq = zeros(n * D)
        self.dk = zeros(n * D)
        self.dv = zeros(n * D)
        self.s = zeros(n * n)
        self.dp = zeros(n * n)

    def fwd(mut self, vec: Bool) raises:
        attn_forward[D](vec, self.q, self.k, self.v, self.o, self.lse, self.n, True)

    def bwd(mut self, vec: Bool) raises:
        attn_backward[D](vec, self.q, self.k, self.v, self.o, self.lse, self.dout, self.dq, self.dk, self.dv, self.delta, self.n, True)

    def fwd_ref(mut self) raises:
        attn_forward_reference[D](self.q, self.k, self.v, self.o, self.lse, self.s, self.n, True)

    def bwd_ref(mut self) raises:
        attn_backward_reference[D](self.q, self.k, self.v, self.dout, self.dq, self.dk, self.dv, self.s, self.dp, self.n, True)


# which: 0 fused forward, 1 fused backward, 2 reference forward, 3 reference backward
def run_once(mut b: Bench, which: Int, vec: Bool) raises:
    if which == 0:
        b.fwd(vec)
    elif which == 1:
        b.bwd(vec)
    elif which == 2:
        b.fwd_ref()
    else:
        b.bwd_ref()


def best_seconds(mut b: Bench, which: Int, vec: Bool) raises -> Float64:
    run_once(b, which, vec)
    var t0 = perf_counter_ns()
    run_once(b, which, vec)
    var one = max(Float64(perf_counter_ns() - t0) / 1e9, 1e-9)
    var iters = max(1, Int(0.08 / one) + 1)
    var best = 1e30
    for _ in range(5):
        var t = perf_counter_ns()
        for _ in range(iters):
            run_once(b, which, vec)
        var dt = Float64(perf_counter_ns() - t) / 1e9 / Float64(iters)
        if dt < best:
            best = dt
    return best


def row(label: String, n: Int, tf: Float64, tb: Float64):
    var pairs = Float64(n * (n + 1) // 2)
    var ff = pairs * 4.0 * Float64(D)
    var bf = pairs * 14.0 * Float64(D) + 2.0 * Float64(n * D)
    print(n, " ", label, " fwd ms=", tf * 1e3, " GF/s=", ff / tf / 1e9, " | bwd ms=", tb * 1e3, " GF/s=", bf / tb / 1e9)


def main() raises:
    print("attention bench (Mojo), d = 64, causal")
    var sizes = List[Int]()
    sizes.append(128)
    sizes.append(512)
    sizes.append(2048)
    for i in range(len(sizes)):
        var n = sizes[i]
        var b = Bench(n)
        row("reference (materialised)", n, best_seconds(b, 2, False), best_seconds(b, 3, False))
        b.fwd(False)
        row("fused scalar           ", n, best_seconds(b, 0, False), best_seconds(b, 1, False))
        row("fused simd (4 x f64)   ", n, best_seconds(b, 0, True), best_seconds(b, 1, True))
