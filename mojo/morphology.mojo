# Symbolic morphology learner — Mojo port. Same architecture, init and online SGD as src/rust.
# embed(16) -> mean-pool -> tanh(32) -> sigmoid(23), BCE, manual backprop, lr 0.5, 3001 epochs.
from std.time import perf_counter_ns
from std.math import exp, tanh, log

comptime E = 16
comptime H = 32
comptime F = 23
comptime LR = 0.5

def sigmoid(x: Float64) -> Float64:
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    var e = exp(x)
    return e / (1.0 + e)

def main() raises:
    var shared = String("../bench/shared/")
    var init = open(shared + "init.txt", "r").read().split("\n")
    var k = 0
    var emb = List[Float64](length=128 * E, fill=0.0)
    var w1 = List[Float64](length=H * E, fill=0.0)
    var b1 = List[Float64](length=H, fill=0.0)
    var w2 = List[Float64](length=F * H, fill=0.0)
    var b2 = List[Float64](length=F, fill=0.0)
    for i in range(128 * E):
        emb[i] = atof(init[k]); k += 1
    for i in range(H * E):
        w1[i] = atof(init[k]); k += 1
    for i in range(H):
        b1[i] = atof(init[k]); k += 1
    for i in range(F * H):
        w2[i] = atof(init[k]); k += 1
    for i in range(F):
        b2[i] = atof(init[k]); k += 1

    var lines = open(shared + "corpus.txt", "r").read().split("\n")
    var words = List[List[Int]]()
    var targets = List[List[Float64]]()
    for ln in lines:
        if ln.byte_length() == 0:
            continue
        var parts = ln.split(" ")
        var w = List[Int]()
        for c in parts[0].codepoints():
            w.append(Int(c))
        var t = List[Float64]()
        for j in range(1, len(parts)):
            t.append(atof(parts[j]))
        words.append(w^)
        targets.append(t^)
    var n = len(words)
    var wc = List[Int]()
    var woff = List[Int]()
    var tflat = List[Float64]()
    for s in range(n):
        woff.append(len(wc))
        for c in words[s]:
            wc.append(c)
        for v in targets[s]:
            tflat.append(v)
    woff.append(len(wc))
    var pooled = List[Float64](length=E, fill=0.0)
    var a1 = List[Float64](length=H, fill=0.0)
    var y = List[Float64](length=F, fill=0.0)
    var dz2 = List[Float64](length=F, fill=0.0)
    var dz1 = List[Float64](length=H, fill=0.0)
    var dh = List[Float64](length=E, fill=0.0)
    var pemb = emb.unsafe_ptr()
    var pw1 = w1.unsafe_ptr()
    var pb1 = b1.unsafe_ptr()
    var pw2 = w2.unsafe_ptr()
    var pb2 = b2.unsafe_ptr()
    var pwc = wc.unsafe_ptr()
    var ptg = tflat.unsafe_ptr()
    var ppooled = pooled.unsafe_ptr()
    var pa1 = a1.unsafe_ptr()
    var py = y.unsafe_ptr()
    var pdz2 = dz2.unsafe_ptr()
    var pdz1 = dz1.unsafe_ptr()
    var pdh = dh.unsafe_ptr()


    var epochs = 3000
    var last = 0.0
    var t0 = perf_counter_ns()
    for _ in range(epochs + 1):
        var total = 0.0
        for s in range(n):
            var wo = woff[s]
            var len_ = woff[s + 1] - wo
            # forward
            for d in range(E):
                var sum = 0.0
                for t in range(len_):
                    sum += pemb[pwc[wo + t] * E + d]
                ppooled[d] = sum / Float64(len_)
            for i in range(H):
                var z = pb1[i]
                for j in range(E):
                    z += pw1[i * E + j] * ppooled[j]
                pa1[i] = tanh(z)
            for i in range(F):
                var z = pb2[i]
                for j in range(H):
                    z += pw2[i * H + j] * pa1[j]
                py[i] = sigmoid(z)
            # loss
            var loss = 0.0
            for i in range(F):
                var p = min(max(py[i], 1e-12), 1.0 - 1e-12)
                var tg = ptg[s * F + i]
                loss -= tg * log(p) + (1.0 - tg) * log(1.0 - p)
            total += loss / Float64(F)
            # backward
            for i in range(F):
                pdz2[i] = (py[i] - ptg[s * F + i]) / Float64(F)
            for j in range(H):
                var sum = 0.0
                for i in range(F):
                    sum += pw2[i * H + j] * pdz2[i]
                pdz1[j] = sum * (1.0 - pa1[j] * pa1[j])
            for j in range(E):
                var sum = 0.0
                for i in range(H):
                    sum += pw1[i * E + j] * pdz1[i]
                pdh[j] = sum
            var inv_n = 1.0 / Float64(len_)
            for t in range(len_):
                var c = pwc[wo + t]
                for d in range(E):
                    pemb[c * E + d] -= LR * (inv_n * pdh[d])
            for i in range(H):
                pb1[i] -= LR * pdz1[i]
                for j in range(E):
                    pw1[i * E + j] -= LR * (pdz1[i] * ppooled[j])
            for i in range(F):
                pb2[i] -= LR * pdz2[i]
                for j in range(H):
                    pw2[i * H + j] -= LR * (pdz2[i] * pa1[j])
        last = total / Float64(n)
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print("mojo  examples=", n, " epochs=", epochs + 1, " final_loss=", last, " time=", dt, "s ex/s=", Float64(n * (epochs + 1)) / dt)
