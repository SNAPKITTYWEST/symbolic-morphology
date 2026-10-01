# Attention kernel

Forward and backward scaled dot-product attention, in Rust (`src/rust/src/attention.rs`), integrated into the
training loop of `src/rust/src/bin/lean.rs`, with a pure-Python port (`src/python/attention.py`) that is checked
against the Rust reference on shared golden vectors.

## The operation

Single head, no learned projections. Tensors are row-major `f64`, `x[i*D + d]`, `n` rows × `D` columns.
`scale = 1/√D`. With `causal`, `S[i][j] = −∞` for `j > i` (so `P[i][j] = 0`).

**Forward**

```
S   = scale · Q Kᵀ
P   = softmax_row(S)
O   = P V
L_i = log Σ_j exp(S[i][j])            (saved for the backward pass)
```

**Backward** (given `dO`)

```
dV = Pᵀ dO
dP = dO Vᵀ
Δ_i = Σ_j P[i][j] dP[i][j]  =  dO_i · O_i
dS = P ⊙ (dP − Δ)                      (row-wise softmax Jacobian)
dQ = scale · dS K
dK = scale · dSᵀ Q
```

`Δ_i = dO_i · O_i` because `O_i = Σ_j P_ij v_j` ⇒ `Σ_j P_ij (dO_i · v_j) = dO_i · O_i`. The fused backward never stores `P`:
it recomputes `P[i][j] = exp(scale·q_i·k_j − L_i)` from the saved `L`. Forward and backward therefore share the same scale,
the same masking and the same log-sum-exp; `exp(s − L)` equals the forward's `exp(s − m)/l` up to rounding.

## Implementations (all compute the above)

| Path | Where | What it is |
|---|---|---|
| reference | `forward_reference`, `backward_reference` | Textbook: materialises `S`/`P`/`dP`/`dS` (n×n scratch). Backward uses `Δ = Σ P·dP`, so it is an independent derivation. Used as the oracle in tests. |
| fused scalar | `forward`/`backward` with `Config::SCALAR` | Online-softmax forward over 16-key blocks (flash-attention recurrence), two-pass backward. Plain loops; LLVM may auto-vectorise element-wise loops. |
| fused AVX2+FMA | `Config { isa: Isa::Avx2Fma, .. }` | Same algorithm; the `D`-length dot / axpy / scale primitives are explicit AVX2+FMA intrinsics. Selected by runtime detection (`Isa::detect()`); requesting it on a CPU without AVX2+FMA, or with `D % 4 != 0`, returns an error. `exp` is still scalar libm. |
| threaded | `Config { threads: t, .. }` | Static contiguous row partition (work-balanced for causal masks). |

**Backward structure.** Pass 0: `Δ_i`. Pass 1 (parallel over query rows): `dQ_i = scale Σ_j dS_ij k_j`, `j` ascending.
Pass 2 (parallel over key rows): `dV_j = Σ_i P_ij dO_i`, `dK_j = scale Σ_i dS_ij q_i`, `i` ascending. Every output element is written
by exactly one thread and summed in a fixed order, so there is no atomic or cross-thread accumulation. The cost is that scores are
recomputed in both passes (backward ≈ 14·D flops per (i,j) pair vs 4·D for the forward).

**Determinism.** For a given ISA, results are bit-identical for every thread count and across runs (tested for 1…256 threads).
Different ISAs (scalar vs AVX2+FMA) differ only by rounding (FMA, reassociated dot products).

**Tolerances.** Any two implementations agree to `1e-11` absolute on inputs of magnitude ≲ 1 with `n ≤ 64`
(observed ≈ `1e-16`); with logits scaled ×300 (softmax saturated) the tests use `1e-8`.

**Errors, not fallbacks.** `AttnError`: shape mismatch, non-finite input, `D = 0`, `threads = 0` or `> 256`, AVX2 with `D % 4 ≠ 0`,
AVX2 on a CPU without it. `n = 0` is a no-op. `n = 1` returns `V` exactly (and `dQ = dK = 0`, `dV = dO`).

**Layout.** `forward_heads`/`backward_heads`: head-major `[h][n][D]`. `forward_interleaved`: the original `n × (H·D)` interleaved
layout (forward only), gathered through caller-provided scratch.

## Integration in `lean.rs`

```
lean [dir] [pool=attn|mean] [epochs=3001] [isa=auto|scalar|avx2] [threads=1]
```

`pool=mean` is the original mean-pool (bit-identical to the pre-change code — proven by a unit test that embeds the original loop —
and the model the cross-language benchmark uses; `bench/run.sh` passes it). `pool=attn` (the default) replaces the pooling step:

```
X = [emb[c_1]; …; emb[c_n]]            n×16
O = softmax(X Xᵀ / √16) X               Q = K = V = X, bidirectional, no new parameters
h = (1/n) Σ_i O_i                       same 16-vector the head already consumes
```

Everything after `h` (tanh layer, sigmoid layer, loss, SGD, `init.txt` layout) is unchanged. Backward: `dh` is computed exactly as before;
`dO_i = dh/n` for every row; the attention backward gives `dQ, dK, dV`; since `Q = K = V = X`, the gradient reaching letter
position `i` is `dX_i = dQ_i + dK_i + dV_i`, and `emb[c_i] −= lr·dX_i` is applied in position order (repeated letters accumulate).
Words must have 1…32 letters (`MAXN`); anything else aborts with a message.

`isa`/`threads` affect only the attention kernel. Threading is *not* useful at word size (see below): the corpus words have ≤ 10 letters.

## Tests

`cd src/rust && cargo test --release` (debug `cargo test` also passes; it is slower and substitutes a 300-epoch mean-pool check for the 3001-epoch one).

* `src/attention.rs` — fused vs reference (forward, `L`, `dQ/dK/dV`) for `D ∈ {4,8,16,64}`, `n ∈ {1,2,3,7,16,17,33,64}`, both masks, scalar & AVX2, 1–4 threads;
  `D ∈ {1,6}` on the scalar path; central finite differences of `Σ dO⊙O` through the *reference* forward; `n = 0`, `n = 1`, constant `V`,
  causal no-leak, large logits, `O` linear in `V`; thread-count bit-identity (1…256 threads, repeated); partition covers; head-major and interleaved
  wrappers; every error path; the golden vectors.
* `src/bin/lean.rs` — finite-difference check of the full training-path gradients (embeddings incl. repeated letters, `w1`, `b2`) for both pools and for
  scalar/AVX2/threaded kernels (a deliberately broken `dX = dQ + dK` fails it); attention gradient ≠ mean-pool gradient; training moves exactly the
  embedding rows of letters present in the corpus and cuts loss > 2× in 60 epochs; determinism and thread-count independence of training;
  scalar-vs-AVX2 training within `1e-10`; mean-pool path bit-identical to the original loop; argument parsing and explicit failures.
* `tests/lean_cli.rs` — the real binary end to end: mean-pool reproduces the cross-language `final_loss=0.000091009`; attention regression values;
  reproducibility and thread invariance; SIMD ≈ scalar; bad arguments.
* `python3 src/python/attention.py` — port self-tests + golden vectors.

Golden vectors (`bench/shared/attention_golden.txt`) are generated from seeds by the Rust reference path
(`cargo run --release --bin attention -- --write-golden`); both Rust (`cargo test`) and Python read them.

## Performance

Reproduce: `cd src/rust && cargo run --release --bin attention` (it first runs a correctness gate — fused == reference within 1e-11 and
threaded == single-thread bit for bit — and refuses to benchmark if that fails). Single head, f64, `D = 64` unless noted.

**Machine / method.** Intel Xeon @ 2.1 GHz, 4 vCPUs in a shared container (AVX-512 present but not used by this code). Each cell is best-of-5 samples of ≥ 80 ms;
the table shows the **median of 3 full runs**. Run-to-run variation is large on this shared host (individual cells moved ±10–40 %; the measured single-thread
AVX2+FMA peak itself ranged 31–41 GF/s between runs), so treat differences under ~20 % as noise. Measured machine limits: AVX2+FMA peak 31–41 GF/s on 1 thread
(67–73 on 2, 128–142 on 4 concurrently), read bandwidth 56–66 GB/s L2-resident and 9.5–11.9 GB/s DRAM (1 thread), `exp` 5.4–8.8 ns/call.
GF/s counts *executed* flops of the fused algorithm (mul and add separately, `exp` excluded): forward 4·D per (query,key) pair, backward 14·D per pair + 2·n·D;
the same numerator is used for every row, including the reference (which also computes masked-out entries in the causal case).

**Rows.** *scalar* = fused kernel, scalar primitives, built with LLVM's loop and SLP vectorisers disabled
(`RUSTFLAGS="-C llvm-args=-vectorize-loops=false -C llvm-args=-vectorize-slp=false"`), 1 thread. *autovec* = the same source with LLVM's default vectoriser, 1 thread.
*AVX2+FMA* = explicit intrinsics, 1/2/4 threads. *reference* = materialised textbook version.

| n, causal | implementation | fwd ms | fwd GF/s | bwd ms | bwd GF/s |
|---|---|---:|---:|---:|---:|
| 8 (D=16), no | reference | 0.002 | 1.95 | 0.005 | 3.11 |
| | scalar | 0.002 | 2.17 | 0.005 | 2.97 |
| | autovec | 0.001 | 3.19 | 0.004 | 3.93 |
| | AVX2+FMA, 1 thread | 0.001 | 3.18 | 0.002 | 7.14 |
| | AVX2+FMA, 2 / 4 threads | 0.089 / 0.143 | 0.05 / 0.03 | 0.168 / 0.289 | 0.09 / 0.05 |
| 128, yes | reference | 1.414 | 1.50 | 5.231 | 1.42 |
| | scalar | 0.552 | 3.83 | 2.199 | 3.37 |
| | autovec | 0.426 | 4.96 | 1.582 | 4.69 |
| | AVX2+FMA, 1 thread | 0.226 | 9.37 | 0.637 | 11.63 |
| | AVX2+FMA, 2 threads | 0.263 | 8.03 | 0.796 | 9.31 |
| | AVX2+FMA, 4 threads | 0.246 | 8.58 | 0.560 | 13.24 |
| 512, no | reference | 32.84 | 2.04 | 144.1 | 1.63 |
| | scalar | 16.71 | 4.02 | 74.67 | 3.15 |
| | autovec | 13.56 | 4.95 | 54.79 | 4.29 |
| | AVX2+FMA, 1 thread | 7.08 | 9.48 | 18.17 | 12.93 |
| | AVX2+FMA, 2 threads | 4.56 | 14.73 | 12.44 | 18.89 |
| | AVX2+FMA, 4 threads | 2.65 | 25.30 | 8.15 | 28.85 |
| 2048, yes | reference | 270.1 | 1.99 | 3901.7 | 0.48 |
| | scalar | 132.1 | 4.07 | 558.7 | 3.37 |
| | autovec | 111.5 | 4.82 | 425.5 | 4.42 |
| | AVX2+FMA, 1 thread | 58.07 | 9.25 | 153.4 | 12.25 |
| | AVX2+FMA, 2 threads | 32.00 | 16.79 | 85.55 | 21.98 |
| | AVX2+FMA, 4 threads | 20.81 | 25.81 | 60.86 | 30.90 |

(The n = 512 / 2048 / 128 causal-and-bidirectional variants are in the raw benchmark output; they show the same pattern.)

**What the measurements support**

* Explicit AVX2+FMA vs the vectoriser-disabled scalar build, 1 thread, n ≥ 512: forward 2.3–2.4×, backward 3.6–4.1×. Vs LLVM's autovectorised scalar source: forward ≈ 1.9×, backward ≈ 2.8–3.0×.
  LLVM's own vectoriser gives the scalar source only ≈ 1.2× (forward) and 1.3× (backward) over the vectoriser-off build.
* The forward gains less than the backward because `exp` (scalar libm) is a larger share of its per-pair cost: with 1 `exp` per pair (5–9 ns) out of ≈ 27 ns,
  vs 2 `exp`s out of ≈ 73 ns in the backward. This is an estimate from the `exp` microbenchmark, not a profile.
* Threads: AVX2+FMA 4 threads vs 1 thread is 2.7–2.8× (forward) and 2.2–2.5× (backward) at n = 512–2048 (≈ 55–70 % parallel efficiency on a shared 4-vCPU host).
  **At n = 128 threads barely help (forward is slower, backward ≤ 1.1× faster) and at word size (n = 8) they are two orders of magnitude slower** (≈ 0.09–0.3 ms per call vs ≈ 1–4 µs): each call pays 1 (forward) or 2 (backward) `thread::scope` spawn rounds, ≈ 80–90 µs each.
  Threading only pays off once a single call takes ≳ 1 ms.
* The reference materialises n×n matrices and is 1.9–7× slower than the fused scalar kernel (forward ≈ 2–2.6×, backward ≈ 1.9–7×, largest at n = 2048 causal, where it also computes the masked half).

**Compute-bound or memory-bound?** Compute-side (in-core), not memory-bound:

* Arithmetic intensity against compulsory traffic is 8–128 flop/B forward and 14–223 flop/B backward for n ≥ 128, versus a measured DRAM ridge point of 2.8–3.9 flop/B.
  Achieved effective bandwidth is ≤ 1.3 GB/s against 9.5–11.9 GB/s available.
* It is not limited by cache bandwidth either: AVX2 time per (query,key) pair is flat at ≈ 27 ns forward and ≈ 72–77 ns backward from a 64 KiB to a 4 MiB working set (n = 128 → 2048).
* It also runs well below the FMA peak (≈ 25–35 % of peak at 1 thread), so the limit is latency/instruction overhead per pair — the scalar `exp`, the horizontal reduction after each dot product, and
  load/store traffic on the accumulator — rather than FMA throughput. That attribution is inferred from the above, not profiled. AVX-512, vectorised `exp`, or processing several queries per K/V pass would be the next levers; none is implemented.
* The only configuration below the DRAM ridge by this metric is n = 8, whose 8 KiB working set lives in L1; there the cost is per-call overhead, not memory.

**End to end (`lean`, 96 words × 3001 epochs, online SGD, best of 3)**

| pooling | kernel | time | examples/s | final loss |
|---|---|---:|---:|---|
| mean (original) | — | 0.76 s | 381,000 | 9.1009e-5 |
| attention | scalar, 1 thread | 2.15 s | 134,000 | 8.3181e-5 |
| attention | AVX2+FMA, 1 thread | 1.40 s | 205,000 | 8.3181e-5 |
| attention (20 epochs) | AVX2+FMA, 1 / 2 / 4 threads | 0.010 s / 0.47 s / 0.83 s | 195,000 / 4,100 / 2,300 | identical |

Attention pooling costs ≈ 1.8× the mean-pool step at word size (n ≤ 10) and trains to a different (lower, on this corpus) training loss — it is a different model, so the cross-language
table in `bench/results.md` stays on `pool=mean`. Threads give a bit-identical result but are strictly slower at this size; use `threads=1`.

## Not done

* **Ports to the other languages.** Only Python is ported (and tested against the Rust golden vectors). BQN, Dyalog APL, Forth, Mojo, C# and F# have **no** attention implementation:
  none of those toolchains is installed in the environment this was built in, and an unrunnable port could not be verified. The golden vectors
  (`bench/shared/attention_golden.txt`) and seeds are the interchange spec for adding them.
* The Python port is fused-scalar only (no SIMD, no threads) and is not wired into the Python training scripts.
* `lean.rs` uses parameter-free attention (Q = K = V = the letter embeddings). Learned Q/K/V projections would change the parameter layout and `init.txt` and were deliberately not introduced.
* No AVX-512 / NEON path; non-x86_64 builds get the scalar path and a clear error if `Isa::Avx2Fma` is requested.
