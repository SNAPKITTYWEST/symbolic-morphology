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
