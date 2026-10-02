# Benchmark Results Summary

Run date: 2026-09-27

| Implementation | Training | Throughput | Inference | Fwd+Bwd | Train Acc | Unseen Acc |
|---|---|---|---|---|---|---|
| F# .NET 8 | 1.45s | 198,881 ex/s | 1.43 µs | 4.07 µs | 96/96 | 1/8 |
| C# .NET 8 | 2.25s | 127,879 ex/s | 2.02 µs | 6.49 µs | 96/96 | 0/8 |
| C# Web SDK | 2.22s | 129,767 ex/s | 2.14 µs | 6.36 µs | 96/96 | 0/8 |
| Rust release | 2.76s | 104,466 ex/s | 2.74 µs | 7.89 µs | 96/96 | 0/8 |
| Python float | 392s | 1,516 ex/s | 147.6 µs | 512.8 µs | 196/198 | — |
| Python NAND | 43.6s | ~37 ex/s | — | — | 30% exact | 83.8% bit |

## Cross-language run — identical data, identical init (2026-09-29)

`bench/run.sh` trains every runtime on the same `bench/shared/corpus.txt` (96 words × 23 flags) from the same
seed-42 `bench/shared/init.txt`: online SGD, lr 0.5, 3001 epochs, embed 16 → tanh 32 → sigmoid 23.
Every runtime reaches the same final loss (9.1009e-5), so the runs are computing the same thing.
Best of 3, single thread, 4-core x86-64 Linux container.

| Rank | Runtime | Time | ex/s | Final loss |
|---|---|---|---|---|
| 1 | Rust, allocation-free (`src/rust/src/bin/lean.rs`) | 0.62 s | 463,000 | 9.1009e-5 |
| 2 | Mojo 1.1.0 `-O3` (`mojo/morphology.mojo`) | 0.91 s | 315,000 | 9.1009e-5 |
| 3 | Rust, original engine (`src/rust/src/bin/export.rs`) | 2.58 s | 112,000 | 9.1009e-5 |
| 4 | BQN, CBQN (`bqn/morphology.bqn`) | 2.77 s | 104,000 | 9.1009e-5 |
| 5 | Dyalog APL 20.0 (`dyalog-apl/morphology.apls`) | 5.91 s | 48,800 | 9.1009e-5 |
| 6 | Forth, gforth 0.7.3 (`forth/morphology.fs`) | 24.9 s | 11,600 | 9.1009e-5 |

Notes: the original Rust engine allocates `Vec`s in every forward and backward call; the lean port removes that
and is the fair native baseline. Dyalog and gforth are interpreters; APL's cost is per-primitive overhead on
tiny arrays (16–32 wide), where the vector primitives don't get room to pay off.

## Attention kernel (Rust, `src/rust/src/attention.rs`, 2026-10-01)

Forward + backward attention kernel with scalar, AVX2+FMA and threaded paths, integrated into `lean.rs` (`pool=attn`, default; `pool=mean` is the model in the
table above). Full method, tables, tolerances and the compute-vs-memory analysis: [`docs/ATTENTION.md`](../docs/ATTENTION.md).
Reproduce: `cd src/rust && cargo test --release && cargo run --release --bin attention`.

Headline (single head, D=64, n=2048 causal, median of 3 runs, shared 4-vCPU host — ±10–40 % run-to-run noise):

| Implementation | Forward | Backward |
|---|---:|---:|
| reference (materialised) | 270 ms · 2.0 GF/s | 3902 ms · 0.5 GF/s |
| fused scalar (vectoriser off) | 132 ms · 4.1 GF/s | 559 ms · 3.4 GF/s |
| fused scalar (LLVM autovec) | 112 ms · 4.8 GF/s | 426 ms · 4.4 GF/s |
| fused AVX2+FMA, 1 thread | 58 ms · 9.3 GF/s | 153 ms · 12.3 GF/s |
| fused AVX2+FMA, 4 threads | 21 ms · 25.8 GF/s | 61 ms · 30.9 GF/s |

Compute-side, not memory-bound (intensity ≫ DRAM ridge; per-pair time flat from 64 KiB to 4 MiB). Threads hurt at word size (n ≤ 10) and do not help at n = 128. `lean` end to end:
mean-pool 0.76 s (9.1009e-5), attention AVX2 1.40 s (8.3181e-5), attention scalar 2.15 s (8.3181e-5).

### Mojo and F# attention kernels (2026-10-02)

Kernels in `mojo/attention.mojo` and `src/fsharp/Attention.fs`; method and checks in [`docs/ATTENTION.md`](../docs/ATTENTION.md). D=64, n=2048 causal, single thread, median of 3 runs.

| Implementation | Forward | Backward |
|---|---:|---:|
| Mojo scalar | 181 ms · 3.0 GF/s | 762 ms · 2.5 GF/s |
| Mojo SIMD (4×f64 FMA) | 66 ms · 8.1 GF/s | 190 ms · 9.9 GF/s |
| F# scalar | 302 ms · 1.8 GF/s | 1135 ms · 1.7 GF/s |
| F# AVX2+FMA | 101 ms · 5.3 GF/s | 310 ms · 6.1 GF/s |

Training, 96 words × 3001 epochs, best of 3 (Mojo shares `bench/shared` with Rust; F# initialises from `Random(42)`):

| Runtime | Pooling | Time | Final loss |
|---|---|---:|---|
| Mojo | mean | 1.14 s | 9.1009e-5 |
| Mojo | attention (SIMD) | 2.76 s | 8.3181e-5 |
| F# | mean | 2.41 s | 6.7e-5 |
| F# | attention (AVX2+FMA) | 5.01 s | 6.5e-5 |
