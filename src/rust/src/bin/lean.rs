// Allocation-free Rust port of the same online-SGD loop (fixed-size stack arrays, flat weights).
// Reads the shared corpus/init files, so results are directly comparable with the other runtimes.
//
//   lean [dir] [pool=attn|mean] [epochs=3001] [isa=auto|scalar|avx2] [threads=1]
//
// pool=mean  the original mean-pool over a word's letter embeddings (the cross-language benchmark path; bench/run.sh uses it).
// pool=attn  (default) replaces mean-pooling with parameter-free self-attention over the same letter embeddings:
//              X = [emb[c_1]; …; emb[c_n]]  (n×E),   O = softmax(X·Xᵀ/√E)·X  (Q = K = V = X, bidirectional),   h = (1/n)·Σ_i O_i
//            Everything downstream of h (tanh layer, sigmoid layer, loss, SGD) is unchanged, and so is the parameter layout
//            (init.txt is read identically). The backward pass uses the attention backward kernel: dO_i = dh/n for every row,
//            and since Q, K and V are all X, the gradient reaching letter position i is dX_i = dQ_i + dK_i + dV_i.
// isa/threads select the attention kernel implementation (src/attention.rs); they have no effect with pool=mean.
use std::time::Instant;
use symbolic_engine::attention::{self, Config, Isa};

const E: usize = 16; const H: usize = 32; const F: usize = 23; const LR: f64 = 0.5;
const NV: usize = 128;      // byte-indexed embedding table
const MAXN: usize = 32;     // longest word (in letters) the stack buffers hold; longer words abort explicitly

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pool { Mean, Attn }

struct Params { emb: [[f64; E]; NV], w1: [[f64; E]; H], b1: [f64; H], w2: [[f64; H]; F], b2: [f64; F] }

/// Per-step buffers, allocated once and reused: forward activations, backward deltas and the attention workspaces.
struct Scratch {
    h: [f64; E], a1: [f64; H], dz2: [f64; F], dz1: [f64; H],
    dx: [[f64; E]; MAXN],                                  // gradient w.r.t. each letter position's embedding
    x: [[f64; E]; MAXN], o: [[f64; E]; MAXN], lse: [f64; MAXN],
    dout: [[f64; E]; MAXN], dq: [[f64; E]; MAXN], dk: [[f64; E]; MAXN], dv: [[f64; E]; MAXN], delta: [f64; MAXN],
}
impl Scratch {
    fn new() -> Scratch {
        Scratch { h: [0.0; E], a1: [0.0; H], dz2: [0.0; F], dz1: [0.0; H], dx: [[0.0; E]; MAXN], x: [[0.0; E]; MAXN], o: [[0.0; E]; MAXN],
                  lse: [0.0; MAXN], dout: [[0.0; E]; MAXN], dq: [[0.0; E]; MAXN], dk: [[0.0; E]; MAXN], dv: [[0.0; E]; MAXN], delta: [0.0; MAXN] }
    }
}

fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) } }

fn load_params(dir: &str) -> Params {
    let v: Vec<f64> = std::fs::read_to_string(format!("{dir}/init.txt")).unwrap().lines().map(|l| l.parse().unwrap()).collect();
    let mut p = Params { emb: [[0f64; E]; NV], w1: [[0f64; E]; H], b1: [0f64; H], w2: [[0f64; H]; F], b2: [0f64; F] };
    let mut k = 0;
    for r in p.emb.iter_mut() { for x in r { *x = v[k]; k += 1; } }
    for r in p.w1.iter_mut() { for x in r { *x = v[k]; k += 1; } }
    for x in p.b1.iter_mut() { *x = v[k]; k += 1; }
    for r in p.w2.iter_mut() { for x in r { *x = v[k]; k += 1; } }
    for x in p.b2.iter_mut() { *x = v[k]; k += 1; }
    p
}

fn load_corpus(dir: &str) -> (Vec<Vec<usize>>, Vec<[f64; F]>) {
    let mut words: Vec<Vec<usize>> = vec![]; let mut tgt: Vec<[f64; F]> = vec![];
    for l in std::fs::read_to_string(format!("{dir}/corpus.txt")).unwrap().lines() {
        let mut it = l.split(' ');
        words.push(it.next().unwrap().bytes().map(|b| b as usize).collect());
        let mut t = [0f64; F];
        for (i, x) in it.enumerate() { t[i] = x.parse().unwrap(); }
        tgt.push(t);
    }
    (words, tgt)
}

/// Forward + backward for one example. Fills `sc` (h, a1, dz2, dz1, dx) from the *current* parameters without modifying them
/// and returns the example's mean-over-features loss.
fn fwd_bwd(p: &Params, w: &[usize], t: &[f64; F], pool: Pool, cfg: Config, sc: &mut Scratch) -> f64 {
    let n = w.len();
    assert!(n >= 1 && n <= MAXN, "word length {n} outside supported range 1..={MAXN}");
    assert!(w.iter().all(|&c| c < NV), "letter byte outside the {NV}-row embedding table");
    let len = n as f64;

    // ── pooling: letter embeddings → h ──
    match pool {
        Pool::Mean => for d in 0..E { let mut sum = 0.0; for &c in w { sum += p.emb[c][d]; } sc.h[d] = sum / len; },
        Pool::Attn => {
            for (i, &c) in w.iter().enumerate() { sc.x[i] = p.emb[c]; }
            let x = sc.x[..n].as_flattened();
            attention::forward::<E>(cfg, x, x, x, sc.o[..n].as_flattened_mut(), &mut sc.lse[..n], n, false).unwrap_or_else(|e| panic!("{e}"));
            for d in 0..E { let mut sum = 0.0; for i in 0..n { sum += sc.o[i][d]; } sc.h[d] = sum / len; }
        }
    }

    // ── head (unchanged) ──
    for i in 0..H { let mut z = p.b1[i]; for j in 0..E { z += p.w1[i][j] * sc.h[j]; } sc.a1[i] = z.tanh(); }
    let mut y = [0f64; F];
    for i in 0..F { let mut z = p.b2[i]; for j in 0..H { z += p.w2[i][j] * sc.a1[j]; } y[i] = sigmoid(z); }
    let mut loss = 0.0;
    for i in 0..F { let q = y[i].clamp(1e-12, 1.0 - 1e-12); loss -= t[i] * q.ln() + (1.0 - t[i]) * (1.0 - q).ln(); }
    for i in 0..F { sc.dz2[i] = (y[i] - t[i]) / F as f64; }
    for j in 0..H { let mut sum = 0.0; for i in 0..F { sum += p.w2[i][j] * sc.dz2[i]; } sc.dz1[j] = sum * (1.0 - sc.a1[j] * sc.a1[j]); }
    let mut dh = [0f64; E];
    for j in 0..E { let mut sum = 0.0; for i in 0..H { sum += p.w1[i][j] * sc.dz1[i]; } dh[j] = sum; }

    // ── pooling backward: dh → per-position embedding gradient dx ──
    let inv_n = 1.0 / len;
    match pool {
        Pool::Mean => for i in 0..n { for d in 0..E { sc.dx[i][d] = inv_n * dh[d]; } },
        Pool::Attn => {
            for i in 0..n { for d in 0..E { sc.dout[i][d] = inv_n * dh[d]; } }   // h = (1/n)·Σ O_i ⇒ dO_i = dh/n
            let x = sc.x[..n].as_flattened();
            attention::backward::<E>(cfg, x, x, x, sc.o[..n].as_flattened(), &sc.lse[..n], sc.dout[..n].as_flattened(),
                                     sc.dq[..n].as_flattened_mut(), sc.dk[..n].as_flattened_mut(), sc.dv[..n].as_flattened_mut(), &mut sc.delta[..n], n, false)
                .unwrap_or_else(|e| panic!("{e}"));
            for i in 0..n { for d in 0..E { sc.dx[i][d] = sc.dq[i][d] + sc.dk[i][d] + sc.dv[i][d]; } }  // Q = K = V = X
        }
    }
    loss / F as f64
}

/// SGD update from the gradients in `sc`. Same arithmetic and order as the original in-loop updates.
fn apply(p: &mut Params, w: &[usize], sc: &Scratch, lr: f64) {
    for (i, &c) in w.iter().enumerate() { for d in 0..E { p.emb[c][d] -= lr * sc.dx[i][d]; } }   // scatter-add: repeated letters accumulate in position order
    for i in 0..H { p.b1[i] -= lr * sc.dz1[i]; for j in 0..E { p.w1[i][j] -= lr * (sc.dz1[i] * sc.h[j]); } }
    for i in 0..F { p.b2[i] -= lr * sc.dz2[i]; for j in 0..H { p.w2[i][j] -= lr * (sc.dz2[i] * sc.a1[j]); } }
}

fn train(p: &mut Params, words: &[Vec<usize>], tgt: &[[f64; F]], pool: Pool, cfg: Config, epochs: usize, sc: &mut Scratch) -> f64 {
    let n = words.len(); let mut last = 0.0;
    for _ in 0..epochs {
        let mut total = 0.0;
        for s in 0..n { total += fwd_bwd(p, &words[s], &tgt[s], pool, cfg, sc); apply(p, &words[s], sc, LR); }
        last = total / n as f64;
    }
    last
}

struct Args { dir: String, pool: Pool, epochs: usize, cfg: Config }

fn parse_args(a: &[String]) -> Result<Args, String> {
    let dir = a.get(1).cloned().unwrap_or_else(|| "../../bench/shared".into());
    let pool = match a.get(2).map(|s| s.as_str()).unwrap_or("attn") { "attn" => Pool::Attn, "mean" => Pool::Mean, x => return Err(format!("unknown pool '{x}'")) };
    let epochs = match a.get(3) { None => 3001, Some(s) => s.parse().map_err(|_| "epochs must be an integer".to_string())? };
    let isa = match a.get(4).map(|s| s.as_str()).unwrap_or("auto") { "auto" => Isa::detect(), "scalar" => Isa::Scalar, "avx2" => Isa::Avx2Fma, x => return Err(format!("unknown isa '{x}'")) };
    let threads = match a.get(5) { None => 1, Some(s) => s.parse().map_err(|_| "threads must be an integer".to_string())? };
    Ok(Args { dir, pool, epochs, cfg: Config { isa, threads } })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let Args { dir, pool, epochs, cfg } = parse_args(&a).unwrap_or_else(|msg| {
        eprintln!("lean: {msg}\nusage: lean [dir] [pool=attn|mean] [epochs=3001] [isa=auto|scalar|avx2] [threads=1]");
        std::process::exit(2)
    });
    let (isa, threads) = (cfg.isa, cfg.threads);

    let mut p = load_params(&dir);
    let (words, tgt) = load_corpus(&dir);
    let n = words.len();
    let mut sc = Scratch::new();
    let t0 = Instant::now();
    let last = train(&mut p, &words, &tgt, pool, cfg, epochs, &mut sc);
    let dt = t0.elapsed().as_secs_f64();
    if pool == Pool::Attn {
        let checksum: f64 = p.emb.iter().flatten().sum();
        println!("attention isa={} threads={threads} emb_sum={checksum:.12e}", isa.name());
    }
    println!("{}  examples={} epochs={epochs} final_loss={:.9} time={:.3}s ex/s={:.0}", if pool == Pool::Mean { "rust-lean" } else { "rust-lean-attn" }, n, last, dt, (n * epochs) as f64 / dt);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> String { concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/shared").to_string() }
    fn copy(p: &Params) -> Params { Params { emb: p.emb, w1: p.w1, b1: p.b1, w2: p.w2, b2: p.b2 } }
    fn flat(p: &Params) -> Vec<f64> {
        p.emb.iter().flatten().chain(p.w1.iter().flatten()).chain(p.b1.iter()).chain(p.w2.iter().flatten()).chain(p.b2.iter()).copied().collect()
    }
    fn avail() -> Vec<Isa> { let mut v = vec![Isa::Scalar]; if Isa::avx2_fma_available() { v.push(Isa::Avx2Fma); } v }

    fn loss_only(p: &Params, w: &[usize], t: &[f64; F], pool: Pool, cfg: Config) -> f64 { fwd_bwd(p, w, t, pool, cfg, &mut Scratch::new()) }

    /// The analytic gradients produced by the training path (dx for embeddings, dz1⊗h for w1, dz1 for b1, dz2 for b2)
    /// must match central finite differences of the loss computed through the same forward path.
    fn gradient_check(pool: Pool, cfg: Config) {
        let p0 = load_params(&dir()); let (words, tgt) = load_corpus(&dir());
        let h = 1e-4;   // loss ≈ O(1): rounding noise ~1e-16/h, truncation ~h²·f‴ — both far below the tolerance below
        let close = |fd: f64, an: f64| (fd - an).abs() < 1e-8 + 1e-6 * an.abs();
        for wi in [0usize, 7, 40, 95] {
            let (w, t) = (&words[wi], &tgt[wi]);
            let mut sc = Scratch::new(); let base = fwd_bwd(&p0, w, t, pool, cfg, &mut sc);
            assert!(base.is_finite());
            // embeddings: FD wrt emb[c][d] equals the sum of dx over every position holding letter c
            let mut seen = vec![]; for &c in w { if !seen.contains(&c) { seen.push(c); } }
            let mut any_nonzero = false;
            for &c in &seen { for d in 0..E {
                let analytic: f64 = (0..w.len()).filter(|&i| w[i] == c).map(|i| sc.dx[i][d]).sum();
                let (mut pp, mut pm) = (copy(&p0), copy(&p0)); pp.emb[c][d] += h; pm.emb[c][d] -= h;
                let fd = (loss_only(&pp, w, t, pool, cfg) - loss_only(&pm, w, t, pool, cfg)) / (2.0 * h);
                assert!(close(fd, analytic), "{pool:?} word {wi} emb[{c}][{d}]: fd={fd} analytic={analytic}");
                any_nonzero |= analytic.abs() > 1e-9;
            } }
            assert!(any_nonzero, "{pool:?}: embedding gradient is identically zero — nothing flows through the pooling op");
            // head parameters (shows the loss/head path is intact for both pools)
            for (i, j) in [(0usize, 0usize), (5, 3), (31, 15)] {
                let (mut pp, mut pm) = (copy(&p0), copy(&p0)); pp.w1[i][j] += h; pm.w1[i][j] -= h;
                let fd = (loss_only(&pp, w, t, pool, cfg) - loss_only(&pm, w, t, pool, cfg)) / (2.0 * h);
                let an = sc.dz1[i] * sc.h[j]; assert!(close(fd, an), "w1[{i}][{j}] fd={fd} an={an}");
            }
            for i in [0usize, 11, 22] {
                let (mut pp, mut pm) = (copy(&p0), copy(&p0)); pp.b2[i] += h; pm.b2[i] -= h;
                let fd = (loss_only(&pp, w, t, pool, cfg) - loss_only(&pm, w, t, pool, cfg)) / (2.0 * h);
                assert!(close(fd, sc.dz2[i]), "b2[{i}] fd={fd} an={}", sc.dz2[i]);
            }
        }
    }
    #[test] fn gradients_match_finite_differences_mean() { gradient_check(Pool::Mean, Config::SCALAR); }
    #[test] fn gradients_match_finite_differences_attention() { for isa in avail() { gradient_check(Pool::Attn, Config { isa, threads: 1 }); } }
    #[test] fn gradients_match_finite_differences_attention_threaded() { gradient_check(Pool::Attn, Config { isa: Isa::detect(), threads: 3 }); }

    #[test] fn attention_gradient_differs_from_mean_pool_gradient() {
        let p = load_params(&dir()); let (words, tgt) = load_corpus(&dir());
        let (mut a, mut m) = (Scratch::new(), Scratch::new());
        fwd_bwd(&p, &words[3], &tgt[3], Pool::Attn, Config::SCALAR, &mut a); fwd_bwd(&p, &words[3], &tgt[3], Pool::Mean, Config::SCALAR, &mut m);
        let n = words[3].len();
        let diff = (0..n).map(|i| (0..E).map(|d| (a.dx[i][d] - m.dx[i][d]).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
        assert!(diff > 1e-6, "attention pooling produced the same embedding gradient as mean pooling (diff {diff:e})");
    }

    #[test] fn training_through_attention_moves_embeddings_and_reduces_loss() {
        let p0 = load_params(&dir()); let (words, tgt) = load_corpus(&dir());
        let mut p = copy(&p0); let mut sc = Scratch::new();
        let first = train(&mut p, &words, &tgt, Pool::Attn, Config::auto(), 1, &mut sc);
        let last = train(&mut p, &words, &tgt, Pool::Attn, Config::auto(), 60, &mut sc);
        assert!(last < first * 0.5, "loss did not fall: {first} -> {last}");
        let mut used = [false; NV]; for w in &words { for &c in w { used[c] = true; } }
        for c in 0..NV {
            let moved = p.emb[c] != p0.emb[c];
            assert_eq!(moved, used[c], "embedding row {c}: used={} moved={moved}", used[c]);   // gradient reaches exactly the letters in the corpus
        }
    }

    #[test] fn attention_training_is_deterministic_and_thread_count_independent() {
        let p0 = load_params(&dir()); let (words, tgt) = load_corpus(&dir());
        let mut results = vec![];
        for isa in avail() { for threads in [1usize, 1, 2, 3, 4] {
            let mut p = copy(&p0); let mut sc = Scratch::new();
            let l = train(&mut p, &words, &tgt, Pool::Attn, Config { isa, threads }, 4, &mut sc);
            results.push((isa, threads, l, flat(&p)));
        } }
        for (isa, th, l, f) in &results {
            let (_, _, l0, f0) = results.iter().find(|(i, t, _, _)| i == isa && *t == 1).unwrap();
            assert_eq!(l.to_bits(), l0.to_bits(), "{} threads={th}: loss not bit-identical", isa.name());
            assert!(f.iter().zip(f0).all(|(a, b)| a.to_bits() == b.to_bits()), "{} threads={th}: parameters not bit-identical", isa.name());
        }
    }

    #[test] fn scalar_and_simd_training_agree_within_tolerance() {
        if !Isa::avx2_fma_available() { return; }
        let p0 = load_params(&dir()); let (words, tgt) = load_corpus(&dir());
        let run = |isa| { let mut p = copy(&p0); let l = train(&mut p, &words, &tgt, Pool::Attn, Config { isa, threads: 1 }, 3, &mut Scratch::new()); (l, flat(&p)) };
        let (ls, ps) = run(Isa::Scalar); let (lv, pv) = run(Isa::Avx2Fma);
        assert!((ls - lv).abs() < 1e-10, "loss {ls} vs {lv}");
        let d = ps.iter().zip(&pv).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(d < 1e-10, "parameters differ by {d:e}");   // 3 epochs × 96 online steps of FMA-vs-separate rounding
    }

    /// Faithful copy of the pre-refactor mean-pool step (original lean.rs inner loop). The refactored Mean path must be bit-identical.
    fn original_epoch(p: &mut Params, words: &[Vec<usize>], tgt: &[[f64; F]]) -> f64 {
        let mut total = 0.0; let n = words.len();
        for s in 0..n {
            let w = &words[s]; let t = &tgt[s]; let len = w.len() as f64;
            let mut h = [0f64; E];
            for d in 0..E { let mut sum = 0.0; for &c in w { sum += p.emb[c][d]; } h[d] = sum / len; }
            let mut a1 = [0f64; H];
            for i in 0..H { let mut z = p.b1[i]; for j in 0..E { z += p.w1[i][j] * h[j]; } a1[i] = z.tanh(); }
            let mut y = [0f64; F];
            for i in 0..F { let mut z = p.b2[i]; for j in 0..H { z += p.w2[i][j] * a1[j]; } y[i] = sigmoid(z); }
            let mut loss = 0.0;
            for i in 0..F { let q = y[i].clamp(1e-12, 1.0 - 1e-12); loss -= t[i] * q.ln() + (1.0 - t[i]) * (1.0 - q).ln(); }
            total += loss / F as f64;
            let mut dz2 = [0f64; F];
            for i in 0..F { dz2[i] = (y[i] - t[i]) / F as f64; }
            let mut dz1 = [0f64; H];
            for j in 0..H { let mut sum = 0.0; for i in 0..F { sum += p.w2[i][j] * dz2[i]; } dz1[j] = sum * (1.0 - a1[j] * a1[j]); }
            let mut dh = [0f64; E];
            for j in 0..E { let mut sum = 0.0; for i in 0..H { sum += p.w1[i][j] * dz1[i]; } dh[j] = sum; }
            let inv_n = 1.0 / len;
            for &c in w { for d in 0..E { p.emb[c][d] -= LR * (inv_n * dh[d]); } }
            for i in 0..H { p.b1[i] -= LR * dz1[i]; for j in 0..E { p.w1[i][j] -= LR * (dz1[i] * h[j]); } }
            for i in 0..F { p.b2[i] -= LR * dz2[i]; for j in 0..H { p.w2[i][j] -= LR * (dz2[i] * a1[j]); } }
        }
        total / n as f64
    }
    #[test] fn mean_pool_path_is_bit_identical_to_original_lean() {
        let p0 = load_params(&dir()); let (words, tgt) = load_corpus(&dir());
        let (mut pa, mut pb) = (copy(&p0), copy(&p0)); let mut sc = Scratch::new();
        for e in 0..8 {
            let la = original_epoch(&mut pa, &words, &tgt);
            let lb = train(&mut pb, &words, &tgt, Pool::Mean, Config::SCALAR, 1, &mut sc);
            assert_eq!(la.to_bits(), lb.to_bits(), "epoch {e} loss");
        }
        assert!(flat(&pa).iter().zip(flat(&pb)).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test] fn defaults_are_attention_pooling_3001_epochs_auto_isa_one_thread() {
        let a = parse_args(&["lean".to_string()]).unwrap();
        assert_eq!((a.pool, a.epochs, a.cfg), (Pool::Attn, 3001, Config { isa: Isa::detect(), threads: 1 }));
        let a = parse_args(&["lean", "d", "mean", "7", "scalar", "3"].map(String::from)).unwrap();
        assert_eq!((a.dir.as_str(), a.pool, a.epochs, a.cfg), ("d", Pool::Mean, 7, Config { isa: Isa::Scalar, threads: 3 }));
        for bad in [&["lean", "d", "x"][..], &["lean", "d", "attn", "-1"], &["lean", "d", "attn", "1", "neon"], &["lean", "d", "attn", "1", "scalar", "z"]] {
            assert!(parse_args(&bad.iter().map(|s| s.to_string()).collect::<Vec<_>>()).is_err(), "{bad:?}");
        }
    }
    #[test] #[should_panic(expected = "threads must be >= 1")]
    fn zero_threads_is_rejected_by_the_kernel_not_coerced() {
        let p = load_params(&dir()); let (w, t) = load_corpus(&dir());
        fwd_bwd(&p, &w[0], &t[0], Pool::Attn, Config { isa: Isa::Scalar, threads: 0 }, &mut Scratch::new());
    }
    #[test] #[should_panic(expected = "outside supported range")]
    fn over_long_word_fails_explicitly() {
        let p = load_params(&dir()); let (_, tgt) = load_corpus(&dir());
        fwd_bwd(&p, &vec![97usize; MAXN + 1], &tgt[0], Pool::Attn, Config::SCALAR, &mut Scratch::new());
    }
    #[test] #[should_panic(expected = "outside supported range")]
    fn empty_word_fails_explicitly() {
        let p = load_params(&dir()); let (_, tgt) = load_corpus(&dir());
        fwd_bwd(&p, &[], &tgt[0], Pool::Mean, Config::SCALAR, &mut Scratch::new());
    }
}
