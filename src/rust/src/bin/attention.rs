#![allow(unsafe_op_in_unsafe_fn)] // the two machine-peak helpers are whole-body unsafe intrinsic loops
// Attention kernel benchmark. Library: src/attention.rs.   Usage:
//   attention                      verify, then benchmark forward + backward
//   attention --write-golden [p]   (re)write the cross-language golden vectors (default ../../bench/shared/attention_golden.txt)
// Correctness gate: the benchmark refuses to run unless every implementation agrees with the materialised reference
// (tolerance TOL) and threaded results are bit-identical to single-thread. The full suite is `cargo test --release`.
use std::time::Instant;
use symbolic_engine::attention::*;

const TOL: f64 = 1e-11;

fn maxd(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) }
fn bits_eq(a: &[f64], b: &[f64]) -> bool { a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) }

/// Best-of-5 seconds per call; each sample repeats the call until it has run for ≥ 80 ms.
fn time_it<F: FnMut()>(mut f: F) -> f64 {
    f();
    let t = Instant::now(); f(); let one = t.elapsed().as_secs_f64().max(1e-9);
    let iters = ((0.08 / one).ceil() as usize).max(1);
    (0..5).map(|_| { let t = Instant::now(); for _ in 0..iters { f(); } t.elapsed().as_secs_f64() / iters as f64 }).fold(f64::INFINITY, f64::min)
}

struct Data { q: Vec<f64>, k: Vec<f64>, v: Vec<f64>, dout: Vec<f64> }
fn data<const D: usize>(n: usize, seed: u64) -> Data { let mut r = Rng(seed); Data { q: r.fill(n * D), k: r.fill(n * D), v: r.fill(n * D), dout: r.fill(n * D) } }

fn pairs(n: usize, causal: bool) -> f64 { if causal { (n * (n + 1) / 2) as f64 } else { (n * n) as f64 } }
// Nominal executed flops of the fused algorithm (mul and add counted separately; exp not counted):
//   forward  4D / pair (QKᵀ 2D, PV 2D);   backward 14D / pair (dQ pass: S 2D, dP 2D, dQ 2D; dK/dV pass: S 2D, dP 2D, dV 2D, dK 2D) + 2nD for Δ.
fn fwd_flops(n: usize, d: usize, c: bool) -> f64 { pairs(n, c) * 4.0 * d as f64 }
fn bwd_flops(n: usize, d: usize, c: bool) -> f64 { pairs(n, c) * 14.0 * d as f64 + 2.0 * (n * d) as f64 }
// Compulsory (cold, each tensor touched once) DRAM traffic in bytes.
fn fwd_bytes(n: usize, d: usize) -> f64 { 8.0 * (4 * n * d + n) as f64 }                 // read Q K V, write O + L
fn bwd_bytes(n: usize, d: usize) -> f64 { 8.0 * (8 * n * d + 2 * n) as f64 }             // read Q K V O dO L, write dQ dK dV Δ

fn verify<const D: usize>(cfgs: &[Config]) -> Result<(), String> {
    for &(n, causal) in &[(1usize, false), (7, true), (64, false), (64, true)] {
        let c = data::<D>(n, 99 + n as u64);
        let (mut o, mut lse, mut s, mut dp) = (vec![0.0; n * D], vec![0.0; n], vec![0.0; n * n], vec![0.0; n * n]);
        let (mut dq, mut dk, mut dv) = (vec![0.0; n * D], vec![0.0; n * D], vec![0.0; n * D]);
        forward_reference::<D>(&c.q, &c.k, &c.v, &mut o, &mut lse, &mut s, n, causal).map_err(|e| e.to_string())?;
        backward_reference::<D>(&c.q, &c.k, &c.v, &c.dout, &mut dq, &mut dk, &mut dv, &mut s, &mut dp, n, causal).map_err(|e| e.to_string())?;
        let mut base: Vec<(Isa, [Vec<f64>; 5])> = vec![];
        for &cfg in cfgs {
            let (mut o2, mut l2, mut dq2, mut dk2, mut dv2, mut de) = (vec![0.0; n * D], vec![0.0; n], vec![0.0; n * D], vec![0.0; n * D], vec![0.0; n * D], vec![0.0; n]);
            forward::<D>(cfg, &c.q, &c.k, &c.v, &mut o2, &mut l2, n, causal).map_err(|e| e.to_string())?;
            backward::<D>(cfg, &c.q, &c.k, &c.v, &o2, &l2, &c.dout, &mut dq2, &mut dk2, &mut dv2, &mut de, n, causal).map_err(|e| e.to_string())?;
            for (nm, a, b) in [("o", &o, &o2), ("lse", &lse, &l2), ("dq", &dq, &dq2), ("dk", &dk, &dk2), ("dv", &dv, &dv2)] {
                if maxd(a, b) >= TOL { return Err(format!("D={D} n={n} causal={causal} {:?}: {nm} differs from reference by {:e}", cfg, maxd(a, b))); }
            }
            let got = [o2, l2, dq2, dk2, dv2];
            if cfg.threads == 1 { base.push((cfg.isa, got)); }
            else {
                let b = &base.iter().find(|(isa, _)| *isa == cfg.isa).expect("cfgs lists threads=1 before threaded for each ISA").1;
                if !got.iter().zip(b.iter()).all(|(x, y)| bits_eq(x, y)) {
                    return Err(format!("D={D} n={n} causal={causal} {:?}: threaded result not bit-identical to single-thread", cfg));
                }
            }
        }
    }
    Ok(())
}

// ─────────── machine peaks ───────────
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn peak_fma_gflops() -> f64 {
    use core::arch::x86_64::*;
    let iters = 40_000_000u64;
    let t = Instant::now();
    let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) = (_mm256_set1_pd(1.0), _mm256_set1_pd(2.0), _mm256_set1_pd(3.0), _mm256_set1_pd(4.0), _mm256_set1_pd(5.0), _mm256_set1_pd(6.0), _mm256_set1_pd(7.0), _mm256_set1_pd(8.0));
    let (m, c) = (_mm256_set1_pd(0.999999), _mm256_set1_pd(1e-9));
    for _ in 0..iters {
        a0 = _mm256_fmadd_pd(a0, m, c); a1 = _mm256_fmadd_pd(a1, m, c); a2 = _mm256_fmadd_pd(a2, m, c); a3 = _mm256_fmadd_pd(a3, m, c);
        a4 = _mm256_fmadd_pd(a4, m, c); a5 = _mm256_fmadd_pd(a5, m, c); a6 = _mm256_fmadd_pd(a6, m, c); a7 = _mm256_fmadd_pd(a7, m, c);
    }
    let dt = t.elapsed().as_secs_f64();
    let s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3)), _mm256_add_pd(_mm256_add_pd(a4, a5), _mm256_add_pd(a6, a7)));
    let mut out = [0f64; 4]; _mm256_storeu_pd(out.as_mut_ptr(), s); std::hint::black_box(out);
    iters as f64 * 8.0 * 4.0 * 2.0 / dt / 1e9
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn read_bw(buf: &[f64], reps: usize) -> f64 {
    use core::arch::x86_64::*;
    let t = Instant::now(); let mut acc = _mm256_setzero_pd();
    for _ in 0..reps { let p = buf.as_ptr(); let mut i = 0; while i + 16 <= buf.len() {
        acc = _mm256_add_pd(acc, _mm256_add_pd(_mm256_add_pd(_mm256_loadu_pd(p.add(i)), _mm256_loadu_pd(p.add(i + 4))), _mm256_add_pd(_mm256_loadu_pd(p.add(i + 8)), _mm256_loadu_pd(p.add(i + 12))))); i += 16; } }
    let dt = t.elapsed().as_secs_f64(); let mut out = [0f64; 4]; _mm256_storeu_pd(out.as_mut_ptr(), acc); std::hint::black_box(out);
    (buf.len() * 8 * reps) as f64 / dt / 1e9
}

/// Aggregate AVX2+FMA peak with `t` threads running concurrently (sum of per-thread rates; hardware threads that share a core split it).
fn peak_mt(t: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    { if Isa::avx2_fma_available() { return std::thread::scope(|s| { let hs: Vec<_> = (0..t).map(|_| s.spawn(|| unsafe { peak_fma_gflops() })).collect(); hs.into_iter().map(|h| h.join().unwrap()).sum() }); } }
    let _ = t; f64::NAN
}
/// Nanoseconds per f64::exp over arguments spread across the softmax range [-20, 0] (throughput of independent calls).
fn exp_ns() -> f64 {
    let xs: Vec<f64> = (0..1 << 16).map(|i| -(i as f64) / (1 << 16) as f64 * 20.0).collect();
    let mut sink = 0.0;
    let t = time_it(|| { let mut a = 0.0; for &x in &xs { a += x.exp(); } sink += a; std::hint::black_box(sink); });
    t / xs.len() as f64 * 1e9
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("--write-golden") {
        let p = args.get(2).cloned().unwrap_or_else(|| "../../bench/shared/attention_golden.txt".into());
        std::fs::write(&p, golden_text()).expect("write golden"); println!("wrote {p}"); return;
    }

    let hw = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
    let simd = Isa::avx2_fma_available();
    let mut tlist = vec![2usize]; if hw >= 4 { tlist.push(4); } if hw > 4 { tlist.push(hw); }
    println!("attention bench — {hw} hardware threads, avx2+fma: {simd}, BK={BK}");

    // ── correctness gate ──
    let mut cfgs = vec![Config::SCALAR]; for &t in &tlist { cfgs.push(Config { isa: Isa::Scalar, threads: t }); }
    if simd { cfgs.push(Config { isa: Isa::Avx2Fma, threads: 1 }); for &t in &tlist { cfgs.push(Config { isa: Isa::Avx2Fma, threads: t }); } }
    for r in [verify::<16>(&cfgs), verify::<64>(&cfgs)] { if let Err(e) = r { eprintln!("CORRECTNESS GATE FAILED — not benchmarking: {e}"); std::process::exit(1); } }
    println!("correctness gate: fused fwd/bwd == materialised reference (tol {TOL:e}), threaded == single-thread bit-for-bit — ok\n");

    // ── machine peaks ──
    let mut peaks: Vec<(usize, f64)> = vec![(1, peak_mt(1))]; for &t in &tlist { peaks.push((t, peak_mt(t))); }
    let (bw_l2, bw_dram);
    #[cfg(target_arch = "x86_64")]
    {
        if simd {
            let small = vec![1.0f64; 32 * 1024];            // 256 KB: L2-resident
            bw_l2 = unsafe { read_bw(&small, 4000) };
            let big = vec![1.0f64; 32 * 1024 * 1024];       // 256 MB: DRAM
            bw_dram = unsafe { read_bw(&big, 3) };
        } else { bw_l2 = f64::NAN; bw_dram = f64::NAN; }
    }
    #[cfg(not(target_arch = "x86_64"))]
    { bw_l2 = f64::NAN; bw_dram = f64::NAN; }
    let peak = peaks[0].1;
    println!("machine: AVX2+FMA peak {}", peaks.iter().map(|(t, g)| format!("{g:.1} GF/s @{t}t")).collect::<Vec<_>>().join(", "));
    println!("         single-thread read bandwidth: L2-resident {bw_l2:.1} GB/s, DRAM {bw_dram:.1} GB/s | DRAM ridge point {:.1} flop/B | libm exp {:.1} ns/call\n", peak / bw_dram, exp_ns());

    bench::<16>("word-sized", 8, false, &tlist, &peaks, bw_dram);
    bench::<64>("small", 128, true, &tlist, &peaks, bw_dram);
    bench::<64>("medium", 512, false, &tlist, &peaks, bw_dram);
    bench::<64>("medium", 512, true, &tlist, &peaks, bw_dram);
    bench::<64>("large", 2048, true, &tlist, &peaks, bw_dram);
}

fn bench<const D: usize>(tag: &str, n: usize, causal: bool, tlist: &[usize], peaks: &[(usize, f64)], bw_dram: f64) {
    let simd = Isa::avx2_fma_available();
    let c = data::<D>(n, 42);
    let ws = (8 * n * D * 8) as f64 / 1024.0;
    println!("── {tag}: n={n} D={D} causal={causal}  (working set Q,K,V,O,dO,dQ,dK,dV ≈ {ws:.0} KiB) ──", );
    println!("{:<34} {:>10} {:>8} {:>8} {:>8} | {:>10} {:>8} {:>8} {:>8}", "implementation", "fwd ms", "GF/s", "GB/s", "%peak", "bwd ms", "GF/s", "GB/s", "%peak");

    let (mut o, mut lse) = (vec![0.0; n * D], vec![0.0; n]);
    let (mut dq, mut dk, mut dv, mut delta) = (vec![0.0; n * D], vec![0.0; n * D], vec![0.0; n * D], vec![0.0; n]);
    let (mut s, mut dp) = (vec![0.0; n * n], vec![0.0; n * n]);
    let (ff, bf, fb, bb) = (fwd_flops(n, D, causal), bwd_flops(n, D, causal), fwd_bytes(n, D), bwd_bytes(n, D));
    let peak_of = |t: usize| peaks.iter().find(|(x, _)| *x == t).map(|(_, g)| *g).unwrap();
    let row = |label: &str, threads: usize, tf: f64, tb: f64| {
        let peak = peak_of(threads);   // %peak is relative to the AVX2+FMA peak measured with the same number of threads
        println!("{label:<34} {:>10.3} {:>8.2} {:>8.2} {:>7.1}% | {:>10.3} {:>8.2} {:>8.2} {:>7.1}%",
                 tf * 1e3, ff / tf / 1e9, fb / tf / 1e9, ff / tf / 1e9 / peak * 100.0, tb * 1e3, bf / tb / 1e9, bb / tb / 1e9, bf / tb / 1e9 / peak * 100.0);
    };

    // reference (materialised). Backward needs no forward output, so it is timed on its own.
    let tf = time_it(|| { forward_reference::<D>(&c.q, &c.k, &c.v, &mut o, &mut lse, &mut s, n, causal).unwrap(); std::hint::black_box(&o); });
    let tb = time_it(|| { backward_reference::<D>(&c.q, &c.k, &c.v, &c.dout, &mut dq, &mut dk, &mut dv, &mut s, &mut dp, n, causal).unwrap(); std::hint::black_box(&dq); });
    row("reference (materialised, scalar)", 1, tf, tb);

    let mut run = |label: String, cfg: Config| {
        forward::<D>(cfg, &c.q, &c.k, &c.v, &mut o, &mut lse, n, causal).unwrap();
        let tf = time_it(|| { forward::<D>(cfg, &c.q, &c.k, &c.v, &mut o, &mut lse, n, causal).unwrap(); std::hint::black_box(&o); });
        let tb = time_it(|| { backward::<D>(cfg, &c.q, &c.k, &c.v, &o, &lse, &c.dout, &mut dq, &mut dk, &mut dv, &mut delta, n, causal).unwrap(); std::hint::black_box(&dq); });
        row(&label, cfg.threads, tf, tb);
    };
    run("fused scalar-ops, 1 thread".into(), Config::SCALAR);
    for &t in tlist { run(format!("fused scalar-ops, {t} threads"), Config { isa: Isa::Scalar, threads: t }); }
    if simd {
        run("fused avx2+fma, 1 thread".into(), Config { isa: Isa::Avx2Fma, threads: 1 });
        for &t in tlist { run(format!("fused avx2+fma, {t} threads"), Config { isa: Isa::Avx2Fma, threads: t }); }
    }
    let ai_f = ff / fb; let ai_b = bf / bb; let ridge = peak_of(1) / bw_dram;
    println!("arithmetic intensity (compulsory bytes): fwd {ai_f:.1} flop/B, bwd {ai_b:.1} flop/B vs DRAM ridge {ridge:.1} flop/B → {}\n",
             if ai_f.min(ai_b) > ridge { "above the ridge: not DRAM-bandwidth-bound" } else { "below the ridge: DRAM-bandwidth-bound is possible" });
}
