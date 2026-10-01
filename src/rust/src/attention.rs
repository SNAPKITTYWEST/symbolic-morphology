//! Scaled dot-product attention: forward and backward kernels, allocation-free, zero dependencies.
//!
//! Layout: every tensor is row-major `f64`, `x[i * D + d]`, with `n` rows (sequence positions) and `D` columns.
//! Single head, no learned projections: the caller supplies Q, K, V.
//!
//! Forward   S = scale·Q·Kᵀ  (scale = 1/√D, causal: S[i][j] = -∞ for j > i)
//!           P = softmax_row(S),   O = P·V,   L_i = logsumexp_j S[i][j]      (L is saved for the backward pass)
//! Backward  given dO:
//!           dV = Pᵀ·dO
//!           dP = dO·Vᵀ
//!           Δ_i = Σ_j P[i][j]·dP[i][j] = dO_i · O_i
//!           dS = P ⊙ (dP − Δ)
//!           dQ = scale·dS·K,   dK = scale·dSᵀ·Q
//! The fused backward never stores P: it recomputes P[i][j] = exp(scale·q_i·k_j − L_i) from the saved L.
//!
//! Implementations (all produce the same maths; see docs/ATTENTION.md):
//!   * `forward_reference` / `backward_reference` — textbook, materialise n×n matrices. Independent check path.
//!   * `forward` / `backward` with `Config { isa: Isa::Scalar, threads: 1 }` — fused scalar reference path.
//!   * `Isa::Avx2Fma` — explicit AVX2+FMA intrinsics for the D-length dot/axpy/scale primitives.
//!   * `threads > 1` — deterministic static row partition; every output element is written by exactly one thread
//!     and is computed with the same operation order as the single-thread path, so results are bit-identical
//!     for any thread count.
//!
//! Error policy: shape mismatches, non-finite inputs, zero/oversized thread counts, an ISA the CPU does not
//! support and SIMD with D % 4 != 0 all return `Err(AttnError)`. Nothing silently falls back.
use std::fmt;

/// Key-block size of the online-softmax recurrence.
pub const BK: usize = 16;
/// Upper bound on `Config::threads` (partition tables are fixed-size stack arrays).
pub const MAX_THREADS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttnError {
    Shape(&'static str),
    NonFinite(&'static str),
    ZeroThreads,
    TooManyThreads(usize),
    DimNotMultipleOf4(usize),
    UnsupportedIsa(&'static str),
}

impl fmt::Display for AttnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttnError::Shape(w) => write!(f, "attention: bad shape: {w}"),
            AttnError::NonFinite(w) => write!(f, "attention: non-finite value in {w}"),
            AttnError::ZeroThreads => write!(f, "attention: threads must be >= 1"),
            AttnError::TooManyThreads(t) => write!(f, "attention: threads {t} exceeds MAX_THREADS {MAX_THREADS}"),
            AttnError::DimNotMultipleOf4(d) => write!(f, "attention: AVX2 path needs D % 4 == 0, got D = {d}"),
            AttnError::UnsupportedIsa(w) => write!(f, "attention: unsupported instruction set: {w}"),
        }
    }
}
impl std::error::Error for AttnError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// Plain loops. LLVM's auto-vectoriser may still vectorise the element-wise loops (not the dot reductions).
    Scalar,
    /// Explicit AVX2 + FMA intrinsics (x86_64 only, runtime-detected).
    Avx2Fma,
}

impl Isa {
    /// Best ISA this CPU supports.
    pub fn detect() -> Isa { if Isa::avx2_fma_available() { Isa::Avx2Fma } else { Isa::Scalar } }
    pub fn avx2_fma_available() -> bool {
        #[cfg(target_arch = "x86_64")]
        { is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") }
        #[cfg(not(target_arch = "x86_64"))]
        { false }
    }
    pub fn name(self) -> &'static str { match self { Isa::Scalar => "scalar", Isa::Avx2Fma => "avx2+fma" } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config { pub isa: Isa, pub threads: usize }

impl Config {
    pub const SCALAR: Config = Config { isa: Isa::Scalar, threads: 1 };
    pub fn auto() -> Config { Config { isa: Isa::detect(), threads: 1 } }
}

fn check_cfg<const D: usize>(cfg: Config) -> Result<(), AttnError> {
    if D == 0 { return Err(AttnError::Shape("D must be > 0")); }
    if cfg.threads == 0 { return Err(AttnError::ZeroThreads); }
    if cfg.threads > MAX_THREADS { return Err(AttnError::TooManyThreads(cfg.threads)); }
    if cfg.isa == Isa::Avx2Fma {
        if D % 4 != 0 { return Err(AttnError::DimNotMultipleOf4(D)); }
        if !Isa::avx2_fma_available() { return Err(AttnError::UnsupportedIsa("avx2+fma not available on this CPU")); }
    }
    Ok(())
}

fn nd<const D: usize>(n: usize) -> Result<usize, AttnError> { n.checked_mul(D).ok_or(AttnError::Shape("n*D overflows usize")) }
fn len_is(x: &[f64], want: usize, what: &'static str) -> Result<(), AttnError> {
    if x.len() == want { Ok(()) } else { Err(AttnError::Shape(what)) }
}
fn finite(x: &[f64], what: &'static str) -> Result<(), AttnError> {
    if x.iter().all(|v| v.is_finite()) { Ok(()) } else { Err(AttnError::NonFinite(what)) }
}

// ───────────────────────────── element-wise primitives ─────────────────────────────

trait Ops {
    fn dot<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64;
    fn axpy<const D: usize>(y: &mut [f64; D], a: f64, x: &[f64; D]);
    fn scale<const D: usize>(y: &mut [f64; D], a: f64);
}

struct ScalarOps;
impl Ops for ScalarOps {
    #[inline(always)]
    fn dot<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 { let mut s = 0.0; for d in 0..D { s += a[d] * b[d]; } s }
    #[inline(always)]
    fn axpy<const D: usize>(y: &mut [f64; D], a: f64, x: &[f64; D]) { for d in 0..D { y[d] += a * x[d]; } }
    #[inline(always)]
    fn scale<const D: usize>(y: &mut [f64; D], a: f64) { for d in 0..D { y[d] *= a; } }
}

#[cfg(target_arch = "x86_64")]
struct Avx2Ops;
#[cfg(target_arch = "x86_64")]
impl Ops for Avx2Ops {
    // Only ever inlined into the `#[target_feature(enable = "avx2,fma")]` wrappers below, which are only
    // reached after `check_cfg` has verified D % 4 == 0 and runtime AVX2+FMA support.
    #[inline(always)]
    fn dot<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
        use core::arch::x86_64::*;
        unsafe {
            let (pa, pb) = (a.as_ptr(), b.as_ptr());
            let mut acc0 = _mm256_setzero_pd(); let mut acc1 = _mm256_setzero_pd();
            let mut d = 0;
            while d + 8 <= D {
                acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(d)), _mm256_loadu_pd(pb.add(d)), acc0);
                acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(d + 4)), _mm256_loadu_pd(pb.add(d + 4)), acc1);
                d += 8;
            }
            if d + 4 <= D { acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(d)), _mm256_loadu_pd(pb.add(d)), acc0); }
            let acc = _mm256_add_pd(acc0, acc1);
            let s = _mm_add_pd(_mm256_castpd256_pd128(acc), _mm256_extractf128_pd(acc, 1));
            _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)))
        }
    }
    #[inline(always)]
    fn axpy<const D: usize>(y: &mut [f64; D], a: f64, x: &[f64; D]) {
        use core::arch::x86_64::*;
        unsafe {
            let (py, px) = (y.as_mut_ptr(), x.as_ptr()); let va = _mm256_set1_pd(a);
            let mut d = 0;
            while d + 4 <= D { _mm256_storeu_pd(py.add(d), _mm256_fmadd_pd(va, _mm256_loadu_pd(px.add(d)), _mm256_loadu_pd(py.add(d)))); d += 4; }
        }
    }
    #[inline(always)]
    fn scale<const D: usize>(y: &mut [f64; D], a: f64) {
        use core::arch::x86_64::*;
        unsafe {
            let py = y.as_mut_ptr(); let va = _mm256_set1_pd(a);
            let mut d = 0;
            while d + 4 <= D { _mm256_storeu_pd(py.add(d), _mm256_mul_pd(va, _mm256_loadu_pd(py.add(d)))); d += 4; }
        }
    }
}

#[inline(always)]
fn row<const D: usize>(x: &[f64], i: usize) -> &[f64; D] { x[i * D..(i + 1) * D].try_into().unwrap() }

// ───────────────────────────── row-range kernels (ISA-generic) ─────────────────────────────
// Each kernel computes rows [i0, i0 + rows) of its outputs; output slices are the sub-slices for those rows.

/// Online-softmax forward for query rows i0..i0+lse.len(). Writes O rows and L (logsumexp).
#[inline(always)]
fn fwd_rows<O: Ops, const D: usize>(q: &[f64], k: &[f64], v: &[f64], o: &mut [f64], lse: &mut [f64], i0: usize, n: usize, causal: bool) {
    let scale = 1.0 / (D as f64).sqrt();
    for r in 0..lse.len() {
        let i = i0 + r;
        let qi = row::<D>(q, i);
        let kmax = if causal { i + 1 } else { n };
        let mut m = f64::NEG_INFINITY; let mut l = 0.0; let mut acc = [0f64; D];
        let mut j0 = 0;
        while j0 < kmax {
            let bl = BK.min(kmax - j0);
            let mut s = [0f64; BK]; let mut bm = m;
            for b in 0..bl { let x = O::dot(qi, row::<D>(k, j0 + b)) * scale; s[b] = x; if x > bm { bm = x; } }
            let corr = if m == f64::NEG_INFINITY { 0.0 } else { (m - bm).exp() };
            l *= corr; O::scale(&mut acc, corr);
            for b in 0..bl { let p = (s[b] - bm).exp(); l += p; O::axpy(&mut acc, p, row::<D>(v, j0 + b)); }
            m = bm; j0 += bl;
        }
        O::scale(&mut acc, 1.0 / l);
        o[r * D..(r + 1) * D].copy_from_slice(&acc);
        lse[r] = m + l.ln();
    }
}

/// Δ_i = dO_i · O_i for all rows.
#[inline(always)]
fn delta_rows<O: Ops, const D: usize>(o: &[f64], do_: &[f64], delta: &mut [f64]) {
    for i in 0..delta.len() { delta[i] = O::dot(row::<D>(do_, i), row::<D>(o, i)); }
}

/// dQ for query rows i0..i0+rows: dQ_i = scale · Σ_j dS[i][j]·k_j, with P recomputed from L.
#[inline(always)]
fn dq_rows<O: Ops, const D: usize>(q: &[f64], k: &[f64], v: &[f64], do_: &[f64], lse: &[f64], delta: &[f64], dq: &mut [f64], i0: usize, n: usize, causal: bool) {
    let scale = 1.0 / (D as f64).sqrt();
    for r in 0..dq.len() / D {
        let i = i0 + r;
        let (qi, doi, li, di) = (row::<D>(q, i), row::<D>(do_, i), lse[i], delta[i]);
        let kmax = if causal { i + 1 } else { n };
        let mut acc = [0f64; D];
        for j in 0..kmax {
            let p = (O::dot(qi, row::<D>(k, j)) * scale - li).exp();
            let ds = p * (O::dot(doi, row::<D>(v, j)) - di);
            O::axpy(&mut acc, ds, row::<D>(k, j));
        }
        O::scale(&mut acc, scale);
        dq[r * D..(r + 1) * D].copy_from_slice(&acc);
    }
}

/// dK and dV for key rows j0..j0+rows: dV_j = Σ_i P[i][j]·dO_i, dK_j = scale · Σ_i dS[i][j]·q_i.
#[inline(always)]
fn dkv_rows<O: Ops, const D: usize>(q: &[f64], k: &[f64], v: &[f64], do_: &[f64], lse: &[f64], delta: &[f64], dk: &mut [f64], dv: &mut [f64], j0: usize, n: usize, causal: bool) {
    let scale = 1.0 / (D as f64).sqrt();
    for r in 0..dk.len() / D {
        let j = j0 + r;
        let (kj, vj) = (row::<D>(k, j), row::<D>(v, j));
        let imin = if causal { j } else { 0 };
        let mut acck = [0f64; D]; let mut accv = [0f64; D];
        for i in imin..n {
            let (qi, doi) = (row::<D>(q, i), row::<D>(do_, i));
            let p = (O::dot(qi, kj) * scale - lse[i]).exp();
            O::axpy(&mut accv, p, doi);
            let ds = p * (O::dot(doi, vj) - delta[i]);
            O::axpy(&mut acck, ds, qi);
        }
        O::scale(&mut acck, scale);
        dk[r * D..(r + 1) * D].copy_from_slice(&acck);
        dv[r * D..(r + 1) * D].copy_from_slice(&accv);
    }
}

// AVX2+FMA entry points: the generic bodies are `inline(always)`, so they are compiled with avx2+fma enabled here.
macro_rules! avx2_entry {
    ($name:ident => $inner:ident($($arg:ident : $ty:ty),*)) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $name<const D: usize>($($arg: $ty),*) { $inner::<Avx2Ops, D>($($arg),*) }
    };
}
avx2_entry!(fwd_rows_avx2 => fwd_rows(q: &[f64], k: &[f64], v: &[f64], o: &mut [f64], lse: &mut [f64], i0: usize, n: usize, causal: bool));
avx2_entry!(delta_rows_avx2 => delta_rows(o: &[f64], do_: &[f64], delta: &mut [f64]));
avx2_entry!(dq_rows_avx2 => dq_rows(q: &[f64], k: &[f64], v: &[f64], do_: &[f64], lse: &[f64], delta: &[f64], dq: &mut [f64], i0: usize, n: usize, causal: bool));
avx2_entry!(dkv_rows_avx2 => dkv_rows(q: &[f64], k: &[f64], v: &[f64], do_: &[f64], lse: &[f64], delta: &[f64], dk: &mut [f64], dv: &mut [f64], j0: usize, n: usize, causal: bool));

// Dispatch. Callers have run `check_cfg`, so Isa::Avx2Fma implies the CPU supports it and D % 4 == 0.
fn run_fwd<const D: usize>(isa: Isa, q: &[f64], k: &[f64], v: &[f64], o: &mut [f64], lse: &mut [f64], i0: usize, n: usize, causal: bool) {
    match isa {
        Isa::Scalar => fwd_rows::<ScalarOps, D>(q, k, v, o, lse, i0, n, causal),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma => unsafe { fwd_rows_avx2::<D>(q, k, v, o, lse, i0, n, causal) },
        #[cfg(not(target_arch = "x86_64"))]
        Isa::Avx2Fma => unreachable!("check_cfg rejects Avx2Fma off x86_64"),
    }
}
fn run_delta<const D: usize>(isa: Isa, o: &[f64], do_: &[f64], delta: &mut [f64]) {
    match isa {
        Isa::Scalar => delta_rows::<ScalarOps, D>(o, do_, delta),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma => unsafe { delta_rows_avx2::<D>(o, do_, delta) },
        #[cfg(not(target_arch = "x86_64"))]
        Isa::Avx2Fma => unreachable!("check_cfg rejects Avx2Fma off x86_64"),
    }
}
fn run_dq<const D: usize>(isa: Isa, q: &[f64], k: &[f64], v: &[f64], do_: &[f64], lse: &[f64], delta: &[f64], dq: &mut [f64], i0: usize, n: usize, causal: bool) {
    match isa {
        Isa::Scalar => dq_rows::<ScalarOps, D>(q, k, v, do_, lse, delta, dq, i0, n, causal),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma => unsafe { dq_rows_avx2::<D>(q, k, v, do_, lse, delta, dq, i0, n, causal) },
        #[cfg(not(target_arch = "x86_64"))]
        Isa::Avx2Fma => unreachable!("check_cfg rejects Avx2Fma off x86_64"),
    }
}
fn run_dkv<const D: usize>(isa: Isa, q: &[f64], k: &[f64], v: &[f64], do_: &[f64], lse: &[f64], delta: &[f64], dk: &mut [f64], dv: &mut [f64], j0: usize, n: usize, causal: bool) {
    match isa {
        Isa::Scalar => dkv_rows::<ScalarOps, D>(q, k, v, do_, lse, delta, dk, dv, j0, n, causal),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma => unsafe { dkv_rows_avx2::<D>(q, k, v, do_, lse, delta, dk, dv, j0, n, causal) },
        #[cfg(not(target_arch = "x86_64"))]
        Isa::Avx2Fma => unreachable!("check_cfg rejects Avx2Fma off x86_64"),
    }
}

// ───────────────────────────── deterministic partitioning ─────────────────────────────

#[derive(Clone, Copy)]
enum Skew { Uniform, Rising, Falling }

/// Contiguous row ranges with roughly equal work. Pure integer/IEEE arithmetic, so identical on every run.
/// Rising: row i costs ∝ i+1 (causal forward / dQ). Falling: row j costs ∝ n−j (causal dK/dV).
/// The partition changes only *who* computes a row, never *how* — each row's arithmetic is thread-independent.
fn partition(n: usize, threads: usize, skew: Skew) -> ([usize; MAX_THREADS + 1], usize) {
    let parts = threads.min(n).max(1);
    let mut b = [0usize; MAX_THREADS + 1];
    for t in 0..=parts {
        let f = t as f64 / parts as f64;
        b[t] = match skew {
            _ if t == 0 => 0,
            _ if t == parts => n,
            Skew::Uniform => n * t / parts,
            Skew::Rising => ((n as f64) * f.sqrt()).round() as usize,
            Skew::Falling => n - ((n as f64) * (1.0 - f).sqrt()).round() as usize,
        }.min(n);
    }
    (b, parts)
}

// ───────────────────────────── public API ─────────────────────────────

/// Fused forward. Writes `o` (n×D) and `lse` (n); both are fully overwritten.
pub fn forward<const D: usize>(cfg: Config, q: &[f64], k: &[f64], v: &[f64], o: &mut [f64], lse: &mut [f64], n: usize, causal: bool) -> Result<(), AttnError> {
    check_cfg::<D>(cfg)?;
    let w = nd::<D>(n)?;
    len_is(q, w, "q")?; len_is(k, w, "k")?; len_is(v, w, "v")?; len_is(o, w, "o")?; len_is(lse, n, "lse")?;
    finite(q, "q")?; finite(k, "k")?; finite(v, "v")?;
    if n == 0 { return Ok(()); }
    let (b, parts) = partition(n, cfg.threads, if causal { Skew::Rising } else { Skew::Uniform });
    if parts == 1 { run_fwd::<D>(cfg.isa, q, k, v, o, lse, 0, n, causal); return Ok(()); }
    std::thread::scope(|s| {
        let (mut orest, mut lrest) = (o, lse);
        for t in 0..parts {
            let (i0, rows) = (b[t], b[t + 1] - b[t]);
            let (oc, on) = orest.split_at_mut(rows * D); orest = on;
            let (lc, ln) = lrest.split_at_mut(rows); lrest = ln;
            s.spawn(move || run_fwd::<D>(cfg.isa, q, k, v, oc, lc, i0, n, causal));
        }
    });
    Ok(())
}

/// Fused backward. Inputs: q, k, v, the forward's `o` and `lse`, and the upstream gradient `do_`.
/// Outputs `dq`, `dk`, `dv` (n×D, fully overwritten, not accumulated) and `delta` (n scratch, Δ_i = dO_i·O_i).
/// Two row-parallel passes (dQ by query row, dK/dV by key row), so no two threads ever touch the same output
/// element and every sum has a fixed order (ascending j, resp. ascending i).
pub fn backward<const D: usize>(cfg: Config, q: &[f64], k: &[f64], v: &[f64], o: &[f64], lse: &[f64], do_: &[f64],
                                dq: &mut [f64], dk: &mut [f64], dv: &mut [f64], delta: &mut [f64], n: usize, causal: bool) -> Result<(), AttnError> {
    check_cfg::<D>(cfg)?;
    let w = nd::<D>(n)?;
    for (x, what) in [(q, "q"), (k, "k"), (v, "v"), (o, "o"), (do_, "do")] { len_is(x, w, what)?; finite(x, what)?; }
    len_is(lse, n, "lse")?; finite(lse, "lse")?;
    len_is(dq, w, "dq")?; len_is(dk, w, "dk")?; len_is(dv, w, "dv")?; len_is(delta, n, "delta")?;
    if n == 0 { return Ok(()); }
    run_delta::<D>(cfg.isa, o, do_, delta);
    let delta: &[f64] = delta;

    let (b, parts) = partition(n, cfg.threads, if causal { Skew::Rising } else { Skew::Uniform });
    if parts == 1 { run_dq::<D>(cfg.isa, q, k, v, do_, lse, delta, dq, 0, n, causal); }
    else {
        std::thread::scope(|s| {
            let mut rest = &mut *dq;
            for t in 0..parts {
                let (i0, rows) = (b[t], b[t + 1] - b[t]);
                let (c, nx) = rest.split_at_mut(rows * D); rest = nx;
                s.spawn(move || run_dq::<D>(cfg.isa, q, k, v, do_, lse, delta, c, i0, n, causal));
            }
        });
    }

    let (b, parts) = partition(n, cfg.threads, if causal { Skew::Falling } else { Skew::Uniform });
    if parts == 1 { run_dkv::<D>(cfg.isa, q, k, v, do_, lse, delta, dk, dv, 0, n, causal); }
    else {
        std::thread::scope(|s| {
            let (mut krest, mut vrest) = (&mut *dk, &mut *dv);
            for t in 0..parts {
                let (j0, rows) = (b[t], b[t + 1] - b[t]);
                let (kc, kn) = krest.split_at_mut(rows * D); krest = kn;
                let (vc, vn) = vrest.split_at_mut(rows * D); vrest = vn;
                s.spawn(move || run_dkv::<D>(cfg.isa, q, k, v, do_, lse, delta, kc, vc, j0, n, causal));
            }
        });
    }
    Ok(())
}

/// Textbook forward: materialises S (n×n scratch `s`) and normalises it explicitly. Independent of the fused kernel.
pub fn forward_reference<const D: usize>(q: &[f64], k: &[f64], v: &[f64], o: &mut [f64], lse: &mut [f64], s: &mut [f64], n: usize, causal: bool) -> Result<(), AttnError> {
    check_cfg::<D>(Config::SCALAR)?;
    let w = nd::<D>(n)?;
    len_is(q, w, "q")?; len_is(k, w, "k")?; len_is(v, w, "v")?; len_is(o, w, "o")?; len_is(lse, n, "lse")?;
    len_is(s, n.checked_mul(n).ok_or(AttnError::Shape("n*n overflows usize"))?, "scratch s (n*n)")?;
    finite(q, "q")?; finite(k, "k")?; finite(v, "v")?;
    let scale = 1.0 / (D as f64).sqrt();
    for i in 0..n {
        let kmax = if causal { i + 1 } else { n };
        let mut m = f64::NEG_INFINITY;
        for j in 0..kmax {
            let mut dot = 0.0; for d in 0..D { dot += q[i * D + d] * k[j * D + d]; }
            let x = dot * scale; s[i * n + j] = x; if x > m { m = x; }
        }
        let mut l = 0.0;
        for j in 0..kmax { let e = (s[i * n + j] - m).exp(); s[i * n + j] = e; l += e; }
        for d in 0..D { let mut acc = 0.0; for j in 0..kmax { acc += s[i * n + j] * v[j * D + d]; } o[i * D + d] = acc / l; }
        lse[i] = m + l.ln();
    }
    Ok(())
}

/// Textbook backward straight from the formulas in the module docs. `p` and `dp` are n×n scratch.
/// Uses Δ_i = Σ_j P·dP (not dO·O), so it is an independent derivation of the same gradient.
pub fn backward_reference<const D: usize>(q: &[f64], k: &[f64], v: &[f64], do_: &[f64],
                                          dq: &mut [f64], dk: &mut [f64], dv: &mut [f64], p: &mut [f64], dp: &mut [f64], n: usize, causal: bool) -> Result<(), AttnError> {
    check_cfg::<D>(Config::SCALAR)?;
    let w = nd::<D>(n)?; let nn = n.checked_mul(n).ok_or(AttnError::Shape("n*n overflows usize"))?;
    for (x, what) in [(q, "q"), (k, "k"), (v, "v"), (do_, "do")] { len_is(x, w, what)?; finite(x, what)?; }
    len_is(dq, w, "dq")?; len_is(dk, w, "dk")?; len_is(dv, w, "dv")?; len_is(p, nn, "scratch p (n*n)")?; len_is(dp, nn, "scratch dp (n*n)")?;
    let scale = 1.0 / (D as f64).sqrt();
    // P = softmax(S) with masked entries exactly 0
    for i in 0..n {
        let kmax = if causal { i + 1 } else { n };
        let mut m = f64::NEG_INFINITY;
        for j in 0..n {
            if j >= kmax { p[i * n + j] = 0.0; continue; }
            let mut dot = 0.0; for d in 0..D { dot += q[i * D + d] * k[j * D + d]; }
            let x = dot * scale; p[i * n + j] = x; if x > m { m = x; }
        }
        let mut l = 0.0;
        for j in 0..kmax { let e = (p[i * n + j] - m).exp(); p[i * n + j] = e; l += e; }
        for j in 0..kmax { p[i * n + j] /= l; }
    }
    // dV = Pᵀ dO ; dP = dO Vᵀ
    for j in 0..n { for d in 0..D { let mut a = 0.0; for i in 0..n { a += p[i * n + j] * do_[i * D + d]; } dv[j * D + d] = a; } }
    for i in 0..n { for j in 0..n {
        let mut a = 0.0; for d in 0..D { a += do_[i * D + d] * v[j * D + d]; } dp[i * n + j] = a;
    } }
    // dS = P ⊙ (dP − Σ_j P dP), stored in dp
    for i in 0..n {
        let mut delta = 0.0; for j in 0..n { delta += p[i * n + j] * dp[i * n + j]; }
        for j in 0..n { dp[i * n + j] = p[i * n + j] * (dp[i * n + j] - delta); }
    }
    // dQ = scale·dS K ; dK = scale·dSᵀ Q
    for i in 0..n { for d in 0..D { let mut a = 0.0; for j in 0..n { a += dp[i * n + j] * k[j * D + d]; } dq[i * D + d] = a * scale; } }
    for j in 0..n { for d in 0..D { let mut a = 0.0; for i in 0..n { a += dp[i * n + j] * q[i * D + d]; } dk[j * D + d] = a * scale; } }
    Ok(())
}

// ───────────────────────────── heads / batches ─────────────────────────────

/// `h` independent heads in head-major layout [h][n][D] (each head's n×D block contiguous; lse is [h][n]).
/// Heads run one after another; `cfg.threads` parallelises within each head.
pub fn forward_heads<const D: usize>(cfg: Config, h: usize, q: &[f64], k: &[f64], v: &[f64], o: &mut [f64], lse: &mut [f64], n: usize, causal: bool) -> Result<(), AttnError> {
    let w = nd::<D>(n)?; let tot = w.checked_mul(h).ok_or(AttnError::Shape("h*n*D overflows usize"))?;
    len_is(q, tot, "q")?; len_is(k, tot, "k")?; len_is(v, tot, "v")?; len_is(o, tot, "o")?; len_is(lse, n * h, "lse")?;
    for i in 0..h {
        forward::<D>(cfg, &q[i * w..(i + 1) * w], &k[i * w..(i + 1) * w], &v[i * w..(i + 1) * w], &mut o[i * w..(i + 1) * w], &mut lse[i * n..(i + 1) * n], n, causal)?;
    }
    Ok(())
}

/// Backward for `h` head-major heads (same layout as `forward_heads`; `delta` is [h][n]).
pub fn backward_heads<const D: usize>(cfg: Config, h: usize, q: &[f64], k: &[f64], v: &[f64], o: &[f64], lse: &[f64], do_: &[f64],
                                      dq: &mut [f64], dk: &mut [f64], dv: &mut [f64], delta: &mut [f64], n: usize, causal: bool) -> Result<(), AttnError> {
    let w = nd::<D>(n)?; let tot = w.checked_mul(h).ok_or(AttnError::Shape("h*n*D overflows usize"))?;
    for (x, what) in [(q, "q"), (k, "k"), (v, "v"), (o, "o"), (do_, "do")] { len_is(x, tot, what)?; }
    len_is(lse, n * h, "lse")?; len_is(dq, tot, "dq")?; len_is(dk, tot, "dk")?; len_is(dv, tot, "dv")?; len_is(delta, n * h, "delta")?;
    for i in 0..h {
        let r = i * w..(i + 1) * w;
        backward::<D>(cfg, &q[r.clone()], &k[r.clone()], &v[r.clone()], &o[r.clone()], &lse[i * n..(i + 1) * n], &do_[r.clone()],
                      &mut dq[r.clone()], &mut dk[r.clone()], &mut dv[r.clone()], &mut delta[i * n..(i + 1) * n], n, causal)?;
    }
    Ok(())
}

/// Forward for interleaved heads: x is n×(H·D), head h owns columns h·D..(h+1)·D (the layout the first
/// version of this kernel used). Heads are gathered into caller-provided contiguous scratch (qh, kh, vh, oh: n·D; lseh: n).
pub fn forward_interleaved<const D: usize, const H: usize>(cfg: Config, q: &[f64], k: &[f64], v: &[f64], o: &mut [f64], n: usize, causal: bool,
                                                          qh: &mut [f64], kh: &mut [f64], vh: &mut [f64], oh: &mut [f64], lseh: &mut [f64]) -> Result<(), AttnError> {
    let w = H * D; let tot = n.checked_mul(w).ok_or(AttnError::Shape("n*H*D overflows usize"))?;
    len_is(q, tot, "q")?; len_is(k, tot, "k")?; len_is(v, tot, "v")?; len_is(o, tot, "o")?;
    for h in 0..H {
        for i in 0..n { for d in 0..D {
            qh[i * D + d] = q[i * w + h * D + d]; kh[i * D + d] = k[i * w + h * D + d]; vh[i * D + d] = v[i * w + h * D + d];
        } }
        forward::<D>(cfg, qh, kh, vh, oh, lseh, n, causal)?;
        for i in 0..n { for d in 0..D { o[i * w + h * D + d] = oh[i * D + d]; } }
    }
    Ok(())
}

// ───────────────────────────── deterministic inputs ─────────────────────────────

/// xorshift64 → uniform in [-1, 1). Bit-for-bit reproducible, and mirrored in src/python/attention.py so the
/// cross-language golden vectors (bench/shared/attention_golden.txt) can be regenerated from a seed.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    pub fn fill(&mut self, len: usize) -> Vec<f64> { (0..len).map(|_| self.next()).collect() }
}

// ───────────────────────────── cross-language golden vectors ─────────────────────────────

/// (n, D, causal, seed) of the golden cases. Inputs q,k,v,dO are drawn in that order from `Rng(seed)`.
pub const GOLDEN_CASES: [(usize, usize, bool, u64); 6] = [(1, 4, false, 11), (5, 8, false, 12), (5, 8, true, 13), (12, 16, false, 14), (12, 16, true, 15), (33, 64, true, 16)];

pub fn golden_inputs(n: usize, d: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut r = Rng(seed); (r.fill(n * d), r.fill(n * d), r.fill(n * d), r.fill(n * d))
}

fn golden_case<const D: usize>(n: usize, causal: bool, seed: u64) -> String {
    let (q, k, v, dout) = golden_inputs(n, D, seed);
    let (mut o, mut lse, mut s, mut dp) = (vec![0.0; n * D], vec![0.0; n], vec![0.0; n * n], vec![0.0; n * n]);
    let (mut dq, mut dk, mut dv) = (vec![0.0; n * D], vec![0.0; n * D], vec![0.0; n * D]);
    forward_reference::<D>(&q, &k, &v, &mut o, &mut lse, &mut s, n, causal).unwrap();
    backward_reference::<D>(&q, &k, &v, &dout, &mut dq, &mut dk, &mut dv, &mut s, &mut dp, n, causal).unwrap();
    let mut t = format!("case {n} {D} {} {seed}\n", causal as u8);
    for (nm, x) in [("o", &o), ("lse", &lse), ("dq", &dq), ("dk", &dk), ("dv", &dv)] {
        t.push_str(nm);
        for e in x.iter() { t.push_str(&format!(" {e:.17e}")); }
        t.push('\n');
    }
    t
}

/// Text of bench/shared/attention_golden.txt, produced by the materialised reference path.
pub fn golden_text() -> String {
    GOLDEN_CASES.iter().map(|&(n, d, causal, seed)| match d {
        4 => golden_case::<4>(n, causal, seed), 8 => golden_case::<8>(n, causal, seed),
        16 => golden_case::<16>(n, causal, seed), 64 => golden_case::<64>(n, causal, seed),
        _ => panic!("golden case with unsupported D {d}"),
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maxd(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) }
    fn bits_eq(a: &[f64], b: &[f64]) -> bool { a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) }

    /// Documented tolerance between any two implementations on inputs of magnitude ≲ 1 and n ≤ 64:
    /// FMA and reassociated dot products change rounding only, so differences stay within a few hundred ulps.
    const TOL: f64 = 1e-11;

    struct Case<const D: usize> { n: usize, q: Vec<f64>, k: Vec<f64>, v: Vec<f64>, dout: Vec<f64> }
    fn case<const D: usize>(n: usize, seed: u64) -> Case<D> {
        let mut r = Rng(seed);
        Case { n, q: r.fill(n * D), k: r.fill(n * D), v: r.fill(n * D), dout: r.fill(n * D) }
    }

    struct Out { o: Vec<f64>, lse: Vec<f64>, dq: Vec<f64>, dk: Vec<f64>, dv: Vec<f64> }
    fn run_fused<const D: usize>(c: &Case<D>, cfg: Config, causal: bool) -> Out {
        let n = c.n;
        let mut o = vec![0.0; n * D]; let mut lse = vec![0.0; n];
        let (mut dq, mut dk, mut dv, mut delta) = (vec![0.0; n * D], vec![0.0; n * D], vec![0.0; n * D], vec![0.0; n]);
        forward::<D>(cfg, &c.q, &c.k, &c.v, &mut o, &mut lse, n, causal).unwrap();
        backward::<D>(cfg, &c.q, &c.k, &c.v, &o, &lse, &c.dout, &mut dq, &mut dk, &mut dv, &mut delta, n, causal).unwrap();
        Out { o, lse, dq, dk, dv }
    }
    fn run_ref<const D: usize>(c: &Case<D>, causal: bool) -> Out {
        let n = c.n;
        let mut o = vec![0.0; n * D]; let mut lse = vec![0.0; n]; let mut s = vec![0.0; n * n]; let mut dp = vec![0.0; n * n];
        let (mut dq, mut dk, mut dv) = (vec![0.0; n * D], vec![0.0; n * D], vec![0.0; n * D]);
        forward_reference::<D>(&c.q, &c.k, &c.v, &mut o, &mut lse, &mut s, n, causal).unwrap();
        backward_reference::<D>(&c.q, &c.k, &c.v, &c.dout, &mut dq, &mut dk, &mut dv, &mut s, &mut dp, n, causal).unwrap();
        Out { o, lse, dq, dk, dv }
    }
    fn close(a: &Out, b: &Out, tol: f64, what: &str) {
        for (x, y, nm) in [(&a.o, &b.o, "o"), (&a.lse, &b.lse, "lse"), (&a.dq, &b.dq, "dq"), (&a.dk, &b.dk, "dk"), (&a.dv, &b.dv, "dv")] {
            let d = maxd(x, y); assert!(d < tol, "{what}: {nm} differs by {d:e}");
        }
    }
    fn isas() -> Vec<Isa> { let mut v = vec![Isa::Scalar]; if Isa::avx2_fma_available() { v.push(Isa::Avx2Fma); } v }

    fn all_impls_agree<const D: usize>() {
        for &n in &[1usize, 2, 3, 7, 16, 17, 33, 64] {
            for causal in [false, true] {
                let c = case::<D>(n, 1000 + n as u64);
                let r = run_ref::<D>(&c, causal);
                for isa in isas() { for threads in [1, 2, 3, 4] {
                    let f = run_fused::<D>(&c, Config { isa, threads }, causal);
                    close(&f, &r, TOL, &format!("D={D} n={n} causal={causal} {} t={threads}", isa.name()));
                } }
            }
        }
    }
    #[test] fn fused_matches_reference_d4() { all_impls_agree::<4>(); }
    #[test] fn fused_matches_reference_d8() { all_impls_agree::<8>(); }
    #[test] fn fused_matches_reference_d16() { all_impls_agree::<16>(); }
    #[test] fn fused_matches_reference_d64() { all_impls_agree::<64>(); }
    #[test] fn scalar_path_supports_d_not_multiple_of_4() {
        for causal in [false, true] {
            let c = case::<6>(11, 5); let r = run_ref::<6>(&c, causal);
            close(&run_fused::<6>(&c, Config::SCALAR, causal), &r, TOL, "D=6 scalar");
            let c = case::<1>(9, 6); let r = run_ref::<1>(&c, causal);
            close(&run_fused::<1>(&c, Config { isa: Isa::Scalar, threads: 3 }, causal), &r, TOL, "D=1 scalar");
        }
    }

    /// Central finite differences of L = Σ dout ⊙ O against the analytic gradients, through the independent reference forward.
    fn fd_check<const D: usize>(n: usize, causal: bool, cfg: Config) {
        let c = case::<D>(n, 77 + n as u64);
        let f = run_fused::<D>(&c, cfg, causal);
        let loss = |q: &[f64], k: &[f64], v: &[f64]| -> f64 {
            let mut o = vec![0.0; n * D]; let mut lse = vec![0.0; n]; let mut s = vec![0.0; n * n];
            forward_reference::<D>(q, k, v, &mut o, &mut lse, &mut s, n, causal).unwrap();
            o.iter().zip(&c.dout).map(|(a, b)| a * b).sum()
        };
        let h = 1e-6;
        for (which, grad) in [(0usize, &f.dq), (1, &f.dk), (2, &f.dv)] {
            for idx in 0..n * D {
                let mut args = [c.q.clone(), c.k.clone(), c.v.clone()];
                args[which][idx] += h; let lp = loss(&args[0], &args[1], &args[2]);
                args[which][idx] -= 2.0 * h; let lm = loss(&args[0], &args[1], &args[2]);
                let fd = (lp - lm) / (2.0 * h);
                let err = (fd - grad[idx]).abs() / grad[idx].abs().max(1.0);
                assert!(err < 1e-7, "FD mismatch tensor {which} idx {idx} n={n} causal={causal} {}: fd={fd} analytic={}", cfg.isa.name(), grad[idx]);
            }
        }
    }
    #[test] fn finite_difference_gradients() {
        for isa in isas() { for &n in &[1usize, 2, 5, 9] { for causal in [false, true] {
            fd_check::<4>(n, causal, Config { isa, threads: 1 });
            fd_check::<8>(n, causal, Config { isa, threads: 2 });
        } } }
    }

    #[test] fn n_equals_one_is_identity_on_v() {
        let c = case::<8>(1, 3);
        for causal in [false, true] { for isa in isas() {
            let f = run_fused::<8>(&c, Config { isa, threads: 1 }, causal);
            assert!(maxd(&f.o, &c.v) < 1e-15);                       // softmax over one key is 1
            assert!(f.dq.iter().all(|x| x.abs() < 1e-15));          // dS = P(dP − Δ) = 0 when P = 1
            assert!(f.dk.iter().all(|x| x.abs() < 1e-15));
            assert!(maxd(&f.dv, &c.dout) < 1e-15);                   // dV = Pᵀ dO = dO
        } }
    }
    #[test] fn empty_input_is_ok_and_writes_nothing() {
        for isa in isas() {
            let cfg = Config { isa, threads: 4 };
            forward::<8>(cfg, &[], &[], &[], &mut [], &mut [], 0, true).unwrap();
            backward::<8>(cfg, &[], &[], &[], &[], &[], &[], &mut [], &mut [], &mut [], &mut [], 0, false).unwrap();
        }
    }
    #[test] fn constant_v_gives_constant_o_and_zero_dq_dk() {
        let mut c = case::<8>(13, 21); c.v = vec![0.25; 13 * 8];
        for causal in [false, true] {
            let f = run_fused::<8>(&c, Config::auto(), causal);
            assert!(f.o.iter().all(|x| (x - 0.25).abs() < 1e-14));
            assert!(f.dq.iter().all(|x| x.abs() < 1e-13) && f.dk.iter().all(|x| x.abs() < 1e-13));
        }
    }
    #[test] fn causal_row0_sees_only_key0_and_future_values_do_not_leak() {
        let c = case::<8>(10, 4);
        let a = run_fused::<8>(&c, Config::auto(), true);
        assert!(maxd(&a.o[..8], &c.v[..8]) < 1e-15);
        let mut c2 = case::<8>(10, 4); for x in &mut c2.v[9 * 8..] { *x += 5.0; } for x in &mut c2.k[9 * 8..] { *x -= 3.0; }
        let b = run_fused::<8>(&c2, Config::auto(), true);
        assert!(bits_eq(&a.o[..9 * 8], &b.o[..9 * 8]), "last key/value leaked into earlier causal rows");
        // causal dK/dV of the last key only receives from the last query
        assert!(a.dk[9 * 8..].iter().all(|x| x.is_finite()));
    }
    #[test] fn large_logits_are_stable_and_match_reference() {
        let mut c = case::<16>(20, 9);
        for x in &mut c.q { *x *= 300.0; }
        for causal in [false, true] {
            let r = run_ref::<16>(&c, causal);
            for isa in isas() {
                let f = run_fused::<16>(&c, Config { isa, threads: 3 }, causal);
                assert!(f.o.iter().chain(&f.dq).chain(&f.dk).chain(&f.dv).all(|x| x.is_finite()));
                close(&f, &r, 1e-8, "large logits");           // softmax saturates; the near-ties amplify rounding
            }
        }
    }
    #[test] fn scaling_v_scales_o_and_leaves_lse_unchanged() {
        // P depends only on Q and K, so O = P·V is linear in V and L is independent of it.
        let c = case::<8>(12, 8); let mut c2 = case::<8>(12, 8); for x in &mut c2.v { *x *= 3.0; }
        let (a, b) = (run_fused::<8>(&c, Config::auto(), false), run_fused::<8>(&c2, Config::auto(), false));
        assert!(maxd(&a.lse, &b.lse) < 1e-15);
        let o3: Vec<f64> = a.o.iter().map(|x| x * 3.0).collect();
        assert!(maxd(&o3, &b.o) < 1e-13);
    }
    #[test] fn threading_is_bit_identical_and_run_to_run_deterministic() {
        for isa in isas() { for causal in [false, true] {
            let c = case::<16>(61, 31);
            let base = run_fused::<16>(&c, Config { isa, threads: 1 }, causal);
            for threads in [2usize, 3, 4, 7, 16, 61, 100, 256] {
                for _ in 0..2 {
                    let t = run_fused::<16>(&c, Config { isa, threads }, causal);
                    for (x, y, nm) in [(&base.o, &t.o, "o"), (&base.lse, &t.lse, "lse"), (&base.dq, &t.dq, "dq"), (&base.dk, &t.dk, "dk"), (&base.dv, &t.dv, "dv")] {
                        assert!(bits_eq(x, y), "{nm} not bit-identical: isa={} threads={threads} causal={causal}", isa.name());
                    }
                }
            }
        } }
    }
    #[test] fn partitions_are_exact_covers() {
        for n in [1usize, 2, 5, 61, 512] { for t in [1usize, 2, 3, 4, 7, 64, 256] { for sk in [Skew::Uniform, Skew::Rising, Skew::Falling] {
            let (b, parts) = partition(n, t, sk);
            assert!(parts >= 1 && parts <= t.max(1) && parts <= n);
            assert_eq!(b[0], 0); assert_eq!(b[parts], n);
            for i in 0..parts { assert!(b[i] <= b[i + 1]); }
        } } }
    }
    #[test] fn heads_equal_independent_single_heads() {
        const D: usize = 8; let (h, n) = (3usize, 11usize);
        let mut r = Rng(5); let q = r.fill(h * n * D); let k = r.fill(h * n * D); let v = r.fill(h * n * D); let dout = r.fill(h * n * D);
        for isa in isas() {
            let cfg = Config { isa, threads: 2 };
            let (mut o, mut lse) = (vec![0.0; h * n * D], vec![0.0; h * n]);
            let (mut dq, mut dk, mut dv, mut delta) = (vec![0.0; h * n * D], vec![0.0; h * n * D], vec![0.0; h * n * D], vec![0.0; h * n]);
            forward_heads::<D>(cfg, h, &q, &k, &v, &mut o, &mut lse, n, true).unwrap();
            backward_heads::<D>(cfg, h, &q, &k, &v, &o, &lse, &dout, &mut dq, &mut dk, &mut dv, &mut delta, n, true).unwrap();
            for i in 0..h {
                let w = n * D; let rg = i * w..(i + 1) * w;
                let c = Case::<D> { n, q: q[rg.clone()].to_vec(), k: k[rg.clone()].to_vec(), v: v[rg.clone()].to_vec(), dout: dout[rg.clone()].to_vec() };
                let s = run_fused::<D>(&c, cfg, true);
                assert!(bits_eq(&s.o, &o[rg.clone()]) && bits_eq(&s.dq, &dq[rg.clone()]) && bits_eq(&s.dk, &dk[rg.clone()]) && bits_eq(&s.dv, &dv[rg.clone()]));
            }
        }
    }
    #[test] fn interleaved_forward_matches_head_major() {
        const D: usize = 8; const H: usize = 4; let n = 9; let w = H * D;
        let mut r = Rng(15); let q = r.fill(n * w); let k = r.fill(n * w); let v = r.fill(n * w);
        let mut o = vec![0.0; n * w];
        let (mut qh, mut kh, mut vh, mut oh, mut lh) = (vec![0.0; n * D], vec![0.0; n * D], vec![0.0; n * D], vec![0.0; n * D], vec![0.0; n]);
        forward_interleaved::<D, H>(Config::SCALAR, &q, &k, &v, &mut o, n, true, &mut qh, &mut kh, &mut vh, &mut oh, &mut lh).unwrap();
        for h in 0..H {
            let g = |x: &[f64]| -> Vec<f64> { (0..n).flat_map(|i| x[i * w + h * D..i * w + (h + 1) * D].to_vec()).collect() };
            let c = Case::<D> { n, q: g(&q), k: g(&k), v: g(&v), dout: vec![0.0; n * D] };
            let r = run_ref::<D>(&c, true);
            assert!(maxd(&r.o, &g(&o)) < TOL);
        }
    }

    #[test] fn invalid_configurations_fail_explicitly() {
        let c = case::<8>(4, 1); let mut o = vec![0.0; 32]; let mut lse = vec![0.0; 4];
        assert_eq!(forward::<8>(Config { isa: Isa::Scalar, threads: 0 }, &c.q, &c.k, &c.v, &mut o, &mut lse, 4, false), Err(AttnError::ZeroThreads));
        assert_eq!(forward::<8>(Config { isa: Isa::Scalar, threads: MAX_THREADS + 1 }, &c.q, &c.k, &c.v, &mut o, &mut lse, 4, false), Err(AttnError::TooManyThreads(MAX_THREADS + 1)));
        assert!(matches!(forward::<8>(Config::SCALAR, &c.q[1..], &c.k, &c.v, &mut o, &mut lse, 4, false), Err(AttnError::Shape("q"))));
        assert!(matches!(forward::<8>(Config::SCALAR, &c.q, &c.k, &c.v, &mut o, &mut lse[1..], 4, false), Err(AttnError::Shape("lse"))));
        assert!(matches!(forward::<0>(Config::SCALAR, &[], &[], &[], &mut [], &mut [], 0, false), Err(AttnError::Shape(_))));
        let mut bad = c.q.clone(); bad[3] = f64::NAN;
        assert_eq!(forward::<8>(Config::SCALAR, &bad, &c.k, &c.v, &mut o, &mut lse, 4, false), Err(AttnError::NonFinite("q")));
        bad[3] = f64::INFINITY;
        assert_eq!(forward::<8>(Config::SCALAR, &c.q, &bad, &c.v, &mut o, &mut lse, 4, false), Err(AttnError::NonFinite("k")));
        let c6 = case::<6>(4, 1); let (mut o6, mut l6) = (vec![0.0; 24], vec![0.0; 4]);
        assert_eq!(forward::<6>(Config { isa: Isa::Avx2Fma, threads: 1 }, &c6.q, &c6.k, &c6.v, &mut o6, &mut l6, 4, false), Err(AttnError::DimNotMultipleOf4(6)));
        let mut s = vec![0.0; 15];
        assert!(matches!(forward_reference::<8>(&c.q, &c.k, &c.v, &mut o, &mut lse, &mut s, 4, false), Err(AttnError::Shape(_))));
    }

    // Cross-language golden vectors: bench/shared/attention_golden.txt, written by `attention --write-golden`.
    fn parse_golden(txt: &str) -> Vec<(usize, usize, bool, u64, std::collections::HashMap<String, Vec<f64>>)> {
        let mut out = vec![]; let mut cur: Option<(usize, usize, bool, u64, std::collections::HashMap<String, Vec<f64>>)> = None;
        for l in txt.lines() {
            let mut it = l.split(' ');
            match it.next().unwrap() {
                "case" => { if let Some(c) = cur.take() { out.push(c); }
                            let p: Vec<&str> = it.collect();
                            cur = Some((p[0].parse().unwrap(), p[1].parse().unwrap(), p[2] == "1", p[3].parse().unwrap(), Default::default())); }
                name => { cur.as_mut().unwrap().4.insert(name.to_string(), it.map(|x| x.parse().unwrap()).collect()); }
            }
        }
        if let Some(c) = cur.take() { out.push(c); } out
    }
    #[test] fn golden_vectors_match_every_implementation() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/shared/attention_golden.txt");
        let g = parse_golden(&std::fs::read_to_string(path).expect("run `cargo run --release --bin attention -- --write-golden` first"));
        assert_eq!(g.len(), GOLDEN_CASES.len());
        for (n, d, causal, seed, m) in g {
            assert!(GOLDEN_CASES.contains(&(n, d, causal, seed)));
            let (q, k, v, dout) = golden_inputs(n, d, seed);
            // golden values were produced by the reference path; check inputs reproduce and each fused impl matches
            for isa in isas() {
                macro_rules! chk { ($D:literal) => {{
                    let c = Case::<$D> { n, q: q.clone(), k: k.clone(), v: v.clone(), dout: dout.clone() };
                    let f = run_fused::<$D>(&c, Config { isa, threads: 2 }, causal);
                    for (nm, x) in [("o", &f.o), ("lse", &f.lse), ("dq", &f.dq), ("dk", &f.dk), ("dv", &f.dv)] {
                        let dd = maxd(x, &m[nm]); assert!(dd < TOL, "golden {nm} n={n} d={d} causal={causal} {}: {dd:e}", isa.name());
                    }
                }}; }
                match d { 4 => chk!(4), 8 => chk!(8), 16 => chk!(16), 64 => chk!(64), _ => panic!("golden case with unsupported D {d}") }
            }
        }
    }
}
