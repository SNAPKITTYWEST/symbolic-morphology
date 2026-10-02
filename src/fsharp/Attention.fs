// Scaled dot-product attention, forward + backward — F# port of src/rust/src/attention.rs.
//
// Layout: row-major float[], x.[i*d + j], n rows by d columns, single head, scale = 1/sqrt(d).
// Forward   S = scale * Q K^T (causal: j > i masked), P = softmax_row(S), O = P V, L_i = logsumexp_j S[i][j]
// Backward  Delta_i = dO_i . O_i
//           pass 1 (query rows):  dQ_i = scale * sum_j dS_ij k_j
//           pass 2 (key rows):    dV_j = sum_i P_ij dO_i,  dK_j = scale * sum_i dS_ij q_i
//           with P_ij = exp(scale q_i.k_j - L_i) and dS_ij = P_ij (dO_i . v_j - Delta_i)
// Isa.Scalar: plain loops.  Isa.Avx2Fma: Vector256<float> with fused multiply-add (needs d % 4 = 0 and a CPU with AVX2+FMA).
// Failures raise AttentionError, never silently fall back: shape mismatch (lengths must be exactly n*d / n), non-finite
// input, d < 1, Avx2Fma with d % 4 <> 0 or on a CPU without AVX2+FMA.
module Attention

open System
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86

[<Literal>]
let BK = 16

exception AttentionError of string

type Isa =
    | Scalar
    | Avx2Fma

let avx2FmaAvailable = Avx2.IsSupported && Fma.IsSupported
let detect () = if avx2FmaAvailable then Avx2Fma else Scalar
let isaName isa = match isa with Scalar -> "scalar" | Avx2Fma -> "avx2+fma"

/// xorshift64 -> uniform in [-1, 1). Bit-identical to attention::Rng in Rust, Rng in attention.py and attention.mojo.
type Rng(seed: uint64) =
    let mutable s = seed
    member _.Next() : float =
        s <- s ^^^ (s <<< 13)
        s <- s ^^^ (s >>> 7)
        s <- s ^^^ (s <<< 17)
        float (s >>> 11) / float (1UL <<< 53) * 2.0 - 1.0
    member r.Fill(count: int) : float[] = Array.init count (fun _ -> r.Next())

// ───────────── element-wise primitives (offset-based, d-length) ─────────────

let inline private dot (simd: bool) (d: int) (a: float[]) (ao: int) (b: float[]) (bo: int) : float =
    if simd then
        let mutable acc = Vector256<float>.Zero
        let mutable j = 0
        while j < d do
            acc <- Fma.MultiplyAdd(Vector256.LoadUnsafe(&a.[ao + j]), Vector256.LoadUnsafe(&b.[bo + j]), acc)
            j <- j + 4
        Vector256.Sum acc
    else
        let mutable s = 0.0
        for j = 0 to d - 1 do
            s <- s + a.[ao + j] * b.[bo + j]
        s

let inline private axpy (simd: bool) (d: int) (y: float[]) (yo: int) (a: float) (x: float[]) (xo: int) =
    if simd then
        let va = Vector256.Create a
        let mutable j = 0
        while j < d do
            Vector256.StoreUnsafe(Fma.MultiplyAdd(va, Vector256.LoadUnsafe(&x.[xo + j]), Vector256.LoadUnsafe(&y.[yo + j])), &y.[yo + j])
            j <- j + 4
    else
        for j = 0 to d - 1 do
            y.[yo + j] <- y.[yo + j] + a * x.[xo + j]

let inline private scaleBy (simd: bool) (d: int) (y: float[]) (a: float) =
    if simd then
        let va = Vector256.Create a
        let mutable j = 0
        while j < d do
            Vector256.StoreUnsafe(va * Vector256.LoadUnsafe(&y.[j]), &y.[j])
            j <- j + 4
    else
        for j = 0 to d - 1 do
            y.[j] <- y.[j] * a

// ───────────── fused kernels ─────────────

let private fwdRows (simd: bool) (d: int) (q: float[]) (k: float[]) (v: float[]) (o: float[]) (lse: float[]) (n: int) (causal: bool) =
    let scale = 1.0 / Math.Sqrt(float d)
    let acc = Array.zeroCreate<float> d
    let s = Array.zeroCreate<float> BK
    for i = 0 to n - 1 do
        let qo = i * d
        let kmax = if causal then i + 1 else n
        let mutable m = Double.NegativeInfinity
        let mutable l = 0.0
        Array.Clear acc
        let mutable j0 = 0
        while j0 < kmax do
            let bl = min BK (kmax - j0)
            let mutable bm = m
            for b = 0 to bl - 1 do
                let x = dot simd d q qo k ((j0 + b) * d) * scale
                s.[b] <- x
                if x > bm then bm <- x
            let corr = if m = Double.NegativeInfinity then 0.0 else Math.Exp(m - bm)
            l <- l * corr
            scaleBy simd d acc corr
            for b = 0 to bl - 1 do
                let p = Math.Exp(s.[b] - bm)
                l <- l + p
                axpy simd d acc 0 p v ((j0 + b) * d)
            m <- bm
            j0 <- j0 + bl
        scaleBy simd d acc (1.0 / l)
        Array.blit acc 0 o qo d
        lse.[i] <- m + Math.Log l

let private deltaRows (simd: bool) (d: int) (o: float[]) (dO: float[]) (delta: float[]) (n: int) =
    for i = 0 to n - 1 do
        delta.[i] <- dot simd d dO (i * d) o (i * d)

let private dqRows (simd: bool) (d: int) (q: float[]) (k: float[]) (v: float[]) (dO: float[]) (lse: float[]) (delta: float[]) (dq: float[]) (n: int) (causal: bool) =
    let scale = 1.0 / Math.Sqrt(float d)
    let acc = Array.zeroCreate<float> d
    for i = 0 to n - 1 do
        let kmax = if causal then i + 1 else n
        Array.Clear acc
        for j = 0 to kmax - 1 do
            let p = Math.Exp(dot simd d q (i * d) k (j * d) * scale - lse.[i])
            let ds = p * (dot simd d dO (i * d) v (j * d) - delta.[i])
            axpy simd d acc 0 ds k (j * d)
        scaleBy simd d acc scale
        Array.blit acc 0 dq (i * d) d

let private dkvRows (simd: bool) (d: int) (q: float[]) (k: float[]) (v: float[]) (dO: float[]) (lse: float[]) (delta: float[]) (dk: float[]) (dv: float[]) (n: int) (causal: bool) =
    let scale = 1.0 / Math.Sqrt(float d)
    let acck = Array.zeroCreate<float> d
    let accv = Array.zeroCreate<float> d
    for j = 0 to n - 1 do
        Array.Clear acck
        Array.Clear accv
        for i = (if causal then j else 0) to n - 1 do
            let p = Math.Exp(dot simd d q (i * d) k (j * d) * scale - lse.[i])
            axpy simd d accv 0 p dO (i * d)
            let ds = p * (dot simd d dO (i * d) v (j * d) - delta.[i])
            axpy simd d acck 0 ds q (i * d)
        scaleBy simd d acck scale
        Array.blit acck 0 dk (j * d) d
        Array.blit accv 0 dv (j * d) d

// ───────────── validation ─────────────

let private fail msg = raise (AttentionError msg)
let private checkLen (x: float[]) (want: int) (what: string) =
    if x.Length <> want then fail (sprintf "attention: bad shape: %s has %d elements, expected %d" what x.Length want)
let private checkFinite (x: float[]) (what: string) =
    if not (Array.forall Double.IsFinite x) then fail (sprintf "attention: non-finite value in %s" what)
let private checkCfg (isa: Isa) (d: int) (n: int) =
    if d < 1 then fail "attention: bad shape: d must be > 0"
    if n < 0 then fail "attention: bad shape: n must be >= 0"
    if isa = Avx2Fma then
        if d % 4 <> 0 then fail (sprintf "attention: AVX2 path needs d %% 4 = 0, got d = %d" d)
        if not avx2FmaAvailable then fail "attention: unsupported instruction set: avx2+fma not available on this CPU"

// ───────────── public API ─────────────

/// Fused forward. Writes o (n*d) and lse (n); both are fully overwritten.
let forward (isa: Isa) (d: int) (q: float[]) (k: float[]) (v: float[]) (o: float[]) (lse: float[]) (n: int) (causal: bool) =
    checkCfg isa d n
    checkLen q (n * d) "q"; checkLen k (n * d) "k"; checkLen v (n * d) "v"; checkLen o (n * d) "o"; checkLen lse n "lse"
    checkFinite q "q"; checkFinite k "k"; checkFinite v "v"
    if n > 0 then fwdRows (isa = Avx2Fma) d q k v o lse n causal

/// Fused two-pass backward. dq, dk, dv (n*d) and delta (n) are fully overwritten, not accumulated.
let backward (isa: Isa) (d: int) (q: float[]) (k: float[]) (v: float[]) (o: float[]) (lse: float[]) (dO: float[])
             (dq: float[]) (dk: float[]) (dv: float[]) (delta: float[]) (n: int) (causal: bool) =
    checkCfg isa d n
    for (x, w) in [ q, "q"; k, "k"; v, "v"; o, "o"; dO, "do"; dq, "dq"; dk, "dk"; dv, "dv" ] do checkLen x (n * d) w
    checkLen lse n "lse"; checkLen delta n "delta"
    for (x, w) in [ q, "q"; k, "k"; v, "v"; o, "o"; dO, "do"; lse, "lse" ] do checkFinite x w
    if n > 0 then
        let simd = (isa = Avx2Fma)
        deltaRows simd d o dO delta n
        dqRows simd d q k v dO lse delta dq n causal
        dkvRows simd d q k v dO lse delta dk dv n causal

/// Textbook forward: materialises S (n*n scratch `s`) and normalises it explicitly.
let forwardReference (d: int) (q: float[]) (k: float[]) (v: float[]) (o: float[]) (lse: float[]) (s: float[]) (n: int) (causal: bool) =
    checkCfg Scalar d n
    checkLen q (n * d) "q"; checkLen k (n * d) "k"; checkLen v (n * d) "v"; checkLen o (n * d) "o"; checkLen lse n "lse"
    checkLen s (n * n) "scratch s (n*n)"
    checkFinite q "q"; checkFinite k "k"; checkFinite v "v"
    let scale = 1.0 / Math.Sqrt(float d)
    for i = 0 to n - 1 do
        let kmax = if causal then i + 1 else n
        let mutable m = Double.NegativeInfinity
        for j = 0 to kmax - 1 do
            let mutable dt = 0.0
            for c = 0 to d - 1 do dt <- dt + q.[i * d + c] * k.[j * d + c]
            let x = dt * scale
            s.[i * n + j] <- x
            if x > m then m <- x
        let mutable l = 0.0
        for j = 0 to kmax - 1 do
            let e = Math.Exp(s.[i * n + j] - m)
            s.[i * n + j] <- e
            l <- l + e
        for c = 0 to d - 1 do
            let mutable a = 0.0
            for j = 0 to kmax - 1 do a <- a + s.[i * n + j] * v.[j * d + c]
            o.[i * d + c] <- a / l
        lse.[i] <- m + Math.Log l

/// Textbook backward from the formulas (Delta = sum_j P dP, independent of dO.O). p and dp are n*n scratch.
let backwardReference (d: int) (q: float[]) (k: float[]) (v: float[]) (dO: float[])
                      (dq: float[]) (dk: float[]) (dv: float[]) (p: float[]) (dp: float[]) (n: int) (causal: bool) =
    checkCfg Scalar d n
    for (x, w) in [ q, "q"; k, "k"; v, "v"; dO, "do"; dq, "dq"; dk, "dk"; dv, "dv" ] do checkLen x (n * d) w
    checkLen p (n * n) "scratch p (n*n)"; checkLen dp (n * n) "scratch dp (n*n)"
    for (x, w) in [ q, "q"; k, "k"; v, "v"; dO, "do" ] do checkFinite x w
    let scale = 1.0 / Math.Sqrt(float d)
    for i = 0 to n - 1 do
        let kmax = if causal then i + 1 else n
        let mutable m = Double.NegativeInfinity
        for j = 0 to n - 1 do
            if j >= kmax then p.[i * n + j] <- 0.0
            else
                let mutable dt = 0.0
                for c = 0 to d - 1 do dt <- dt + q.[i * d + c] * k.[j * d + c]
                let x = dt * scale
                p.[i * n + j] <- x
                if x > m then m <- x
        let mutable l = 0.0
        for j = 0 to kmax - 1 do
            let e = Math.Exp(p.[i * n + j] - m)
            p.[i * n + j] <- e
            l <- l + e
        for j = 0 to kmax - 1 do p.[i * n + j] <- p.[i * n + j] / l
    for j = 0 to n - 1 do
        for c = 0 to d - 1 do
            let mutable a = 0.0
            for i = 0 to n - 1 do a <- a + p.[i * n + j] * dO.[i * d + c]
            dv.[j * d + c] <- a
    for i = 0 to n - 1 do
        for j = 0 to n - 1 do
            let mutable a = 0.0
            for c = 0 to d - 1 do a <- a + dO.[i * d + c] * v.[j * d + c]
            dp.[i * n + j] <- a
    for i = 0 to n - 1 do
        let mutable delta = 0.0
        for j = 0 to n - 1 do delta <- delta + p.[i * n + j] * dp.[i * n + j]
        for j = 0 to n - 1 do dp.[i * n + j] <- p.[i * n + j] * (dp.[i * n + j] - delta)
    for i = 0 to n - 1 do
        for c = 0 to d - 1 do
            let mutable a = 0.0
            for j = 0 to n - 1 do a <- a + dp.[i * n + j] * k.[j * d + c]
            dq.[i * d + c] <- a * scale
    for j = 0 to n - 1 do
        for c = 0 to d - 1 do
            let mutable a = 0.0
            for i = 0 to n - 1 do a <- a + dp.[i * n + j] * q.[i * d + c]
            dk.[j * d + c] <- a * scale
