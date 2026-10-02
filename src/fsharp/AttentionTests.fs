// Self-tests and kernel benchmark for Attention.fs.
//   dotnet run -c Release --project src/fsharp -- selftest          kernel tests (exit code 1 on failure)
//   dotnet run -c Release --project src/fsharp -- bench-attention    forward/backward timings, scalar vs AVX2+FMA
module AttentionTests

open System
open System.Diagnostics
open System.IO
open Attention

let tol = 1e-11

let maxd (a: float[]) (b: float[]) =
    let mutable m = 0.0
    for i = 0 to a.Length - 1 do m <- max m (abs (a.[i] - b.[i]))
    m

let expectClose (a: float[]) (b: float[]) (t: float) (what: string) =
    let d = maxd a b
    if not (d < t) then failwithf "FAIL %s: max diff %e" what d

let raisesAttention (f: unit -> unit) =
    try f (); false with AttentionError _ -> true

type Out = { o: float[]; lse: float[]; dq: float[]; dk: float[]; dv: float[] }

let runFused isa d (q: float[]) k v (dout: float[]) n causal =
    let o = Array.zeroCreate (n * d)
    let lse = Array.zeroCreate n
    let dq = Array.zeroCreate (n * d)
    let dk = Array.zeroCreate (n * d)
    let dv = Array.zeroCreate (n * d)
    forward isa d q k v o lse n causal
    backward isa d q k v o lse dout dq dk dv (Array.zeroCreate n) n causal
    { o = o; lse = lse; dq = dq; dk = dk; dv = dv }

let runReference d (q: float[]) k v (dout: float[]) n causal =
    let o = Array.zeroCreate (n * d)
    let lse = Array.zeroCreate n
    let s = Array.zeroCreate (n * n)
    let dq = Array.zeroCreate (n * d)
    let dk = Array.zeroCreate (n * d)
    let dv = Array.zeroCreate (n * d)
    forwardReference d q k v o lse s n causal
    backwardReference d q k v dout dq dk dv s (Array.zeroCreate (n * n)) n causal
    { o = o; lse = lse; dq = dq; dk = dk; dv = dv }

let inputs n d (seed: uint64) =
    let r = Rng seed
    let q = r.Fill(n * d)
    let k = r.Fill(n * d)
    let v = r.Fill(n * d)
    let dout = r.Fill(n * d)
    q, k, v, dout

let compare (a: Out) (b: Out) (t: float) (tag: string) =
    expectClose a.o b.o t ("o " + tag)
    expectClose a.lse b.lse t ("lse " + tag)
    expectClose a.dq b.dq t ("dq " + tag)
    expectClose a.dk b.dk t ("dk " + tag)
    expectClose a.dv b.dv t ("dv " + tag)

let isas d = if d % 4 = 0 && avx2FmaAvailable then [ Scalar; Avx2Fma ] else [ Scalar ]

let sweep () =
    for d in [ 1; 4; 6; 8; 16; 64 ] do
        for n in [ 1; 2; 3; 7; 16; 17; 33; 64 ] do
            for causal in [ false; true ] do
                let q, k, v, dout = inputs n d (uint64 (1000 + n))
                let r = runReference d q k v dout n causal
                for isa in isas d do
                    compare (runFused isa d q k v dout n causal) r tol (sprintf "d=%d n=%d causal=%b %s" d n causal (isaName isa))
    printfn "fused == reference: d in {1, 4, 6, 8, 16, 64}, n in {1..64}, causal and bidirectional, %s" (if avx2FmaAvailable then "scalar and AVX2+FMA" else "scalar only (no AVX2+FMA on this CPU)")

let finiteDifferences () =
    let h = 1e-6
    for isa in isas 4 do
        for d, n, causal in [ 4, 1, false; 4, 2, true; 4, 5, false; 4, 5, true; 8, 3, true ] do
            let q, k, v, dout = inputs n d (uint64 (77 + n))
            let f = runFused isa d q k v dout n causal
            let loss (q: float[]) (k: float[]) (v: float[]) = (runReference d q k v dout n causal).o |> Array.mapi (fun i x -> x * dout.[i]) |> Array.sum
            for which in 0 .. 2 do
                let grad = [| f.dq; f.dk; f.dv |].[which]
                for idx in 0 .. n * d - 1 do
                    let args sign =
                        let a = [| Array.copy q; Array.copy k; Array.copy v |]
                        a.[which].[idx] <- a.[which].[idx] + sign * h
                        a
                    let p, m = args 1.0, args -1.0
                    let fd = (loss p.[0] p.[1] p.[2] - loss m.[0] m.[1] m.[2]) / (2.0 * h)
                    if not (abs (fd - grad.[idx]) / max 1.0 (abs grad.[idx]) < 1e-7) then
                        failwithf "FAIL finite difference tensor %d idx %d n=%d causal=%b %s: fd=%g analytic=%g" which idx n causal (isaName isa) fd grad.[idx]
    printfn "finite-difference gradients (dq, dk, dv) match"

let edgeCases () =
    for isa in isas 8 do
        forward isa 8 [||] [||] [||] [||] [||] 0 true
        backward isa 8 [||] [||] [||] [||] [||] [||] [||] [||] [||] [||] 0 false
        // n = 1: softmax over one key is 1, so O = V, dQ = dK = 0, dV = dO
        let q, k, v, dout = inputs 1 8 3UL
        let f = runFused isa 8 q k v dout 1 true
        expectClose f.o v 1e-15 "n=1 o == v"
        expectClose f.dq (Array.zeroCreate 8) 1e-15 "n=1 dq == 0"
        expectClose f.dk (Array.zeroCreate 8) 1e-15 "n=1 dk == 0"
        expectClose f.dv dout 1e-15 "n=1 dv == dout"
        // constant V gives constant O
        let q2, k2, _, d2 = inputs 13 8 21UL
        let vc = Array.create (13 * 8) 0.25
        expectClose (runFused isa 8 q2 k2 vc d2 13 false).o vc 1e-14 "constant V"
        // large logits stay finite and match the reference
        let q3, k3, v3, d3 = inputs 20 16 9UL
        let qb = q3 |> Array.map (fun x -> x * 300.0)
        let fb = runFused isa 16 qb k3 v3 d3 20 false
        if not (Array.forall Double.IsFinite (Array.concat [ fb.o; fb.dq; fb.dk; fb.dv ])) then failwith "FAIL large logits: non-finite output"
        expectClose fb.o (runReference 16 qb k3 v3 d3 20 false).o 1e-8 "large logits"
        // causal: changing the last key/value must not change earlier output rows
        let q4, k4, v4, d4 = inputs 10 8 4UL
        let a = runFused isa 8 q4 k4 v4 d4 10 true
        let k5, v5 = Array.copy k4, Array.copy v4
        for c in 0 .. 7 do
            k5.[9 * 8 + c] <- k5.[9 * 8 + c] - 3.0
            v5.[9 * 8 + c] <- v5.[9 * 8 + c] + 5.0
        let b = runFused isa 8 q4 k5 v5 d4 10 true
        if Array.sub a.o 0 72 <> Array.sub b.o 0 72 then failwith "FAIL causal: last key/value leaked into earlier rows"
    printfn "edge cases: n=0, n=1, constant V, large logits, causal no-leak"

let errorPaths () =
    let q, k, v, _ = inputs 4 8 1UL
    let o, lse = Array.zeroCreate 32, Array.zeroCreate 4
    let expectErr label f = if not (raisesAttention f) then failwithf "FAIL expected an AttentionError: %s" label
    expectErr "shape mismatch" (fun () -> forward Scalar 8 q.[1..] k v o lse 4 false)
    expectErr "lse length" (fun () -> forward Scalar 8 q k v o lse.[1..] 4 false)
    let bad = Array.copy q
    bad.[3] <- nan
    expectErr "NaN input" (fun () -> forward Scalar 8 bad k v o lse 4 false)
    bad.[3] <- infinity
    expectErr "infinite input" (fun () -> forward Scalar 8 q bad v o lse 4 false)
    expectErr "d = 0" (fun () -> forward Scalar 0 [||] [||] [||] [||] [||] 0 false)
    expectErr "negative n" (fun () -> forward Scalar 8 [||] [||] [||] [||] [||] -1 false)
    expectErr "AVX2 with d % 4 <> 0" (fun () -> forward Avx2Fma 6 (Array.zeroCreate 24) (Array.zeroCreate 24) (Array.zeroCreate 24) (Array.zeroCreate 24) lse 4 false)
    expectErr "reference scratch" (fun () -> forwardReference 8 q k v o lse (Array.zeroCreate 15) 4 false)
    printfn "error paths: shape, non-finite, d = 0, n < 0, AVX2 with d %% 4 <> 0, scratch size"

let rec findGolden (dir: string) =
    let p = Path.Combine(dir, "bench", "shared", "attention_golden.txt")
    if File.Exists p then p
    else
        let parent = Directory.GetParent dir
        if isNull parent then failwith "attention_golden.txt not found above the working/base directory" else findGolden parent.FullName

let golden (path: string) =
    let lines = File.ReadAllLines path |> Array.filter (fun l -> l <> "")
    let row (l: string) = l.Split(' ').[1..] |> Array.map float
    let mutable cases = 0
    for c in 0 .. lines.Length / 6 - 1 do
        let head = lines.[c * 6].Split(' ')
        let n, d, causal, seed = int head.[1], int head.[2], head.[3] = "1", uint64 head.[4]
        let eo, el, edq, edk, edv = row lines.[c * 6 + 1], row lines.[c * 6 + 2], row lines.[c * 6 + 3], row lines.[c * 6 + 4], row lines.[c * 6 + 5]
        let q, k, v, dout = inputs n d seed
        for isa in isas d do
            let f = runFused isa d q k v dout n causal
            let tag = sprintf "golden n=%d d=%d causal=%b %s" n d causal (isaName isa)
            expectClose f.o eo tol ("o " + tag)
            expectClose f.lse el tol ("lse " + tag)
            expectClose f.dq edq tol ("dq " + tag)
            expectClose f.dk edk tol ("dk " + tag)
            expectClose f.dv edv tol ("dv " + tag)
        cases <- cases + 1
    if cases <> 6 then failwithf "expected 6 golden cases, found %d" cases
    printfn "golden vectors: %d cases match the Rust reference within %g" cases tol

let runAll () =
    edgeCases ()
    errorPaths ()
    sweep ()
    finiteDifferences ()
    golden (findGolden (Directory.GetCurrentDirectory()) |> fun p -> p)
    printfn "all attention kernel tests passed"

// ───────────── benchmark ─────────────

let private bestSeconds (f: unit -> unit) =
    f ()
    let sw = Stopwatch.StartNew()
    f ()
    let one = max sw.Elapsed.TotalSeconds 1e-9
    let iters = max 1 (int (Math.Ceiling(0.08 / one)))
    [ 1 .. 5 ]
    |> List.map (fun _ ->
        let t = Stopwatch.StartNew()
        for _ in 1 .. iters do f ()
        t.Elapsed.TotalSeconds / float iters)
    |> List.min

let bench () =
    printfn "attention bench — avx2+fma: %b, d = 64, causal" avx2FmaAvailable
    printfn "%-6s %-28s %10s %8s | %10s %8s" "n" "implementation" "fwd ms" "GF/s" "bwd ms" "GF/s"
    let d = 64
    for n in [ 128; 512; 2048 ] do
        let q, k, v, dout = inputs n d 42UL
        let o, lse, delta = Array.zeroCreate (n * d), Array.zeroCreate n, Array.zeroCreate n
        let dq, dk, dv = Array.zeroCreate (n * d), Array.zeroCreate (n * d), Array.zeroCreate (n * d)
        let pairs = float (n * (n + 1) / 2)
        let ff, bf = pairs * 4.0 * float d, pairs * 14.0 * float d + 2.0 * float (n * d)
        let row label tf tb = printfn "%-6d %-28s %10.3f %8.2f | %10.3f %8.2f" n label (tf * 1e3) (ff / tf / 1e9) (tb * 1e3) (bf / tb / 1e9)
        let s, dp = Array.zeroCreate (n * n), Array.zeroCreate (n * n)
        row "reference (materialised)"
            (bestSeconds (fun () -> forwardReference d q k v o lse s n true))
            (bestSeconds (fun () -> backwardReference d q k v dout dq dk dv s dp n true))
        for isa in isas d do
            forward isa d q k v o lse n true
            row (sprintf "fused %s" (isaName isa))
                (bestSeconds (fun () -> forward isa d q k v o lse n true))
                (bestSeconds (fun () -> backward isa d q k v o lse dout dq dk dv delta n true))
