// Symbolic Morphology Engine — Latin verb morphology from raw letters to Boolean grammar
// Copyright (C) 2026 Ahmad Ali Parr
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

open System
open System.Diagnostics

// ═══════════════════════════════════════════════════════════════
//  SYMBOLIC LEARNING ENGINE — F# IMPLEMENTATION
//  Hand-rolled from mathematical primitives
// ═══════════════════════════════════════════════════════════════

module Features =
    let names = [|
        "PERSON_1";"PERSON_2";"PERSON_3"
        "SINGULAR";"PLURAL"
        "PRESENT";"IMPERFECT";"FUTURE";"PERFECT"
        "INDICATIVE";"SUBJUNCTIVE";"IMPERATIVE"
        "ACTIVE";"PASSIVE"
        "CONJ_1";"CONJ_2";"CONJ_3" |]
    let count = names.Length
    let threshold = 0.5
    let index = names |> Array.mapi (fun i n -> n, i) |> Map.ofArray
    let groups = [|
        "PERSON", [|"PERSON_1";"PERSON_2";"PERSON_3"|]
        "NUMBER", [|"SINGULAR";"PLURAL"|]
        "TENSE",  [|"PRESENT";"IMPERFECT";"FUTURE";"PERFECT"|]
        "MOOD",   [|"INDICATIVE";"SUBJUNCTIVE";"IMPERATIVE"|]
        "VOICE",  [|"ACTIVE";"PASSIVE"|]
        "CONJ",   [|"CONJ_1";"CONJ_2";"CONJ_3"|] |]

    let makeTarget (specs: (string*string)[]) =
        let t = Array.zeroCreate<float> count
        for _, feat in specs do t.[index.[feat]] <- 1.0
        t

    let countCorrect (pred: float[]) (target: float[]) =
        let mutable c = 0
        for i = 0 to count - 1 do
            if (pred.[i] >= threshold) = (target.[i] >= 0.5) then c <- c + 1
        c

module Dataset =
    open Features
    let p1 = "PERSON","PERSON_1"
    let p2 = "PERSON","PERSON_2"
    let p3 = "PERSON","PERSON_3"
    let sg = "NUMBER","SINGULAR"
    let pl = "NUMBER","PLURAL"
    let pres = "TENSE","PRESENT"
    let impf = "TENSE","IMPERFECT"
    let fut = "TENSE","FUTURE"
    let perf = "TENSE","PERFECT"
    let ind = "MOOD","INDICATIVE"
    let act = "VOICE","ACTIVE"
    let c1 = "CONJ","CONJ_1"
    let c2 = "CONJ","CONJ_2"
    let c3 = "CONJ","CONJ_3"
    let t specs = makeTarget specs

    let corpus() = [|
        "AMO",t[|p1;sg;pres;ind;act;c1|];"AMAS",t[|p2;sg;pres;ind;act;c1|]
        "AMAT",t[|p3;sg;pres;ind;act;c1|];"AMAMUS",t[|p1;pl;pres;ind;act;c1|]
        "AMATIS",t[|p2;pl;pres;ind;act;c1|];"AMANT",t[|p3;pl;pres;ind;act;c1|]
        "AMABAM",t[|p1;sg;impf;ind;act;c1|];"AMABAS",t[|p2;sg;impf;ind;act;c1|]
        "AMABAT",t[|p3;sg;impf;ind;act;c1|];"AMABAMUS",t[|p1;pl;impf;ind;act;c1|]
        "AMABATIS",t[|p2;pl;impf;ind;act;c1|];"AMABANT",t[|p3;pl;impf;ind;act;c1|]
        "AMABO",t[|p1;sg;fut;ind;act;c1|];"AMABIS",t[|p2;sg;fut;ind;act;c1|]
        "AMABIT",t[|p3;sg;fut;ind;act;c1|];"AMABIMUS",t[|p1;pl;fut;ind;act;c1|]
        "AMABITIS",t[|p2;pl;fut;ind;act;c1|];"AMABUNT",t[|p3;pl;fut;ind;act;c1|]
        "AMAVI",t[|p1;sg;perf;ind;act;c1|];"AMAVISTI",t[|p2;sg;perf;ind;act;c1|]
        "AMAVIT",t[|p3;sg;perf;ind;act;c1|];"AMAVIMUS",t[|p1;pl;perf;ind;act;c1|]
        "AMAVISTIS",t[|p2;pl;perf;ind;act;c1|];"AMAVERUNT",t[|p3;pl;perf;ind;act;c1|]
        "LAUDO",t[|p1;sg;pres;ind;act;c1|];"LAUDAS",t[|p2;sg;pres;ind;act;c1|]
        "LAUDAT",t[|p3;sg;pres;ind;act;c1|];"LAUDAMUS",t[|p1;pl;pres;ind;act;c1|]
        "LAUDATIS",t[|p2;pl;pres;ind;act;c1|];"LAUDANT",t[|p3;pl;pres;ind;act;c1|]
        "LAUDABAM",t[|p1;sg;impf;ind;act;c1|];"LAUDABAS",t[|p2;sg;impf;ind;act;c1|]
        "LAUDABAT",t[|p3;sg;impf;ind;act;c1|];"LAUDABAMUS",t[|p1;pl;impf;ind;act;c1|]
        "LAUDABATIS",t[|p2;pl;impf;ind;act;c1|];"LAUDABANT",t[|p3;pl;impf;ind;act;c1|]
        "LAUDAVI",t[|p1;sg;perf;ind;act;c1|];"LAUDAVISTI",t[|p2;sg;perf;ind;act;c1|]
        "LAUDAVIT",t[|p3;sg;perf;ind;act;c1|]
        "MONEO",t[|p1;sg;pres;ind;act;c2|];"MONES",t[|p2;sg;pres;ind;act;c2|]
        "MONET",t[|p3;sg;pres;ind;act;c2|];"MONEMUS",t[|p1;pl;pres;ind;act;c2|]
        "MONETIS",t[|p2;pl;pres;ind;act;c2|];"MONENT",t[|p3;pl;pres;ind;act;c2|]
        "MONEBAM",t[|p1;sg;impf;ind;act;c2|];"MONEBAS",t[|p2;sg;impf;ind;act;c2|]
        "MONEBAT",t[|p3;sg;impf;ind;act;c2|];"MONEBAMUS",t[|p1;pl;impf;ind;act;c2|]
        "MONEBATIS",t[|p2;pl;impf;ind;act;c2|];"MONEBANT",t[|p3;pl;impf;ind;act;c2|]
        "MONUI",t[|p1;sg;perf;ind;act;c2|];"MONUISTI",t[|p2;sg;perf;ind;act;c2|]
        "MONUIT",t[|p3;sg;perf;ind;act;c2|]
        "HABEO",t[|p1;sg;pres;ind;act;c2|];"HABES",t[|p2;sg;pres;ind;act;c2|]
        "HABET",t[|p3;sg;pres;ind;act;c2|];"HABEMUS",t[|p1;pl;pres;ind;act;c2|]
        "HABETIS",t[|p2;pl;pres;ind;act;c2|];"HABENT",t[|p3;pl;pres;ind;act;c2|]
        "HABUI",t[|p1;sg;perf;ind;act;c2|];"HABUISTI",t[|p2;sg;perf;ind;act;c2|]
        "HABUIT",t[|p3;sg;perf;ind;act;c2|]
        "REGO",t[|p1;sg;pres;ind;act;c3|];"REGIS",t[|p2;sg;pres;ind;act;c3|]
        "REGIT",t[|p3;sg;pres;ind;act;c3|];"REGIMUS",t[|p1;pl;pres;ind;act;c3|]
        "REGITIS",t[|p2;pl;pres;ind;act;c3|];"REGUNT",t[|p3;pl;pres;ind;act;c3|]
        "REGEBAM",t[|p1;sg;impf;ind;act;c3|];"REGEBAS",t[|p2;sg;impf;ind;act;c3|]
        "REGEBAT",t[|p3;sg;impf;ind;act;c3|];"REGEBAMUS",t[|p1;pl;impf;ind;act;c3|]
        "REGEBATIS",t[|p2;pl;impf;ind;act;c3|];"REGEBANT",t[|p3;pl;impf;ind;act;c3|]
        "REXI",t[|p1;sg;perf;ind;act;c3|];"REXISTI",t[|p2;sg;perf;ind;act;c3|]
        "REXIT",t[|p3;sg;perf;ind;act;c3|]
        "AGO",t[|p1;sg;pres;ind;act;c3|];"AGIS",t[|p2;sg;pres;ind;act;c3|]
        "AGIT",t[|p3;sg;pres;ind;act;c3|];"AGIMUS",t[|p1;pl;pres;ind;act;c3|]
        "AGITIS",t[|p2;pl;pres;ind;act;c3|];"AGUNT",t[|p3;pl;pres;ind;act;c3|]
        "EGI",t[|p1;sg;perf;ind;act;c3|];"EGISTI",t[|p2;sg;perf;ind;act;c3|]
        "EGIT",t[|p3;sg;perf;ind;act;c3|]
        "DUCO",t[|p1;sg;pres;ind;act;c3|];"DUCIS",t[|p2;sg;pres;ind;act;c3|]
        "DUCIT",t[|p3;sg;pres;ind;act;c3|];"DUCIMUS",t[|p1;pl;pres;ind;act;c3|]
        "DUCITIS",t[|p2;pl;pres;ind;act;c3|];"DUCUNT",t[|p3;pl;pres;ind;act;c3|]
        "DUXI",t[|p1;sg;perf;ind;act;c3|];"DUXISTI",t[|p2;sg;perf;ind;act;c3|]
        "DUXIT",t[|p3;sg;perf;ind;act;c3|] |]

    let unseen() = [|
        "NARRAT",t[|p3;sg;pres;ind;act;c1|];"NARRANT",t[|p3;pl;pres;ind;act;c1|]
        "NARRABAT",t[|p3;sg;impf;ind;act;c1|]
        "VIDET",t[|p3;sg;pres;ind;act;c2|];"VIDENT",t[|p3;pl;pres;ind;act;c2|]
        "VIDEBAT",t[|p3;sg;impf;ind;act;c2|]
        "SCRIBIT",t[|p3;sg;pres;ind;act;c3|];"SCRIBUNT",t[|p3;pl;pres;ind;act;c3|] |]

module Engine =
    let sigmoid x = if x >= 0.0 then 1.0/(1.0+Math.Exp(-x)) else Math.Exp(x)/(1.0+Math.Exp(x))

    type Model = {
        embedDim: int; hiddenDim: int; outputDim: int; lr: float
        embeddings: float[,]
        w1: float[,]; b1: float[]
        w2: float[,]; b2: float[]
    }

    let mutable charIndices: int[] = [||]
    let mutable pooled: float[] = [||]
    let mutable a1: float[] = [||]
    let mutable yPred: float[] = [||]

    // Pooling: false = original mean-pool. true = self-attention over the word's letter embeddings (Q = K = V = X,
    // bidirectional, no new parameters): O = softmax(X X^T / sqrt(D)) X, h = (1/n) sum_i O_i, using Attention.fs.
    let mutable poolAttention = false
    let mutable attentionIsa = Attention.detect ()
    let mutable xAttn: float[] = [||]
    let mutable oAttn: float[] = [||]
    let mutable lseAttn: float[] = [||]

    let init eD hD oD lr seed =
        let rng = Random(seed)
        let norm() =
            let u1 = 1.0 - rng.NextDouble()
            let u2 = rng.NextDouble()
            Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2)
        let es = Math.Sqrt(2.0/float(1+eD))
        let emb = Array2D.init 128 eD (fun _ _ -> norm()*es)
        let s1 = Math.Sqrt(2.0/float(eD+hD))
        let w1 = Array2D.init hD eD (fun _ _ -> norm()*s1)
        let s2 = Math.Sqrt(2.0/float(hD+oD))
        let w2 = Array2D.init oD hD (fun _ _ -> norm()*s2)
        { embedDim=eD; hiddenDim=hD; outputDim=oD; lr=lr; embeddings=emb; w1=w1; b1=Array.zeroCreate hD; w2=w2; b2=Array.zeroCreate oD }

    let forward m (word: string) =
        let chars = word.ToUpperInvariant().ToCharArray() |> Array.map int
        let len = chars.Length
        let D = m.embedDim
        let H = m.hiddenDim
        let K = m.outputDim
        charIndices <- chars
        if poolAttention then
            let x = Array.zeroCreate<float> (len * D)
            for t = 0 to len - 1 do
                for d = 0 to D - 1 do x.[t * D + d] <- m.embeddings.[chars.[t], d]
            let o = Array.zeroCreate<float> (len * D)
            let lse = Array.zeroCreate<float> len
            Attention.forward attentionIsa D x x x o lse len false
            xAttn <- x
            oAttn <- o
            lseAttn <- lse
            pooled <- Array.init D (fun d ->
                let mutable s = 0.0
                for t = 0 to len - 1 do s <- s + o.[t * D + d]
                s / float len)
        else
            let embV = Array.init len (fun t -> Array.init D (fun d -> m.embeddings.[chars.[t],d]))
            pooled <- Array.init D (fun d ->
                let mutable s = 0.0
                for t = 0 to len - 1 do s <- s + embV.[t].[d]
                s / float len)
        let z1 = Array.init H (fun i ->
            let mutable s = m.b1.[i]
            for j = 0 to D - 1 do s <- s + m.w1.[i,j] * pooled.[j]
            s)
        a1 <- z1 |> Array.map Math.Tanh
        let z2 = Array.init K (fun i ->
            let mutable s = m.b2.[i]
            for j = 0 to H - 1 do s <- s + m.w2.[i,j] * a1.[j]
            s)
        yPred <- z2 |> Array.map sigmoid
        yPred

    let computeLoss (target: float[]) =
        let eps = 1e-12
        let K = yPred.Length
        let mutable loss = 0.0
        for i = 0 to K - 1 do
            let y = Math.Clamp(yPred.[i], eps, 1.0 - eps)
            loss <- loss - target.[i] * Math.Log(y) - (1.0 - target.[i]) * Math.Log(1.0 - y)
        loss / float K

    let backward m (target: float[]) =
        let len = charIndices.Length
        let D = m.embedDim
        let H = m.hiddenDim
        let K = m.outputDim
        let invD = 1.0 / float K
        let invN = 1.0 / float len
        let dl_dz2 = Array.init K (fun i -> (yPred.[i] - target.[i]) * invD)
        let dl_da1 = Array.init H (fun j ->
            let mutable s = 0.0
            for i = 0 to K - 1 do s <- s + m.w2.[i,j] * dl_dz2.[i]
            s)
        let dl_dz1 = Array.init H (fun j -> dl_da1.[j] * (1.0 - a1.[j] * a1.[j]))
        let dl_dh = Array.init D (fun j ->
            let mutable s = 0.0
            for i = 0 to H - 1 do s <- s + m.w1.[i,j] * dl_dz1.[i]
            s)
        if poolAttention then
            // h = (1/n) sum_i O_i  =>  dO_i = dh / n for every row; with Q = K = V = X the gradient at position i is dQ_i + dK_i + dV_i
            let dOut = Array.init (len * D) (fun idx -> invN * dl_dh.[idx % D])
            let dq = Array.zeroCreate<float> (len * D)
            let dk = Array.zeroCreate<float> (len * D)
            let dv = Array.zeroCreate<float> (len * D)
            Attention.backward attentionIsa D xAttn xAttn xAttn oAttn lseAttn dOut dq dk dv (Array.zeroCreate len) len false
            for t = 0 to len - 1 do
                let c = charIndices.[t]
                for d = 0 to D - 1 do
                    m.embeddings.[c,d] <- m.embeddings.[c,d] - m.lr * (dq.[t * D + d] + dk.[t * D + d] + dv.[t * D + d])
        else
            for t = 0 to len - 1 do
                let c = charIndices.[t]
                for d = 0 to D - 1 do
                    m.embeddings.[c,d] <- m.embeddings.[c,d] - m.lr * invN * dl_dh.[d]
        for i = 0 to H - 1 do
            m.b1.[i] <- m.b1.[i] - m.lr * dl_dz1.[i]
            for j = 0 to D - 1 do
                m.w1.[i,j] <- m.w1.[i,j] - m.lr * dl_dz1.[i] * pooled.[j]
        for i = 0 to K - 1 do
            m.b2.[i] <- m.b2.[i] - m.lr * dl_dz2.[i]
            for j = 0 to H - 1 do
                m.w2.[i,j] <- m.w2.[i,j] - m.lr * dl_dz2.[i] * a1.[j]

/// Checks on the real engine: gradients reaching the embeddings through each pooling, training behaviour, determinism.
module EngineChecks =
    let private fresh () = Engine.init 16 32 Features.count 0.5 42

    /// Central finite differences of the loss w.r.t. every embedding entry of the word's letters, against the gradient
    /// that Engine.backward applied ((before - after) / lr on a fresh model). Rows of letters absent from the word must not move.
    let gradCheck (attn: bool) (word: string) =
        Engine.poolAttention <- attn
        let target = Dataset.corpus() |> Array.find (fun (w, _) -> w = word) |> snd
        let m0 = fresh ()
        let before = Array2D.copy m0.embeddings
        Engine.forward m0 word |> ignore
        Engine.backward m0 target
        let letters = word.ToUpperInvariant() |> Seq.map int |> Seq.distinct |> Seq.toArray
        let h = 1e-4
        let mutable nonzero = false
        for c in letters do
            for d = 0 to 15 do
                let analytic = (before.[c, d] - m0.embeddings.[c, d]) / m0.lr
                let lossAt delta =
                    let m = fresh ()
                    m.embeddings.[c, d] <- m.embeddings.[c, d] + delta
                    Engine.forward m word |> ignore
                    Engine.computeLoss target
                let fd = (lossAt h - lossAt (-h)) / (2.0 * h)
                if abs (fd - analytic) > 1e-8 + 1e-6 * abs analytic then
                    failwithf "FAIL gradient (attn=%b) %s emb[%d,%d]: fd=%g analytic=%g" attn word c d fd analytic
                if abs analytic > 1e-9 then nonzero <- true
        if not nonzero then failwithf "FAIL (attn=%b) %s: embedding gradient is identically zero" attn word
        for c = 0 to 127 do
            if not (Array.contains c letters) then
                for d = 0 to 15 do
                    if before.[c, d] <> m0.embeddings.[c, d] then failwithf "FAIL (attn=%b) %s: row %d moved but is not in the word" attn word c

    let private epoch (m: Engine.Model) (corpus: (string * float[])[]) =
        let mutable total = 0.0
        for (w, t) in corpus do
            Engine.forward m w |> ignore
            total <- total + Engine.computeLoss t
            Engine.backward m t
        total / float corpus.Length

    let private train attn isa epochs =
        Engine.poolAttention <- attn
        Engine.attentionIsa <- isa
        let m = fresh ()
        let corpus = Dataset.corpus ()
        let mutable last = 0.0
        for _ in 1 .. epochs do last <- epoch m corpus
        last, Array2D.copy m.embeddings

    let runAll () =
        for attn in [ false; true ] do
            for word in [ "AMO"; "AMABAMUS"; "REGEBATIS" ] do gradCheck attn word
        printfn "engine gradients (mean and attention pooling) match finite differences; only the word's letters move"
        let first, _ = train true (Attention.detect ()) 1
        let last, _ = train true (Attention.detect ()) 60
        if not (last < first * 0.5) then failwithf "FAIL attention training did not reduce loss: %g -> %g" first last
        let a, ea = train true (Attention.detect ()) 3
        let b, eb = train true (Attention.detect ()) 3
        if a <> b || ea <> eb then failwith "FAIL attention training is not deterministic"
        if Attention.avx2FmaAvailable then
            let ls, _ = train true Attention.Scalar 3
            let lv, _ = train true Attention.Avx2Fma 3
            if abs (ls - lv) > 1e-10 then failwithf "FAIL scalar vs AVX2 training: %g vs %g" ls lv
        Engine.poolAttention <- false
        printfn "attention training: loss falls %.4f -> %.4f in 60 epochs, deterministic, scalar == AVX2+FMA" first last

[<EntryPoint>]
let main argv =
  // dotnet run -c Release --project src/fsharp -- [mean|attn] [simd|scalar]   benchmark suite (default: mean pooling)
  //                                              selftest                     attention kernel + engine checks
  //                                              bench-attention              attention kernel timings
  match List.ofArray argv with
  | "selftest" :: _ ->
    try
        AttentionTests.runAll ()
        EngineChecks.runAll ()
        0
    with e ->
        eprintfn "%s" e.Message
        1
  | "bench-attention" :: _ ->
    AttentionTests.bench ()
    0
  | args ->
    match args with
    | [] | "mean" :: _ -> ()
    | "attn" :: _ -> Engine.poolAttention <- true
    | x :: _ -> failwithf "unknown pool '%s' (expected mean or attn)" x
    match args with
    | _ :: "scalar" :: _ -> Engine.attentionIsa <- Attention.Scalar
    | _ :: "simd" :: _ | [ _ ] | [] -> ()
    | _ :: x :: _ -> failwithf "unknown kernel '%s' (expected simd or scalar)" x
    printfn "================================================================="
    printfn "  F# BENCHMARK SUITE"
    printfn "================================================================="
    if Engine.poolAttention then printfn "  Pooling: attention (kernel: %s)" (Attention.isaName Engine.attentionIsa)
    let corpus = Dataset.corpus()
    let unseen = Dataset.unseen()
    let m = Engine.init 16 32 Features.count 0.5 42
    printfn "  Corpus: %d words, %d features, %d unseen" corpus.Length Features.count unseen.Length
    printfn ""

    // BENCHMARK 1: Training
    printfn "  BENCHMARK 1: TRAINING (3000 epochs x %d examples)" corpus.Length
    let sw = Stopwatch.StartNew()
    let mutable finalLoss = 0.0
    for ep = 0 to 3000 do
        let mutable totalLoss = 0.0
        for i = 0 to corpus.Length - 1 do
            let w, t = corpus.[i]
            Engine.forward m w |> ignore
            totalLoss <- totalLoss + Engine.computeLoss t
            Engine.backward m t
        finalLoss <- totalLoss / float corpus.Length
        if ep % 500 = 0 || ep = 3000 then
            printfn "  Epoch %5d | Loss: %10.6f | %.2fs" ep finalLoss sw.Elapsed.TotalSeconds
    sw.Stop()
    let totalEx = int64 3001 * int64 corpus.Length
    printfn "\n  Training: %.3fs, %d examples, %.0f ex/sec" sw.Elapsed.TotalSeconds totalEx (float totalEx / sw.Elapsed.TotalSeconds)

    // BENCHMARK 2: Inference
    printfn "\n  BENCHMARK 2: INFERENCE LATENCY"
    for i = 0 to corpus.Length - 1 do Engine.forward m (fst corpus.[i]) |> ignore
    let nInf = 1000
    let infSw = Stopwatch.StartNew()
    for _ = 0 to nInf - 1 do
        for i = 0 to corpus.Length - 1 do
            Engine.forward m (fst corpus.[i]) |> ignore
    infSw.Stop()
    let totalInf = int64 nInf * int64 corpus.Length
    printfn "  %d inferences in %.3fs, %.2f us/inf, %.0f inf/sec" totalInf infSw.Elapsed.TotalSeconds (float infSw.Elapsed.TotalMicroseconds / float totalInf) (float totalInf / infSw.Elapsed.TotalSeconds)

    // BENCHMARK 3: Forward+Backward
    printfn "\n  BENCHMARK 3: FORWARD+BACKWARD LATENCY"
    let nFb = 1000
    let fbSw = Stopwatch.StartNew()
    for _ = 0 to nFb - 1 do
        for i = 0 to corpus.Length - 1 do
            let w, t = corpus.[i]
            Engine.forward m w |> ignore
            Engine.computeLoss t |> ignore
            Engine.backward m t
    fbSw.Stop()
    let totalFb = int64 nFb * int64 corpus.Length
    printfn "  %d passes in %.3fs, %.2f us/pass, %.0f pass/sec" totalFb fbSw.Elapsed.TotalSeconds (float fbSw.Elapsed.TotalMicroseconds / float totalFb) (float totalFb / fbSw.Elapsed.TotalSeconds)

    // BENCHMARK 4: Quality
    printfn "\n  BENCHMARK 4: GENERALIZATION"
    let mutable trainOk = 0
    for i = 0 to corpus.Length - 1 do
        let w, t = corpus.[i]
        let p = Engine.forward m w
        if Features.countCorrect p t = Features.count then trainOk <- trainOk + 1
    let mutable unseenOk = 0
    for i = 0 to unseen.Length - 1 do
        let w, t = unseen.[i]
        let p = Engine.forward m w
        if Features.countCorrect p t = Features.count then unseenOk <- unseenOk + 1
    printfn "  Train: %d/%d perfect | Unseen: %d/%d perfect" trainOk corpus.Length unseenOk unseen.Length

    printfn "\n================================================================="
    printfn "  F# BENCHMARK COMPLETE"
    printfn "================================================================="
    0
