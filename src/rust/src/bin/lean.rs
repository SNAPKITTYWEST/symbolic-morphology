// Allocation-free Rust port of the same online-SGD loop (fixed-size stack arrays, flat weights).
// Reads the shared corpus/init files, so results are directly comparable with the other runtimes.
use std::time::Instant;
const E: usize = 16; const H: usize = 32; const F: usize = 23; const LR: f64 = 0.5;

fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) } }

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "../../bench/shared".into());
    let v: Vec<f64> = std::fs::read_to_string(format!("{dir}/init.txt")).unwrap()
        .lines().map(|l| l.parse().unwrap()).collect();
    let mut emb = [[0f64; E]; 128]; let mut w1 = [[0f64; E]; H]; let mut b1 = [0f64; H];
    let mut w2 = [[0f64; H]; F]; let mut b2 = [0f64; F];
    let mut k = 0;
    for r in emb.iter_mut() { for x in r { *x = v[k]; k += 1; } }
    for r in w1.iter_mut() { for x in r { *x = v[k]; k += 1; } }
    for x in b1.iter_mut() { *x = v[k]; k += 1; }
    for r in w2.iter_mut() { for x in r { *x = v[k]; k += 1; } }
    for x in b2.iter_mut() { *x = v[k]; k += 1; }

    let mut words: Vec<Vec<usize>> = vec![]; let mut tgt: Vec<[f64; F]> = vec![];
    for l in std::fs::read_to_string(format!("{dir}/corpus.txt")).unwrap().lines() {
        let mut it = l.split(' ');
        words.push(it.next().unwrap().bytes().map(|b| b as usize).collect());
        let mut t = [0f64; F];
        for (i, x) in it.enumerate() { t[i] = x.parse().unwrap(); }
        tgt.push(t);
    }
    let n = words.len();
    let t0 = Instant::now();
    let mut last = 0.0;
    for _ in 0..=3000 {
        let mut total = 0.0;
        for s in 0..n {
            let w = &words[s]; let t = &tgt[s]; let len = w.len() as f64;
            let mut h = [0f64; E];
            for d in 0..E { let mut sum = 0.0; for &c in w { sum += emb[c][d]; } h[d] = sum / len; }
            let mut a1 = [0f64; H];
            for i in 0..H { let mut z = b1[i]; for j in 0..E { z += w1[i][j] * h[j]; } a1[i] = z.tanh(); }
            let mut y = [0f64; F];
            for i in 0..F { let mut z = b2[i]; for j in 0..H { z += w2[i][j] * a1[j]; } y[i] = sigmoid(z); }
            let mut loss = 0.0;
            for i in 0..F { let p = y[i].clamp(1e-12, 1.0 - 1e-12); loss -= t[i] * p.ln() + (1.0 - t[i]) * (1.0 - p).ln(); }
            total += loss / F as f64;
            let mut dz2 = [0f64; F];
            for i in 0..F { dz2[i] = (y[i] - t[i]) / F as f64; }
            let mut dz1 = [0f64; H];
            for j in 0..H { let mut sum = 0.0; for i in 0..F { sum += w2[i][j] * dz2[i]; } dz1[j] = sum * (1.0 - a1[j] * a1[j]); }
            let mut dh = [0f64; E];
            for j in 0..E { let mut sum = 0.0; for i in 0..H { sum += w1[i][j] * dz1[i]; } dh[j] = sum; }
            let inv_n = 1.0 / len;
            for &c in w { for d in 0..E { emb[c][d] -= LR * (inv_n * dh[d]); } }
            for i in 0..H { b1[i] -= LR * dz1[i]; for j in 0..E { w1[i][j] -= LR * (dz1[i] * h[j]); } }
            for i in 0..F { b2[i] -= LR * dz2[i]; for j in 0..H { w2[i][j] -= LR * (dz2[i] * a1[j]); } }
        }
        last = total / n as f64;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("rust-lean  examples={} epochs=3001 final_loss={:.9} time={:.3}s ex/s={:.0}", n, last, dt, (n * 3001) as f64 / dt);
}
