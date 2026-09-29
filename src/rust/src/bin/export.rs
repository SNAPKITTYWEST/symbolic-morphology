// Exports the shared benchmark inputs (corpus + seed-42 initial weights) so every
// language implementation trains from byte-identical data, and prints the Rust
// reference loss/timing to compare against.
#[path = "../features.rs"] mod features;
#[path = "../engine.rs"] mod engine;
#[path = "../dataset.rs"] mod dataset;
use engine::Engine;
use std::fmt::Write;
use std::time::Instant;

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "../../bench/shared".into());
    let corpus = dataset::build_corpus();
    let mut e = Engine::new(16, 32, 0.5, 42);

    let mut data = String::new();
    for (w, t) in &corpus {
        write!(data, "{}", w).unwrap();
        for v in t { write!(data, " {}", *v as i32).unwrap(); }
        data.push('\n');
    }
    std::fs::write(format!("{out}/corpus.txt"), data).unwrap();

    let mut init = String::new();
    for r in &e.embeddings { for v in r { writeln!(init, "{:e}", v).unwrap(); } }
    for r in &e.w1 { for v in r { writeln!(init, "{:e}", v).unwrap(); } }
    for v in &e.b1 { writeln!(init, "{:e}", v).unwrap(); }
    for r in &e.w2 { for v in r { writeln!(init, "{:e}", v).unwrap(); } }
    for v in &e.b2 { writeln!(init, "{:e}", v).unwrap(); }
    std::fs::write(format!("{out}/init.txt"), init).unwrap();

    let epochs = 3000usize;
    let t0 = Instant::now();
    let mut last = 0.0;
    for _ in 0..=epochs {
        let mut tl = 0.0;
        for (w, t) in &corpus { e.forward(w); tl += e.compute_loss(t); e.backward(t); }
        last = tl / corpus.len() as f64;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("rust  examples={} epochs={} final_loss={:.9} time={:.3}s ex/s={:.0}",
        corpus.len(), epochs + 1, last, dt, (corpus.len() * (epochs + 1)) as f64 / dt);
}
