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

mod features;
mod engine;
mod dataset;

use features::{NUM_FEATURES, format_prediction, count_correct};
use engine::Engine;
use dataset::{build_corpus, unseen_words};
use std::time::Instant;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SYMBOLIC LEARNING ENGINE — BENCHMARK SUITE");
    println!("  Rust native binary — zero dependencies");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let corpus = build_corpus();
    let unseen = unseen_words();

    // ─── Initialize engine ───────────────────────────────────────
    const EMBED_DIM: usize = 16;
    const HIDDEN_DIM: usize = 32;
    const LEARNING_RATE: f64 = 0.5;
    const EPOCHS: usize = 3000;

    let mut engine = Engine::new(EMBED_DIM, HIDDEN_DIM, LEARNING_RATE, 42);

    println!("  Corpus: {} word forms, {} features, {} unseen", corpus.len(), NUM_FEATURES, unseen.len());
    println!("  Arch: {}-dim embed -> mean pool -> {}-dim hidden (tanh) -> {}-dim output (sigmoid)", EMBED_DIM, HIDDEN_DIM, NUM_FEATURES);
    println!();

    // ═══════════════════════════════════════════════════════════════
    //  BENCHMARK 1: TRAINING
    // ═══════════════════════════════════════════════════════════════

    println!("─────────────────────────────────────────────────────────────");
    println!("  BENCHMARK 1: TRAINING ({} epochs x {} examples)", EPOCHS, corpus.len());
    println!("─────────────────────────────────────────────────────────────");

    let train_start = Instant::now();
    let mut final_loss = 0.0;

    for epoch in 0..=EPOCHS {
        let mut total_loss: f64 = 0.0;

        for (word, target) in &corpus {
            engine.forward(word);
            let loss = engine.compute_loss(target);
            total_loss += loss;
            engine.backward(target);
        }

        final_loss = total_loss / corpus.len() as f64;

        if epoch % 500 == 0 || epoch == EPOCHS {
            let elapsed = train_start.elapsed().as_secs_f64();
            let mut total_correct = 0usize;
            let mut total_features = 0usize;
            for (word, target) in &corpus {
                let pred = engine.forward(word);
                total_correct += count_correct(&pred, target);
                total_features += NUM_FEATURES;
            }
            let accuracy = total_correct as f64 / total_features as f64 * 100.0;
            println!("  Epoch {:>5} | Loss: {:>10.6} | Acc: {:>6.1}% | {:.2}s",
                epoch, final_loss, accuracy, elapsed);
        }
    }

    let train_time = train_start.elapsed().as_secs_f64();
    let total_examples = (EPOCHS as u64 + 1) * corpus.len() as u64;
    let examples_per_sec = total_examples as f64 / train_time;

    println!();
    println!("  TRAINING RESULTS:");
    println!("    Wall clock:       {:.3}s", train_time);
    println!("    Total examples:   {}", total_examples);
    println!("    Throughput:       {:.0} examples/sec", examples_per_sec);
    println!("    Final loss:       {:.6}", final_loss);
    println!();

    // ═══════════════════════════════════════════════════════════════
    //  BENCHMARK 2: INFERENCE LATENCY
    // ═══════════════════════════════════════════════════════════════

    println!("─────────────────────────────────────────────────────────────");
    println!("  BENCHMARK 2: INFERENCE LATENCY");
    println!("─────────────────────────────────────────────────────────────");

    // Warm up
    for (word, _) in &corpus {
        engine.forward(word);
    }

    let bench_words: Vec<&str> = corpus.iter().map(|(w, _)| w.as_str()).collect();
    let n_inference = 1000;
    let inf_start = Instant::now();

    for _ in 0..n_inference {
        for word in &bench_words {
            engine.forward(word);
        }
    }

    let inf_time = inf_start.elapsed().as_secs_f64();
    let total_inferences = n_inference * bench_words.len();
    let us_per_inference = inf_time * 1_000_000.0 / total_inferences as f64;

    println!("  {} inferences ({} words x {} passes)", total_inferences, bench_words.len(), n_inference);
    println!("  Wall clock:       {:.3}s", inf_time);
    println!("  Per inference:    {:.2} µs", us_per_inference);
    println!("  Throughput:       {:.0} inferences/sec", total_inferences as f64 / inf_time);
    println!();

    // ═══════════════════════════════════════════════════════════════
    //  BENCHMARK 3: FULL FORWARD+BACKWARD LATENCY
    // ═══════════════════════════════════════════════════════════════

    println!("─────────────────────────────────────────────────────────────");
    println!("  BENCHMARK 3: FORWARD + BACKWARD LATENCY");
    println!("─────────────────────────────────────────────────────────────");

    let n_fb = 1000;
    let fb_start = Instant::now();

    for _ in 0..n_fb {
        for (word, target) in &corpus {
            engine.forward(word);
            engine.compute_loss(target);
            engine.backward(target);
        }
    }

    let fb_time = fb_start.elapsed().as_secs_f64();
    let total_fb = n_fb * corpus.len();
    let us_per_fb = fb_time * 1_000_000.0 / total_fb as f64;

    println!("  {} forward+backward passes ({} words x {} passes)", total_fb, corpus.len(), n_fb);
    println!("  Wall clock:       {:.3}s", fb_time);
    println!("  Per pass:         {:.2} µs", us_per_fb);
    println!("  Throughput:       {:.0} passes/sec", total_fb as f64 / fb_time);
    println!();

    // ═══════════════════════════════════════════════════════════════
    //  BENCHMARK 4: GENERALIZATION QUALITY
    // ═══════════════════════════════════════════════════════════════

    println!("─────────────────────────────────────────────────────────────");
    println!("  BENCHMARK 4: GENERALIZATION QUALITY");
    println!("─────────────────────────────────────────────────────────────");

    let mut train_correct = 0;
    let mut train_total = 0;
    let mut train_perfect = 0;
    for (word, target) in &corpus {
        let pred = engine.forward(word);
        let c = count_correct(&pred, target);
        train_correct += c;
        train_total += NUM_FEATURES;
        if c == NUM_FEATURES { train_perfect += 1; }
    }

    let mut unseen_correct = 0;
    let mut unseen_total = 0;
    let mut unseen_perfect = 0;
    for (word, target) in &unseen {
        let pred = engine.forward(word);
        let c = count_correct(&pred, target);
        unseen_correct += c;
        unseen_total += NUM_FEATURES;
        if c == NUM_FEATURES { unseen_perfect += 1; }
    }

    println!("  Training set:  {}/{} perfect words, {:.1}% feature accuracy",
        train_perfect, corpus.len(),
        train_correct as f64 / train_total as f64 * 100.0);
    println!("  Unseen set:    {}/{} perfect words, {:.1}% feature accuracy",
        unseen_perfect, unseen.len(),
        unseen_correct as f64 / unseen_total as f64 * 100.0);
    println!();

    // ═══════════════════════════════════════════════════════════════
    //  SAMPLE INFERENCE
    // ═══════════════════════════════════════════════════════════════

    println!("─────────────────────────────────────────────────────────────");
    println!("  SAMPLE INFERENCE");
    println!("─────────────────────────────────────────────────────────────");

    for word in &["AMO", "AMABAT", "MONET", "REGIT", "AGUNT"] {
        let target = corpus.iter().find(|(w, _)| w == word).map(|(_, t)| *t).unwrap();
        let pred = engine.forward(word);
        let errors = NUM_FEATURES - count_correct(&pred, &target);
        println!();
        println!("  INPUT: {}", word);
        println!("{}", format_prediction(&pred));
        println!("  ERRORS: {}/{}", errors, NUM_FEATURES);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  BENCHMARK COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
}
