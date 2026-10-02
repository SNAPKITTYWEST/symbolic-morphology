// End-to-end tests through the real `lean` binary: argument handling, the mean-pool regression against the original
// engine, determinism, and agreement between ISA / thread-count settings of the attention training path.
use std::process::{Command, Output};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/shared");

fn lean(args: &[&str]) -> Output { Command::new(env!("CARGO_BIN_EXE_lean")).arg(DIR).args(args).output().expect("run lean") }

/// The final summary line, with the (non-deterministic) timing fields removed, and its final_loss.
fn summary(o: &Output) -> (String, f64) {
    assert!(o.status.success(), "lean failed: {}", String::from_utf8_lossy(&o.stderr));
    let out = String::from_utf8(o.stdout.clone()).unwrap();
    let last = out.lines().last().unwrap().to_string();
    let loss = last.split(' ').find_map(|t| t.strip_prefix("final_loss=")).unwrap().parse().unwrap();
    (last.split(" time=").next().unwrap().to_string(), loss)
}

#[test] fn mean_pool_reproduces_the_cross_language_final_loss() {
    // 9.1009e-5 is the loss every runtime in bench/results.md reaches on the shared corpus after 3001 epochs. The full run takes
    // ~1 s in release but ~1 min unoptimised, so debug builds check the 300-epoch value instead (recorded from the same path,
    // which lean.rs's unit tests prove is bit-identical to the original engine).
    if cfg!(debug_assertions) {
        let (line, _) = summary(&lean(&["mean", "300"]));
        assert_eq!(line, "rust-lean  examples=96 epochs=300 final_loss=0.001349468");
    } else {
        let (line, loss) = summary(&lean(&["mean", "3001"]));
        assert_eq!(line, "rust-lean  examples=96 epochs=3001 final_loss=0.000091009");
        assert!((loss - 9.1009e-5).abs() < 1e-8);
    }
}

#[test] fn default_pool_is_attention() {
    // The defaults themselves are unit-tested in lean.rs (parse_args); this runs the full default invocation, so release builds only.
    if cfg!(debug_assertions) { return; }
    let (line, _) = summary(&lean(&[]));   // no pool argument
    assert!(line.starts_with("rust-lean-attn  examples=96 epochs=3001"), "{line}");
}

#[test] fn attention_regression_values() {
    // Recorded from a scalar, single-thread run; other ISAs/thread counts must land within 1e-6 absolute of it.
    let (_, scalar) = summary(&lean(&["attn", "300", "scalar", "1"]));
    assert!((scalar - 0.001235581).abs() < 1e-8, "scalar attn loss drifted: {scalar}");
    let (_, mean) = summary(&lean(&["mean", "300"]));
    assert!((mean - 0.001349468).abs() < 1e-8, "mean-pool loss drifted: {mean}");
    assert!((scalar - mean).abs() > 1e-5, "attention and mean-pool trained identically — the attention op is not in the training path");
}

#[test] fn attention_runs_are_reproducible_and_thread_invariant() {
    let a = lean(&["attn", "50", "scalar", "1"]); let b = lean(&["attn", "50", "scalar", "1"]);
    let strip = |o: &Output| String::from_utf8(o.stdout.clone()).unwrap().lines().map(|l| l.split(" time=").next().unwrap().to_string()).collect::<Vec<_>>();
    assert_eq!(strip(&a), strip(&b), "identical invocations differ");
    for t in ["2", "3", "4"] {
        let c = lean(&["attn", "50", "scalar", t]);
        let (sa, sc) = (strip(&a), strip(&c));
        assert_eq!(sa[1], sc[1], "threads={t}: summary line differs");
        assert_eq!(sa[0].split("threads=").nth(1).unwrap().split(' ').nth(1), sc[0].split("threads=").nth(1).unwrap().split(' ').nth(1), "threads={t}: embedding checksum differs");
    }
}

#[test] fn simd_and_scalar_attention_agree() {
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") { return; }
    let (_, s) = summary(&lean(&["attn", "300", "scalar", "1"]));
    let (_, v) = summary(&lean(&["attn", "300", "avx2", "1"]));
    let (_, vt) = summary(&lean(&["attn", "300", "avx2", "4"]));
    assert!((s - v).abs() < 1e-6, "scalar {s} vs avx2 {v}");
    assert_eq!(v.to_bits(), vt.to_bits(), "avx2 threaded result differs from avx2 single-thread");
}

#[test] fn bad_arguments_fail_explicitly() {
    for args in [&["bogus"][..], &["attn", "x"], &["attn", "10", "neon"], &["attn", "10", "scalar", "y"]] {
        let o = lean(args);
        assert_eq!(o.status.code(), Some(2), "{args:?}");
        assert!(String::from_utf8_lossy(&o.stderr).contains("usage:"));
    }
    // zero threads is rejected by the kernel, not silently coerced
    let o = lean(&["attn", "1", "scalar", "0"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("threads must be >= 1"), "{}", String::from_utf8_lossy(&o.stderr));
}
