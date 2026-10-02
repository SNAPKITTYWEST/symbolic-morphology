// Shared library for the Rust binaries. Only the attention kernel lives here; the original engine
// (main.rs, engine.rs, features.rs, dataset.rs) and the other bins are untouched.
pub mod attention;
