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

use crate::features::NUM_FEATURES;

/// LEARNING LAYER — Differentiable computation graph.
///
/// Architecture:
///   1. Character embedding table (learnable)
///   2. Mean pooling over character positions (handles variable-length input)
///   3. Hidden layer: z1 = W1·h + b1,  a1 = tanh(z1)
///   4. Output layer: z2 = W2·a1 + b2,  y_hat = sigmoid(z2)
///   5. Binary cross-entropy loss
///   6. Manual chain-rule backpropagation
///   7. Gradient descent parameter update
///
/// Every derivative is computed by hand. No autodiff framework.
pub struct Engine {
    pub embed_dim: usize,
    pub hidden_dim: usize,

    // Learnable parameters
    // Embedding table: maps each uppercase ASCII letter to a vector
    pub embeddings: Vec<Vec<f64>>,   // [128][embed_dim]

    // Hidden layer: z1 = W1·h + b1
    pub w1: Vec<Vec<f64>>,           // [hidden_dim][embed_dim]
    pub b1: Vec<f64>,                // [hidden_dim]

    // Output layer: z2 = W2·a1 + b2
    pub w2: Vec<Vec<f64>>,           // [NUM_FEATURES][hidden_dim]
    pub b2: Vec<f64>,                // [NUM_FEATURES]

    pub learning_rate: f64,

    // Cached forward state for one sample (used by backward)
    char_indices: Vec<usize>,
    emb_vecs: Vec<Vec<f64>>,
    pooled: Vec<f64>,
    z1: Vec<f64>,
    a1: Vec<f64>,
    _z2: Vec<f64>,
    y_pred: Vec<f64>,
}

impl Engine {
    pub fn new(embed_dim: usize, hidden_dim: usize, learning_rate: f64, seed: u64) -> Self {
        let mut rng = Rng::new(seed);

        // Xavier/Glorot initialization
        let emb_scale = (2.0 / (1.0 + embed_dim as f64)).sqrt();
        let mut embeddings = vec![vec![0.0; embed_dim]; 128];
        for c in 0..128 {
            for d in 0..embed_dim {
                embeddings[c][d] = rng.normal() * emb_scale;
            }
        }

        let w1_scale = (2.0 / (embed_dim as f64 + hidden_dim as f64)).sqrt();
        let mut w1 = vec![vec![0.0; embed_dim]; hidden_dim];
        for i in 0..hidden_dim {
            for j in 0..embed_dim {
                w1[i][j] = rng.normal() * w1_scale;
            }
        }
        let b1 = vec![0.0; hidden_dim];

        let w2_scale = (2.0 / (hidden_dim as f64 + NUM_FEATURES as f64)).sqrt();
        let mut w2 = vec![vec![0.0; hidden_dim]; NUM_FEATURES];
        for i in 0..NUM_FEATURES {
            for j in 0..hidden_dim {
                w2[i][j] = rng.normal() * w2_scale;
            }
        }
        let b2 = vec![0.0; NUM_FEATURES];

        Engine {
            embed_dim,
            hidden_dim,
            embeddings,
            w1,
            b1,
            w2,
            b2,
            learning_rate,
            char_indices: Vec::new(),
            emb_vecs: Vec::new(),
            pooled: Vec::new(),
            z1: Vec::new(),
            a1: Vec::new(),
            _z2: Vec::new(),
            y_pred: Vec::new(),
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  FORWARD PASS
    // ═══════════════════════════════════════════════════════════════
    //
    //  For input word "AMO" (characters A, M, O):
    //
    //  Step 1 — Embedding lookup:
    //    e_A = embeddings['A'],  e_M = embeddings['M'],  e_O = embeddings['O']
    //
    //  Step 2 — Mean pooling (variable-length -> fixed-size):
    //    h = (e_A + e_M + e_O) / 3
    //
    //  Step 3 — Hidden layer:
    //    z1 = W1 * h + b1
    //    a1 = tanh(z1)
    //
    //  Step 4 — Output layer:
    //    z2 = W2 * a1 + b2
    //    y_hat = sigmoid(z2)
    //
    // ═══════════════════════════════════════════════════════════════

    pub fn forward(&mut self, word: &str) -> [f64; NUM_FEATURES] {
        let chars: Vec<usize> = word.chars()
            .map(|c| c.to_ascii_uppercase() as usize)
            .collect();
        let len = chars.len();

        // Step 1: Character embedding lookup
        self.char_indices = chars.clone();
        self.emb_vecs = vec![vec![0.0; self.embed_dim]; len];
        for (t, &c) in chars.iter().enumerate() {
            for d in 0..self.embed_dim {
                self.emb_vecs[t][d] = self.embeddings[c][d];
            }
        }

        // Step 2: Mean pooling — h = (1/N) * sum(e_t)
        self.pooled = vec![0.0; self.embed_dim];
        for d in 0..self.embed_dim {
            let mut sum = 0.0;
            for t in 0..len {
                sum += self.emb_vecs[t][d];
            }
            self.pooled[d] = sum / len as f64;
        }

        // Step 3: Hidden layer — z1 = W1*h + b1,  a1 = tanh(z1)
        self.z1 = vec![0.0; self.hidden_dim];
        self.a1 = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut z = self.b1[i];
            for j in 0..self.embed_dim {
                z += self.w1[i][j] * self.pooled[j];
            }
            self.z1[i] = z;
            self.a1[i] = z.tanh();
        }

        // Step 4: Output layer — z2 = W2*a1 + b2,  y_hat = sigmoid(z2)
        self._z2 = vec![0.0; NUM_FEATURES];
        self.y_pred = vec![0.0; NUM_FEATURES];
        for i in 0..NUM_FEATURES {
            let mut z = self.b2[i];
            for j in 0..self.hidden_dim {
                z += self.w2[i][j] * self.a1[j];
            }
            self._z2[i] = z;
            self.y_pred[i] = sigmoid(z);
        }

        let mut result = [0.0; NUM_FEATURES];
        result.copy_from_slice(&self.y_pred);
        result
    }

    // ═══════════════════════════════════════════════════════════════
    //  LOSS — Binary Cross-Entropy
    // ═══════════════════════════════════════════════════════════════
    //
    //  L = -(1/D) * sum_i [ y_i * log(y_hat_i) + (1-y_i) * log(1-y_hat_i) ]
    //
    //  where D = number of output features (23)
    //
    // ═══════════════════════════════════════════════════════════════

    pub fn compute_loss(&self, target: &[f64; NUM_FEATURES]) -> f64 {
        let eps = 1e-12;
        let mut loss = 0.0;
        for i in 0..NUM_FEATURES {
            let y = self.y_pred[i].clamp(eps, 1.0 - eps);
            loss -= target[i] * y.ln() + (1.0 - target[i]) * (1.0 - y).ln();
        }
        loss / NUM_FEATURES as f64
    }

    // ═══════════════════════════════════════════════════════════════
    //  BACKWARD PASS — Classical Chain Rule
    // ═══════════════════════════════════════════════════════════════
    //
    //  For every operation in the forward graph, we compute the local
    //  derivative and propagate gradients backward:
    //
    //    gradient_in = gradient_out * local_derivative
    //
    //  At branching points (where a value feeds multiple paths):
    //
    //    gradient = sum(all incoming gradient contributions)
    //
    //  The full chain for this network:
    //
    //    dL/dz2  = (y_hat - y) / D                    [BCE + sigmoid]
    //    dL/dW2  = dL/dz2 * a1^T                      [linear layer]
    //    dL/db2  = dL/dz2                              [bias]
    //    dL/da1  = W2^T * dL/dz2                      [linear layer]
    //    dL/dz1  = dL/da1 . (1 - a1^2)                [tanh activation]
    //    dL/dW1  = dL/dz1 * h^T                       [linear layer]
    //    dL/db1  = dL/dz1                              [bias]
    //    dL/dh   = W1^T * dL/dz1                      [linear layer]
    //    dL/de_t = (1/N) * dL/dh                      [mean pooling]
    //
    // ═══════════════════════════════════════════════════════════════

    pub fn backward(&mut self, target: &[f64; NUM_FEATURES]) -> f64 {
        let len = self.char_indices.len();
        let inv_d = 1.0 / NUM_FEATURES as f64;
        let inv_n = 1.0 / len as f64;

        // ── dL/dz2 = (y_hat - y) / D ──────────────────────────────
        // Local derivative of BCE w.r.t. z2 (through sigmoid):
        //   dL/dy_hat = -(y/y_hat) + (1-y)/(1-y_hat)
        //   dy_hat/dz2 = y_hat*(1-y_hat)
        //   dL/dz2 = dL/dy_hat * dy_hat/dz2 = (y_hat - y) / D
        let mut dl_dz2 = vec![0.0; NUM_FEATURES];
        for i in 0..NUM_FEATURES {
            dl_dz2[i] = (self.y_pred[i] - target[i]) * inv_d;
        }

        // ── dL/dW2[i][j] = dL/dz2[i] * a1[j] ─────────────────────
        // z2[i] = sum_j W2[i][j]*a1[j] + b2[i]
        // Local derivative: dz2[i]/dW2[i][j] = a1[j]
        let mut grad_w2 = vec![vec![0.0; self.hidden_dim]; NUM_FEATURES];
        for i in 0..NUM_FEATURES {
            for j in 0..self.hidden_dim {
                grad_w2[i][j] = dl_dz2[i] * self.a1[j];
            }
        }

        // ── dL/db2[i] = dL/dz2[i] ─────────────────────────────────
        // dz2[i]/db2[i] = 1
        let grad_b2 = dl_dz2.clone();

        // ── dL/da1[j] = sum_i W2[i][j] * dL/dz2[i] ───────────────
        // Branching point: a1 feeds all output dimensions.
        // gradient = sum of all incoming contributions
        let mut dl_da1 = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut sum = 0.0;
            for i in 0..NUM_FEATURES {
                sum += self.w2[i][j] * dl_dz2[i];
            }
            dl_da1[j] = sum;
        }

        // ── dL/dz1[j] = dL/da1[j] * (1 - a1[j]^2) ───────────────
        // Local derivative of tanh: d(tanh(z))/dz = 1 - tanh(z)^2
        let mut dl_dz1 = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim {
            dl_dz1[j] = dl_da1[j] * (1.0 - self.a1[j] * self.a1[j]);
        }

        // ── dL/dW1[i][j] = dL/dz1[i] * h[j] ──────────────────────
        let mut grad_w1 = vec![vec![0.0; self.embed_dim]; self.hidden_dim];
        for i in 0..self.hidden_dim {
            for j in 0..self.embed_dim {
                grad_w1[i][j] = dl_dz1[i] * self.pooled[j];
            }
        }

        // ── dL/db1[i] = dL/dz1[i] ─────────────────────────────────
        let grad_b1 = dl_dz1.clone();

        // ── dL/dh[j] = sum_i W1[i][j] * dL/dz1[i] ────────────────
        // Another branching point: h feeds all hidden dimensions.
        let mut dl_dh = vec![0.0; self.embed_dim];
        for j in 0..self.embed_dim {
            let mut sum = 0.0;
            for i in 0..self.hidden_dim {
                sum += self.w1[i][j] * dl_dz1[i];
            }
            dl_dh[j] = sum;
        }

        // ── dL/de_t[d] = (1/N) * dL/dh[d] ─────────────────────────
        // Mean pooling: h[d] = (1/N) * sum_t e_t[d]
        // Local derivative: dh[d]/de_t[d] = 1/N for each t

        // Update embeddings directly (accumulate gradients per character)
        for t in 0..len {
            let c = self.char_indices[t];
            for d in 0..self.embed_dim {
                let grad = inv_n * dl_dh[d];
                self.embeddings[c][d] -= self.learning_rate * grad;
            }
        }

        // Update W1, b1
        for i in 0..self.hidden_dim {
            self.b1[i] -= self.learning_rate * grad_b1[i];
            for j in 0..self.embed_dim {
                self.w1[i][j] -= self.learning_rate * grad_w1[i][j];
            }
        }

        // Update W2, b2
        for i in 0..NUM_FEATURES {
            self.b2[i] -= self.learning_rate * grad_b2[i];
            for j in 0..self.hidden_dim {
                self.w2[i][j] -= self.learning_rate * grad_w2[i][j];
            }
        }

        // Compute gradient norm (for monitoring convergence)
        let mut grad_norm_sq = 0.0;
        for row in &grad_w1 { for &v in row { grad_norm_sq += v * v; } }
        for &v in &grad_b1 { grad_norm_sq += v * v; }
        for row in &grad_w2 { for &v in row { grad_norm_sq += v * v; } }
        for &v in &grad_b2 { grad_norm_sq += v * v; }

        grad_norm_sq.sqrt()
    }
}

// ─── Activation functions ──────────────────────────────────────────

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ─── Minimal PRNG (no dependencies) ───────────────────────────────
// SplitMix64 — fast, good quality, zero dependencies

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller transform: standard normal from uniform
    fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.next_f64(); // avoid log(0)
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
