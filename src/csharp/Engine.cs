/*
 * Symbolic Morphology Engine — Latin verb morphology from raw letters to Boolean grammar
 * Copyright (C) 2026 Ahmad Ali Parr
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

namespace Sovereign.Engine;

/// <summary>
/// LEARNING LAYER — Differentiable computation graph.
///
/// Architecture:
///   1. Character embedding table (learnable)
///   2. Mean pooling over character positions (handles variable-length input)
///   3. Hidden layer: z1 = W1·h + b1,  a1 = tanh(z1)
///   4. Output layer: z2 = W2·a1 + b2,  ŷ = sigmoid(z2)
///   5. Binary cross-entropy loss
///   6. Manual chain-rule backpropagation
///   7. Gradient descent parameter update
///
/// Every derivative is computed by hand. No autodiff framework.
/// </summary>
public sealed class Engine
{
    // ─── Hyperparameters ───────────────────────────────────────────
    public readonly int EmbedDim;
    public readonly int HiddenDim;
    public readonly int OutputDim;
    public readonly double LearningRate;

    // ─── Learnable parameters ──────────────────────────────────────
    // Embedding table: maps each uppercase letter to a vector
    public readonly double[,] Embeddings;   // [128, embedDim]  (ASCII-indexed)

    // Hidden layer: z1 = W1·h + b1
    public readonly double[,] W1;           // [hiddenDim, embedDim]
    public readonly double[] B1;            // [hiddenDim]

    // Output layer: z2 = W2·a1 + b2
    public readonly double[,] W2;           // [outputDim, hiddenDim]
    public readonly double[] B2;            // [outputDim]

    // ─── Cached forward state (for one sample) ─────────────────────
    double[]? _charIndices;
    double[,]? _embVecs;
    double[]? _pooled;
    double[]? _z1;
    double[]? _a1;
    double[]? _z2;
    double[]? _yPred;

    readonly Random _rng;

    public Engine(int embedDim, int hiddenDim, int outputDim, double learningRate, int seed = 42)
    {
        EmbedDim = embedDim;
        HiddenDim = hiddenDim;
        OutputDim = outputDim;
        LearningRate = learningRate;
        _rng = new Random(seed);

        Embeddings = new double[128, embedDim];
        W1 = new double[hiddenDim, embedDim];
        B1 = new double[hiddenDim];
        W2 = new double[outputDim, hiddenDim];
        B2 = new double[outputDim];

        InitializeParameters();
    }

    // ─── Xavier/Glorot initialization ──────────────────────────────
    // Scale = sqrt(2 / (fan_in + fan_out)) for tanh, sqrt(1/fan_in) for sigmoid
    void InitializeParameters()
    {
        double embScale = Math.Sqrt(2.0 / (1 + EmbedDim));
        for (int c = 0; c < 128; c++)
            for (int d = 0; d < EmbedDim; d++)
                Embeddings[c, d] = SampleNormal() * embScale;

        double w1Scale = Math.Sqrt(2.0 / (EmbedDim + HiddenDim));
        for (int i = 0; i < HiddenDim; i++)
            for (int j = 0; j < EmbedDim; j++)
                W1[i, j] = SampleNormal() * w1Scale;

        double w2Scale = Math.Sqrt(2.0 / (HiddenDim + OutputDim));
        for (int i = 0; i < OutputDim; i++)
            for (int j = 0; j < HiddenDim; j++)
                W2[i, j] = SampleNormal() * w2Scale;
    }

    double SampleNormal()
    {
        // Box-Muller transform: generate standard normal from uniform
        double u1 = 1.0 - _rng.NextDouble(); // avoid log(0)
        double u2 = _rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    // ═══════════════════════════════════════════════════════════════
    //  FORWARD PASS
    // ═══════════════════════════════════════════════════════════════
    //
    //  For input word "AMO" (characters A, M, O):
    //
    //  Step 1 — Embedding lookup:
    //    e_A = Embeddings['A'],  e_M = Embeddings['M'],  e_O = Embeddings['O']
    //
    //  Step 2 — Mean pooling (variable-length → fixed-size):
    //    h = (e_A + e_M + e_O) / 3
    //
    //  Step 3 — Hidden layer:
    //    z1 = W1·h + b1
    //    a1 = tanh(z1)
    //
    //  Step 4 — Output layer:
    //    z2 = W2·a1 + b2
    //    ŷ = sigmoid(z2)
    //
    // ═══════════════════════════════════════════════════════════════

    public double[] Forward(string word)
    {
        int len = word.Length;
        _charIndices = new double[len];
        _embVecs = new double[len, EmbedDim];

        // Step 1: Character embedding lookup
        for (int t = 0; t < len; t++)
        {
            int c = char.ToUpperInvariant(word[t]);
            _charIndices[t] = c;
            for (int d = 0; d < EmbedDim; d++)
                _embVecs[t, d] = Embeddings[c, d];
        }

        // Step 2: Mean pooling — h = (1/N) Σ e_t
        _pooled = new double[EmbedDim];
        for (int d = 0; d < EmbedDim; d++)
        {
            double sum = 0;
            for (int t = 0; t < len; t++)
                sum += _embVecs[t, d];
            _pooled[d] = sum / len;
        }

        // Step 3: Hidden layer — z1 = W1·h + b1,  a1 = tanh(z1)
        _z1 = new double[HiddenDim];
        _a1 = new double[HiddenDim];
        for (int i = 0; i < HiddenDim; i++)
        {
            double z = B1[i];
            for (int j = 0; j < EmbedDim; j++)
                z += W1[i, j] * _pooled[j];
            _z1[i] = z;
            _a1[i] = Math.Tanh(z);
        }

        // Step 4: Output layer — z2 = W2·a1 + b2,  ŷ = sigmoid(z2)
        _z2 = new double[OutputDim];
        _yPred = new double[OutputDim];
        for (int i = 0; i < OutputDim; i++)
        {
            double z = B2[i];
            for (int j = 0; j < HiddenDim; j++)
                z += W2[i, j] * _a1[j];
            _z2[i] = z;
            _yPred[i] = Sigmoid(z);
        }

        return _yPred;
    }

    // ═══════════════════════════════════════════════════════════════
    //  LOSS — Binary Cross-Entropy
    // ═══════════════════════════════════════════════════════════════
    //
    //  L = -(1/D) Σ_i [ y_i·log(ŷ_i) + (1-y_i)·log(1-ŷ_i) ]
    //
    //  where D = number of output features (23)
    //
    // ═══════════════════════════════════════════════════════════════

    public double ComputeLoss(double[] target)
    {
        double loss = 0;
        const double eps = 1e-12;
        for (int i = 0; i < OutputDim; i++)
        {
            double y = Math.Clamp(_yPred![i], eps, 1.0 - eps);
            loss -= target[i] * Math.Log(y) + (1.0 - target[i]) * Math.Log(1.0 - y);
        }
        return loss / OutputDim;
    }

    // ═══════════════════════════════════════════════════════════════
    //  BACKWARD PASS — Classical Chain Rule
    // ═══════════════════════════════════════════════════════════════
    //
    //  For every operation in the forward graph, we compute the local
    //  derivative and propagate gradients backward:
    //
    //    gradient_in = gradient_out × local_derivative
    //
    //  At branching points (where a value feeds multiple paths):
    //
    //    gradient = sum(all incoming gradient contributions)
    //
    //  The full chain for this network:
    //
    //    ∂L/∂z2  = (ŷ - y) / D                    [BCE + sigmoid]
    //    ∂L/∂W2  = (∂L/∂z2) · a1ᵀ                 [linear layer]
    //    ∂L/∂b2  = ∂L/∂z2                          [bias]
    //    ∂L/∂a1  = W2ᵀ · (∂L/∂z2)                 [linear layer]
    //    ∂L/∂z1  = (∂L/∂a1) ⊙ (1 - a1²)           [tanh activation]
    //    ∂L/∂W1  = (∂L/∂z1) · hᵀ                  [linear layer]
    //    ∂L/∂b1  = ∂L/∂z1                          [bias]
    //    ∂L/∂h   = W1ᵀ · (∂L/∂z1)                 [linear layer]
    //    ∂L/∂e_t = (1/N) · ∂L/∂h                  [mean pooling]
    //
    // ═══════════════════════════════════════════════════════════════

    public (double[] GradW1, double[] GradB1, double[] GradW2, double[] GradB2, double GradNorm) Backward(double[] target)
    {
        int len = _charIndices!.Length;
        double invD = 1.0 / OutputDim;
        double invN = 1.0 / len;

        // ── dL/dz2 = (ŷ - y) / D ──────────────────────────────────
        // Local derivative of BCE w.r.t. z2 (through sigmoid):
        //   dL/dŷ = -(y/ŷ) + (1-y)/(1-ŷ)
        //   dŷ/dz2 = ŷ(1-ŷ)
        //   dL/dz2 = dL/dŷ × dŷ/dz2 = (ŷ - y) / D
        var dL_dz2 = new double[OutputDim];
        for (int i = 0; i < OutputDim; i++)
            dL_dz2[i] = (_yPred![i] - target[i]) * invD;

        // ── dL/dW2[i,j] = dL/dz2[i] × a1[j] ──────────────────────
        // z2[i] = Σ_j W2[i,j]·a1[j] + b2[i]
        // Local derivative: ∂z2[i]/∂W2[i,j] = a1[j]
        var gradW2 = new double[OutputDim * HiddenDim];
        for (int i = 0; i < OutputDim; i++)
            for (int j = 0; j < HiddenDim; j++)
                gradW2[i * HiddenDim + j] = dL_dz2[i] * _a1![j];

        // ── dL/db2[i] = dL/dz2[i] ─────────────────────────────────
        // ∂z2[i]/∂b2[i] = 1
        var gradB2 = new double[OutputDim];
        Array.Copy(dL_dz2, gradB2, OutputDim);

        // ── dL/da1[j] = Σ_i W2[i,j] × dL/dz2[i] ──────────────────
        // Branching point: a1 feeds all output dimensions.
        // gradient = sum of all incoming contributions
        var dL_da1 = new double[HiddenDim];
        for (int j = 0; j < HiddenDim; j++)
        {
            double sum = 0;
            for (int i = 0; i < OutputDim; i++)
                sum += W2[i, j] * dL_dz2[i];
            dL_da1[j] = sum;
        }

        // ── dL/dz1[j] = dL/da1[j] × (1 - a1[j]²) ────────────────
        // Local derivative of tanh: d(tanh(z))/dz = 1 - tanh(z)²
        var dL_dz1 = new double[HiddenDim];
        for (int j = 0; j < HiddenDim; j++)
            dL_dz1[j] = dL_da1[j] * (1.0 - _a1![j] * _a1[j]);

        // ── dL/dW1[i,j] = dL/dz1[i] × h[j] ───────────────────────
        var gradW1 = new double[HiddenDim * EmbedDim];
        for (int i = 0; i < HiddenDim; i++)
            for (int j = 0; j < EmbedDim; j++)
                gradW1[i * EmbedDim + j] = dL_dz1[i] * _pooled![j];

        // ── dL/db1[i] = dL/dz1[i] ─────────────────────────────────
        var gradB1 = new double[HiddenDim];
        Array.Copy(dL_dz1, gradB1, HiddenDim);

        // ── dL/dh[j] = Σ_i W1[i,j] × dL/dz1[i] ──────────────────
        // Another branching point: h feeds all hidden dimensions.
        var dL_dh = new double[EmbedDim];
        for (int j = 0; j < EmbedDim; j++)
        {
            double sum = 0;
            for (int i = 0; i < HiddenDim; i++)
                sum += W1[i, j] * dL_dz1[i];
            dL_dh[j] = sum;
        }

        // ── dL/de_t[d] = (1/N) × dL/dh[d] ────────────────────────
        // Mean pooling: h[d] = (1/N) Σ_t e_t[d]
        // Local derivative: ∂h[d]/∂e_t[d] = 1/N for each t
        var gradEmbed = new double[len, EmbedDim];
        for (int t = 0; t < len; t++)
            for (int d = 0; d < EmbedDim; d++)
                gradEmbed[t, d] = invN * dL_dh[d];

        // Accumulate embedding gradients into the table
        for (int t = 0; t < len; t++)
        {
            int c = (int)_charIndices![t];
            for (int d = 0; d < EmbedDim; d++)
                Embeddings[c, d] -= LearningRate * gradEmbed[t, d];
        }

        // Compute gradient norm (for monitoring convergence)
        double gradNorm = 0;
        for (int i = 0; i < gradW1.Length; i++) gradNorm += gradW1[i] * gradW1[i];
        for (int i = 0; i < gradB1.Length; i++) gradNorm += gradB1[i] * gradB1[i];
        for (int i = 0; i < gradW2.Length; i++) gradNorm += gradW2[i] * gradW2[i];
        for (int i = 0; i < gradB2.Length; i++) gradNorm += gradB2[i] * gradB2[i];
        gradNorm = Math.Sqrt(gradNorm);

        return (gradW1, gradB1, gradW2, gradB2, gradNorm);
    }

    // ═══════════════════════════════════════════════════════════════
    //  GRADIENT DESCENT STEP
    // ═══════════════════════════════════════════════════════════════
    //
    //  θ := θ - η × ∇θL
    //
    //  Embeddings already updated in Backward (for efficiency).
    //  Here we update W1, b1, W2, b2.
    //
    // ═══════════════════════════════════════════════════════════════

    public void GradientDescentStep(double[] gradW1, double[] gradB1, double[] gradW2, double[] gradB2)
    {
        for (int i = 0; i < HiddenDim; i++)
        {
            B1[i] -= LearningRate * gradB1[i];
            for (int j = 0; j < EmbedDim; j++)
                W1[i, j] -= LearningRate * gradW1[i * EmbedDim + j];
        }

        for (int i = 0; i < OutputDim; i++)
        {
            B2[i] -= LearningRate * gradB2[i];
            for (int j = 0; j < HiddenDim; j++)
                W2[i, j] -= LearningRate * gradW2[i * HiddenDim + j];
        }
    }

    // ─── Activation functions ──────────────────────────────────────

    static double Sigmoid(double x)
    {
        // Numerically stable sigmoid
        if (x >= 0)
        {
            double e = Math.Exp(-x);
            return 1.0 / (1.0 + e);
        }
        else
        {
            double e = Math.Exp(x);
            return e / (1.0 + e);
        }
    }
}
