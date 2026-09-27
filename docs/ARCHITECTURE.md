# Architecture Deep-Dive

## Design Principles

1. **No frameworks.** Every mathematical operation is implemented from primitives. No PyTorch, no NumPy, no autodiff.

2. **Three-layer separation.** Observation (characters), Learning (differentiable computation), Symbolic (Boolean grammar) are strictly separated.

3. **Identical math across runtimes.** Rust, C#, F#, and Python all compute the same forward/backward pass. Differences are only in language and performance.

4. **Variable-length input.** The system handles words of any length through mean pooling over character embeddings.

5. **Manual chain rule.** Every derivative is written out explicitly. At branching points, gradients are summed over all incoming paths.

## Why Mean Pooling?

Mean pooling converts a variable-length sequence of character embeddings into a fixed-size vector by averaging. This is the simplest sequence-to-fixed-size mapping. It loses positional information but is fast and differentiable.

For morphology, where suffixes carry most of the grammatical information, an Elman RNN would be architecturally superior. The Python float implementation uses an RNN. The compiled implementations use mean pooling for simplicity and speed.

## Why 23 Features?

The 23 Boolean features encode:
- 3 person values (1st, 2nd, 3rd)
- 2 number values (singular, plural)
- 4 tense values (present, imperfect, future, perfect)
- 3 mood values (indicative, subjunctive, imperative)
- 2 voice values (active, passive)
- 3 conjugation classes (1st, 2nd, 3rd)

Each feature is independently predicted. The network learns that certain features are mutually exclusive (e.g., PERSON_1 and PERSON_2 cannot both be true) through the training signal, not through architectural constraints.

## Weight Initialization

Xavier/Glorot initialization: `scale = sqrt(2 / (fan_in + fan_out))`. This keeps the variance of activations and gradients stable across layers at initialization, preventing vanishing or exploding signals in the early epochs.

## Numerical Stability

The sigmoid function uses a numerically stable implementation:
- For x ≥ 0: σ(x) = 1 / (1 + exp(-x))
- For x < 0: σ(x) = exp(x) / (1 + exp(x))

This prevents overflow in exp() for large positive or negative inputs.

Binary cross-entropy clamps predictions to [1e-12, 1-1e-12] before taking logarithms, preventing log(0).
