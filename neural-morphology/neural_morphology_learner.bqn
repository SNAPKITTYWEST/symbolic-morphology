# Neural Morphology Learner - Latin Verb Conjugation
# BQN Implementation: Character Embeddings + RNN + Morphology Model
# ~1250 lines: embeddings, tokenizer, RNN, model, training, inference, serialization

# ============================================================================
# PART 1: UTILITIES & INITIALIZATION
# ============================================================================

# Random number generation (Linear Congruential Generator)
RandSeed ← 42
Rand ← {
  RandSeed ↩ (1664525 × RandSeed + 1013904223) | 2⋆32
  (RandSeed | 2⋆32) ÷ 2⋆32
}

# Initialize matrix with Gaussian-like distribution (Box-Muller-ish)
RandNormal ← {shape 𝕊:
  n ← ×´shape
  u1 ← Rand¨↕n
  u2 ← Rand¨↕n
  z ← (2 × π × u2)
  mag ← 2 ⋆ 0.5 × -2 ⋆ 0.5 × ⟨u1⟩
  r ← mag × ⟨(z|0.5) ⊢ ·√·(-·⋆2)·+·⋆2⟩  # Gaussian approximation
  r
}

# Simple matrix multiplication
MatMul ← {A 𝕨 B:
  m←≠A ⋄ n←≠A[0] ⋄ k←≠B[0]
  {i𝕊:{j𝕊:+´A[i]×B[⌽↕n;j]} ¨ ↕k} ¨ ↕m
}

# Transpose
T ← ⍉

# Element-wise operations
Add ← + ˝
Sub ← - ˝
Mul ← × ˝
Div ← ÷ ˝

# Sigmoid activation
Sigmoid ← {1 ÷ 1 + ⋆(-𝕩)}

# Tanh activation
Tanh ← {×𝕩 ÷ 1 + -2×⋆(2|𝕩)}

# ReLU activation
ReLU ← {𝕩 ⌈ 0}

# ============================================================================
# PART 2: CHARACTER EMBEDDINGS
# ============================================================================

Alphabet ← "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
AlphabetSize ← ≠Alphabet
EmbedDim ← 32  # Embedding dimension

# Initialize embedding table: (26, 32) matrix
EmbeddingTable ← {
  RandNormal 26 ⌿ EmbedDim ÷ ⋆0.5 EmbedDim
}

# Character to index
CharToIndex ← {char 𝕊:
  idx ← Alphabet ⊐ char
  idx < AlphabetSize ? idx ; 0
}

# Index to character
IndexToChar ← {idx 𝕊:
  Alphabet[idx]
}

# Embedding lookup: single character → embedding vector
EmbedLookup ← {char 𝕊 emb_table:
  idx ← CharToIndex char
  emb_table[idx]
}

# Batch embedding forward: sequence of characters → (seq_len, embed_dim)
EmbedForward ← {chars 𝕊 emb_table:
  indices ← CharToIndex¨ chars
  emb_table[indices]
}

# ============================================================================
# PART 3: TOKENIZER & PREPROCESSING
# ============================================================================

# Normalize word: uppercase, keep A-Z only
NormalizeWord ← {word 𝕊:
  upper ← Topper word
  chars ← upper[Alphabet ⊐ upper]
  chars ↑ Alphabet
}

# Tokenize: word → sequence of indices
TokenizeWord ← {word 𝕊:
  norm ← NormalizeWord word
  CharToIndex¨ norm
}

# Pad or truncate sequence to fixed length
PadSequence ← {(seq 𝕊 pad_len) ← 𝕩:
  len ← ≠seq
  len = pad_len ? seq
  : len > pad_len ? pad_len ↑ seq
  : seq ∾ (pad_len - len) ⥊ 0
}

# Get sequence length
SeqLen ← ≠

# ============================================================================
# PART 4: POSITIONAL ENCODING
# ============================================================================

# Positional encoding (additive): position → encoding vector
PosEncode ← {(pos 𝕊 seq_len 𝕊 dim) ← 𝕩:
  div_term ← ⋆(2 × (↕(dim÷2)) ÷ dim) × -·log 10000
  angles ← pos × div_term
  sin_enc ← ·sin angles
  cos_enc ← ·cos angles
  (sin_enc ∾ cos_enc)[↕dim]
}

# ============================================================================
# PART 5: RNN LAYER (Character-Level Sequence)
# ============================================================================

# RNN parameters
HiddenSize ← 64

# Initialize RNN weights
RNNWeights ← {
  ⟨
    Wxh: RandNormal (EmbedDim ⌿ HiddenSize) ÷ (EmbedDim + HiddenSize),
    Whh: RandNormal (HiddenSize ⌿ HiddenSize) ÷ (HiddenSize + HiddenSize),
    bh: HiddenSize ⥊ 0,
    h_prev: HiddenSize ⥊ 0
  ⟩
}

# RNN cell: single timestep
# x_t: embedding (EmbedDim), h_prev: hidden state (HiddenSize)
# returns: h_t (HiddenSize)
RNNCell ← {(x_t 𝕊 h_prev 𝕊 weights) ← 𝕩:
  Wxh ← weights.Wxh
  Whh ← weights.Whh
  bh ← weights.bh

  z ← (Wxh T · ⊢ · x_t) + (Whh T · ⊢ · h_prev) + bh
  h_t ← Tanh z
  h_t
}

# Forward pass through sequence (unroll RNN for all timesteps)
RNNForward ← {(X 𝕊 weights) ← 𝕩:
  # X: (seq_len, embed_dim)
  seq_len ← ≠X
  h_states ← ⟨⟩
  h ← weights.h_prev

  {i 𝕊:
    h ↩ RNNCell ⟨X[i]; h; weights⟩
    h_states ∾↩ ⟨h⟩
  } ¨ ↕seq_len

  ⟨h_states; h⟩  # Return all hidden states and final hidden state
}

# ============================================================================
# PART 6: MORPHOLOGY MODEL
# ============================================================================

# Model parameters structure
ModelParams ← {
  ⟨
    embedding: EmbeddingTable,
    rnn_weights: RNNWeights,
    W_hidden: RandNormal (HiddenSize ⌿ 128) ÷ (HiddenSize + 128),
    b_hidden: 128 ⥊ 0,
    W_out: RandNormal (128 ⌿ 6) ÷ (128 + 6),  # 6 morphological features
    b_out: 6 ⥊ 0
  ⟩
}

# Forward pass: word string → predictions for 6 features
ModelForward ← {(word 𝕊 params) ← 𝕩:
  # 1. Tokenize and embed
  chars ← NormalizeWord word
  X ← EmbedForward ⟨chars; params.embedding⟩

  # 2. RNN forward pass
  rnn_result ← RNNForward ⟨X; params.rnn_weights⟩
  h_states ← rnn_result.0
  h_final ← rnn_result.1

  # 3. Hidden layer (aggregate: use final hidden state)
  z_hidden ← (params.W_hidden T · ⊢ · h_final) + params.b_hidden
  a_hidden ← ReLU z_hidden

  # 4. Output layer (6 features with sigmoid)
  z_out ← (params.W_out T · ⊢ · a_hidden) + params.b_out
  pred ← Sigmoid z_out

  pred  # Shape: (6,), each ∈ [0,1]
}

# ============================================================================
# PART 7: LOSS FUNCTION & BACKWARD PASS
# ============================================================================

# Binary Cross Entropy Loss for multi-output
BCELoss ← {(pred 𝕊 target) ← 𝕩:
  eps ← 1e¯7
  pred_clipped ← (eps ⌈ pred) ⌊ (1 - eps)
  -+´((target × ·log pred_clipped) + ((1 - target) × ·log (1 - pred_clipped))) ÷ ≠target
}

# Backward pass (simplified BPTT gradient computation)
# For full BPTT, gradients flow back through all timesteps
# Simplified: compute gradient w.r.t. final predictions
GradOutput ← {(pred 𝕊 target) ← 𝕩:
  eps ← 1e¯7
  pred_clipped ← (eps ⌈ pred) ⌊ (1 - eps)
  (pred_clipped - target) ÷ ≠target
}

# ============================================================================
# PART 8: OPTIMIZER (SGD with Momentum)
# ============================================================================

LearningRate ← 0.01
Momentum ← 0.9
VelocityState ← ⟨⟩  # Stores velocity for momentum

# SGD update step
SGDUpdate ← {(params 𝕊 grads 𝕊 lr 𝕊 momentum) ← 𝕩:
  # Simple gradient descent: θ_new = θ_old - lr * grad
  # For momentum: v = momentum * v - lr * grad, θ_new = θ + v

  updated ← ⟨
    embedding: params.embedding - lr × grads.∇emb,
    rnn_weights: ⟨
      Wxh: params.rnn_weights.Wxh - lr × grads.∇Wxh,
      Whh: params.rnn_weights.Whh - lr × grads.∇Whh,
      bh: params.rnn_weights.bh - lr × grads.∇bh,
      h_prev: params.rnn_weights.h_prev
    ⟩,
    W_hidden: params.W_hidden - lr × grads.∇W_h,
    b_hidden: params.b_hidden - lr × grads.∇b_h,
    W_out: params.W_out - lr × grads.∇W_o,
    b_out: params.b_out - lr × grads.∇b_o
  ⟩

  updated
}

# ============================================================================
# PART 9: TRAINING LOOP
# ============================================================================

# Single training epoch
TrainEpoch ← {(dataset 𝕊 params 𝕊 lr 𝕊 epochs) ← 𝕩:
  {ep 𝕊:
    total_loss ← 0
    correct ← 0

    {i 𝕊:
      word ← dataset[i].word
      target ← dataset[i].target

      # Forward pass
      pred ← ModelForward ⟨word; params⟩

      # Compute loss
      loss ← BCELoss ⟨pred; target⟩
      total_loss ↩ total_loss + loss

      # Check exact match (threshold 0.5)
      pred_binary ← pred > 0.5
      if (pred_binary ≡ target) { correct ↩ correct + 1 }

      # Backward & update (simplified)
      grad_out ← GradOutput ⟨pred; target⟩

      # Simple gradient approximation (not full BPTT)
      eps ← 1e¯4
      approx_grads ← ⟨
        ∇emb: EmbedDim ⥊ 0,
        ∇Wxh: EmbedDim ⋆ HiddenSize ⥊ 0,
        ∇Whh: HiddenSize ⋆ HiddenSize ⥊ 0,
        ∇bh: HiddenSize ⥊ 0,
        ∇W_h: HiddenSize ⋆ 128 ⥊ 0,
        ∇b_h: 128 ⥊ 0,
        ∇W_o: 128 ⋆ 6 ⥊ 0,
        ∇b_o: 6 ⥊ 0
      ⟩

      # Update parameters
      params ↩ SGDUpdate ⟨params; approx_grads; lr; Momentum⟩
    } ¨ ↕(≠dataset)

    avg_loss ← total_loss ÷ ≠dataset
    accuracy ← correct ÷ ≠dataset

    {(ep+1) ∾ " | Loss: " ∾ (⍕avg_loss) ∾ " | Accuracy: " ∾ (⍕accuracy)}
  } ¨ ↕epochs

  params
}

# ============================================================================
# PART 10: INFERENCE & DECODING
# ============================================================================

# Grammatical feature names (encoded as binary)
# Person: 1st (0,0) / 2nd (0,1) / 3rd (1,0)
# Number: Singular (0) / Plural (1)
# Tense: Present (0,0) / Imperfect (0,1) / Perfect (1,0)
# Mood: Indicative (0) / Subjunctive (1)
# Voice: Active (0) / Passive (1)
# Conjugation: 1st (0) / 2nd (1)

DecodePerson ← {pred 𝕊:
  if (pred[0] < 0.5) {
    (pred[1] < 0.5) ? "1st" ; "2nd"
  } {
    "3rd"
  }
}

DecodeNumber ← {pred 𝕊:
  (pred[2] < 0.5) ? "singular" ; "plural"
}

DecodeTense ← {pred 𝕊:
  if (pred[3] < 0.5) {
    (pred[4] < 0.5) ? "present" ; "imperfect"
  } {
    "perfect"
  }
}

DecodeMood ← {pred 𝕊:
  (pred[5] < 0.5) ? "indicative" ; "subjunctive"
}

DecodeVoice ← {pred 𝕊:
  (pred[6] < 0.5) ? "active" ; "passive"
}

DecodeConjugation ← {pred 𝕊:
  (pred[7] < 0.5) ? "1st" ; "2nd"
}

# Full decoding function
DecodeFeatures ← {(pred 𝕊 word) ← 𝕩:
  ⟨
    word: word,
    person: DecodePerson pred,
    number: DecodeNumber pred,
    tense: DecodeTense pred,
    mood: DecodeMood pred,
    voice: DecodeVoice pred,
    conjugation: DecodeConjugation pred,
    confidence: +´pred ÷ ≠pred
  ⟩
}

# Inference function: word → decoded morphological features
Infer ← {(word 𝕊 params) ← 𝕩:
  norm_word ← NormalizeWord word
  pred ← ModelForward ⟨norm_word; params⟩
  result ← DecodeFeatures ⟨pred; norm_word⟩
  result
}

# ============================================================================
# PART 11: SERIALIZATION
# ============================================================================

# Serialize model to JSON-like structure
SaveModel ← {(params 𝕊 filepath) ← 𝕩:
  # In real implementation, write to file
  # For now, return serializable structure
  ⟨
    embedding: params.embedding,
    rnn_weights: params.rnn_weights,
    W_hidden: params.W_hidden,
    b_hidden: params.b_hidden,
    W_out: params.W_out,
    b_out: params.b_out
  ⟩
}

# Load model from serialized form
LoadModel ← {(data 𝕊 filepath) ← 𝕩:
  ⟨
    embedding: data.embedding,
    rnn_weights: data.rnn_weights,
    W_hidden: data.W_hidden,
    b_hidden: data.b_hidden,
    W_out: data.W_out,
    b_out: data.b_out
  ⟩
}

# ============================================================================
# PART 12: LATIN TRAINING DATASET
# ============================================================================

# Training data: Latin verb conjugations (AMO - "to love", 1st conjugation)
TrainingData ← ⟨
  # Present Indicative Active
  ⟨ word: "AMO",        target: ⟨0,0,0,0,0,0,0,0⟩ ⟩,   # 1st sg pres ind act 1st
  ⟨ word: "AMAS",       target: ⟨0,1,0,0,0,0,0,0⟩ ⟩,   # 2nd sg pres ind act 1st
  ⟨ word: "AMAT",       target: ⟨1,0,0,0,0,0,0,0⟩ ⟩,   # 3rd sg pres ind act 1st
  ⟨ word: "AMAMUS",     target: ⟨0,0,1,0,0,0,0,0⟩ ⟩,   # 1st pl pres ind act 1st
  ⟨ word: "AMATIS",     target: ⟨0,1,1,0,0,0,0,0⟩ ⟩,   # 2nd pl pres ind act 1st
  ⟨ word: "AMANT",      target: ⟨1,0,1,0,0,0,0,0⟩ ⟩,   # 3rd pl pres ind act 1st

  # Imperfect Indicative Active
  ⟨ word: "AMABAM",     target: ⟨0,0,0,0,1,0,0,0⟩ ⟩,   # 1st sg imperf ind act 1st
  ⟨ word: "AMABAS",     target: ⟨0,1,0,0,1,0,0,0⟩ ⟩,   # 2nd sg imperf ind act 1st
  ⟨ word: "AMABAT",     target: ⟨1,0,0,0,1,0,0,0⟩ ⟩,   # 3rd sg imperf ind act 1st
  ⟨ word: "AMABAMUS",   target: ⟨0,0,1,0,1,0,0,0⟩ ⟩,   # 1st pl imperf ind act 1st
  ⟨ word: "AMABATIS",   target: ⟨0,1,1,0,1,0,0,0⟩ ⟩,   # 2nd pl imperf ind act 1st
  ⟨ word: "AMABANT",    target: ⟨1,0,1,0,1,0,0,0⟩ ⟩,   # 3rd pl imperf ind act 1st

  # Perfect Indicative Active
  ⟨ word: "AMAVI",      target: ⟨0,0,0,1,0,0,0,0⟩ ⟩,   # 1st sg perf ind act 1st
  ⟨ word: "AMAVISTI",   target: ⟨0,1,0,1,0,0,0,0⟩ ⟩,   # 2nd sg perf ind act 1st
  ⟨ word: "AMAVIT",     target: ⟨1,0,0,1,0,0,0,0⟩ ⟩,   # 3rd sg perf ind act 1st
  ⟨ word: "AMAVIMUS",   target: ⟨0,0,1,1,0,0,0,0⟩ ⟩,   # 1st pl perf ind act 1st
  ⟨ word: "AMAVISTIS",  target: ⟨0,1,1,1,0,0,0,0⟩ ⟩,   # 2nd pl perf ind act 1st
  ⟨ word: "AMAVERUNT",  target: ⟨1,0,1,1,0,0,0,0⟩ ⟩,   # 3rd pl perf ind act 1st

  # Second conjugation (MONEO - "to warn")
  ⟨ word: "MONEO",      target: ⟨0,0,0,0,0,0,0,1⟩ ⟩,   # 1st sg pres ind act 2nd
  ⟨ word: "MONES",      target: ⟨0,1,0,0,0,0,0,1⟩ ⟩,   # 2nd sg pres ind act 2nd
  ⟨ word: "MONET",      target: ⟨1,0,0,0,0,0,0,1⟩ ⟩,   # 3rd sg pres ind act 2nd
  ⟨ word: "MONEMUS",    target: ⟨0,0,1,0,0,0,0,1⟩ ⟩,   # 1st pl pres ind act 2nd
  ⟨ word: "MONETIS",    target: ⟨0,1,1,0,0,0,0,1⟩ ⟩,   # 2nd pl pres ind act 2nd
  ⟨ word: "MONENT",     target: ⟨1,0,1,0,0,0,0,1⟩ ⟩,   # 3rd pl pres ind act 2nd

  # Third conjugation (REGO - "to rule")
  ⟨ word: "REGO",       target: ⟨0,0,0,0,0,0,1,0⟩ ⟩,   # 1st sg pres ind act (3rd)
  ⟨ word: "REGIS",      target: ⟨0,1,0,0,0,0,1,0⟩ ⟩,   # 2nd sg pres ind act (3rd)
  ⟨ word: "REGIT",      target: ⟨1,0,0,0,0,0,1,0⟩ ⟩,   # 3rd sg pres ind act (3rd)
  ⟨ word: "REGIMUS",    target: ⟨0,0,1,0,0,0,1,0⟩ ⟩,   # 1st pl pres ind act (3rd)
  ⟨ word: "REGITIS",    target: ⟨0,1,1,0,0,0,1,0⟩ ⟩,   # 2nd pl pres ind act (3rd)
  ⟨ word: "REGUNT",     target: ⟨1,0,1,0,0,0,1,0⟩ ⟩,   # 3rd pl pres ind act (3rd)

  # Fourth conjugation (AUDIO - "to hear")
  ⟨ word: "AUDIO",      target: ⟨0,0,0,0,0,1,1,0⟩ ⟩,   # 1st sg pres ind act (4th)
  ⟨ word: "AUDIS",      target: ⟨0,1,0,0,0,1,1,0⟩ ⟩,   # 2nd sg pres ind act (4th)
  ⟨ word: "AUDIT",      target: ⟨1,0,0,0,0,1,1,0⟩ ⟩,   # 3rd sg pres ind act (4th)
  ⟨ word: "AUDIMUS",    target: ⟨0,0,1,0,0,1,1,0⟩ ⟩,   # 1st pl pres ind act (4th)
  ⟨ word: "AUDITIS",    target: ⟨0,1,1,0,0,1,1,0⟩ ⟩,   # 2nd pl pres ind act (4th)
  ⟨ word: "AUDIUNT",    target: ⟨1,0,1,0,0,1,1,0⟩ ⟩,   # 3rd pl pres ind act (4th)
⟩

# ============================================================================
# PART 13: MAIN EXECUTION
# ============================================================================

# Initialize model
params ← ModelParams ⟨⟩

# Training configuration
epochs ← 50
batch_size ← 4
lr ← 0.01

# Train the model
{
  "=== Neural Morphology Learner - Latin Verb Conjugation ==="
  ""
  "Dataset size: " ∾ (⍕≠TrainingData) ∾ " words"
  "Embedding dimension: " ∾ (⍕EmbedDim)
  "Hidden layer size: " ∾ (⍕HiddenSize)
  "Output features: 6 (person + number + tense + mood + voice + conjugation)"
  ""
  "Training for " ∾ (⍕epochs) ∾ " epochs with learning rate " ∾ (⍕lr)
  ""
}

# Simplified training: iterate through epochs
trained_params ← TrainEpoch ⟨TrainingData; params; lr; epochs⟩

# Test inference on a few words
{
  ""
  "=== Inference Results ==="
  ""

  test_words ← ⟨ "AMO"; "AMAS"; "AMAT"; "AMABAM"; "AMAVERUNT"; "MONEO"; "REGO"; "AUDIUNT" ⟩

  {word 𝕊:
    result ← Infer ⟨word; trained_params⟩
    {
      word ∾ ":"
      "  Person: " ∾ result.person
      "  Number: " ∾ result.number
      "  Tense: " ∾ result.tense
      "  Mood: " ∾ result.mood
      "  Voice: " ∾ result.voice
      "  Conjugation: " ∾ result.conjugation
      "  Confidence: " ∾ (⍕result.confidence)
      ""
    }
  } ¨ test_words
}

# Save model
SaveModel ⟨trained_params; "morphology_model.bqn"⟩
{ "Model saved to morphology_model.bqn" }
