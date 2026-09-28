# ============================================================================
# NEURAL MORPHOLOGY LEARNER - CORE MATHEMATICAL LAYER
# BQN Implementation (Array Language)
# ============================================================================
# Complete mathematical foundation for neural network operations
# Includes: scalar ops, vector ops, matrix ops, activations, loss functions,
# numerical stability checks
# ============================================================================

# ============================================================================
# SCALAR OPERATIONS (Element-wise operations on numbers)
# ============================================================================

Add ← +
Subtract ← -
Multiply ← ×
Divide ← ÷
Negate ← -∘⊢
Absolute ← |
Minimum ← ⌊
Maximum ← ⌈

# Clamp a scalar to range [min, max]
Clamp ← {
  min ← 𝕨.min
  max ← 𝕨.max
  value ← 𝕩
  (min ⌈ value) ⌊ max
}

Power ← ^
Exponential ← {⊢^𝕩}
Logarithm ← {𝕩⍟⊢}
Square ← {𝕩 × 𝕩}
SquareRoot ← {𝕩 ^ 0.5}

# ============================================================================
# VECTOR OPERATIONS (1D array operations)
# ============================================================================

VectorAdd ← +
VectorSubtract ← -

# Scale vector by scalar: scale_factor × vector
VectorScale ← {
  scale ← 𝕨
  vec ← 𝕩
  scale × vec
}

# Dot product: sum of element-wise multiplication
VectorDot ← {
  v1 ← 𝕨
  v2 ← 𝕩
  +´ v1 × v2
}

# Sum all vector elements
VectorSum ← +´

# Mean of vector elements
VectorMean ← {
  vec ← 𝕩
  (+´ vec) ÷ (≠vec)
}

# L2 norm (Euclidean length)
VectorNorm ← {
  vec ← 𝕩
  (+´ vec × vec) ^ 0.5
}

# Create zero vector of length n
VectorZero ← {
  n ← 𝕩
  n ↑ 0
}

# Create vector filled with value v, length n
VectorFill ← {
  v ← 𝕨
  n ← 𝕩
  (n ↑ 0) + v
}

# Element-wise multiplication
ElementWiseMultiply ← ×

# Element-wise division (with safety)
ElementWiseDivide ← {
  a ← 𝕨
  b ← 𝕩
  a ÷ b
}

# Normalize vector to unit length
VectorNormalize ← {
  vec ← 𝕩
  norm ← VectorNorm vec
  vec ÷ norm
}

# ============================================================================
# MATRIX OPERATIONS (2D array operations)
# ============================================================================

# Create m×n zero matrix
MatrixZeros ← {
  dims ← 𝕩
  m ← dims.0
  n ← dims.1
  m ↑ (n ↑ 0)
}

# Create n×n identity matrix
MatrixIdentity ← {
  n ← 𝕩
  =⌜ ↕n
}

# Matrix addition
MatrixAdd ← +

# Matrix subtraction
MatrixSubtract ← -

# Scale matrix by scalar
MatrixScale ← {
  scale ← 𝕨
  mat ← 𝕩
  scale × mat
}

# Matrix transpose
MatrixTranspose ← ⍉

# Matrix multiplication: A × B
MatrixMultiply ← {
  A ← 𝕨
  B ← 𝕩
  # C[i,j] = sum_k A[i,k] * B[k,j]
  # Using outer product and reduction
  +´˘ A ×⌜ ⍉B
}

# Matrix-vector multiplication: A × v
MatrixVectorMultiply ← {
  A ← 𝕨
  v ← 𝕩
  # Result[i] = sum_j A[i,j] * v[j]
  +´˘ A × v
}

# Outer product of two vectors: u ⊗ v
OuterProduct ← {
  u ← 𝕨
  v ← 𝕩
  u ×⌜ v
}

# Sum all matrix elements
MatrixSum ← {
  mat ← 𝕩
  +´ +´˘ mat
}

# Mean of all matrix elements
MatrixMean ← {
  mat ← 𝕩
  total ← +´ +´˘ mat
  count ← (≠mat.0) × (≠mat.1)
  total ÷ count
}

# Matrix row-wise sum
MatrixRowSum ← +´˘

# Matrix column-wise sum
MatrixColumnSum ← +´

# Get dimensions of matrix as [rows, cols]
MatrixShape ← {
  mat ← 𝕩
  ⟨≠mat.0, ≠mat.1⟩
}

# ============================================================================
# ACTIVATION FUNCTIONS (with explicit derivatives)
# ============================================================================

# Sigmoid function: σ(x) = 1 / (1 + exp(-x))
Sigmoid ← {
  x ← 𝕩
  1 ÷ (1 + ⊢^(-x))
}

# Sigmoid derivative: σ'(a) = a(1-a), where a = sigmoid(x)
# Input a is the sigmoid OUTPUT, not x
SigmoidDerivative ← {
  a ← 𝕩
  a × (1 - a)
}

# Sigmoid derivative via chain rule: σ'(x) = σ(x)(1-σ(x))
SigmoidDerivativeViaInput ← {
  x ← 𝕩
  sig ← Sigmoid x
  sig × (1 - sig)
}

# Tanh function: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
Tanh ← {
  x ← 𝕩
  exp2x ← ⊢^(2 × x)
  (exp2x - 1) ÷ (exp2x + 1)
}

# Tanh derivative: tanh'(x) = 1 - tanh²(x)
TanhDerivativeViaInput ← {
  x ← 𝕩
  t ← Tanh x
  1 - (t × t)
}

# Tanh derivative via output: tanh'(t) = 1 - t², where t = tanh(x)
TanhDerivativeViaOutput ← {
  t ← 𝕩
  1 - (t × t)
}

# ReLU (Rectified Linear Unit): max(0, x)
ReLU ← {
  x ← 𝕩
  0 ⌈ x
}

# ReLU derivative: 1 if x > 0, else 0
ReLUDerivative ← {
  x ← 𝕩
  x > 0
}

# Leaky ReLU: max(α*x, x) where α is small (e.g., 0.01)
LeakyReLU ← {
  alpha ← 𝕨 ⊣ 0.01
  x ← 𝕩
  (alpha × x) ⌈ x
}

# Leaky ReLU derivative
LeakyReLUDerivative ← {
  alpha ← 𝕨 ⊣ 0.01
  x ← 𝕩
  (x > 0) + alpha × (x ≤ 0)
}

# Softmax for classification (normalized exponential)
Softmax ← {
  x ← 𝕩
  # Subtract max for numerical stability
  x_shifted ← x - (⌈´ x)
  exp_x ← ⊢^x_shifted
  exp_x ÷ (+´ exp_x)
}

# ============================================================================
# LOSS FUNCTIONS (with gradients)
# ============================================================================

# Binary Cross Entropy Loss (BCELoss)
# BCE(y, p) = -(y*log(p) + (1-y)*log(1-p))
# Numerically stable version with clipping
BinaryCrossEntropy ← {
  y ← 𝕨  # true labels (0 or 1)
  p ← 𝕩  # predicted probabilities

  # Clip probabilities to valid range [1e-7, 1-1e-7]
  epsilon ← 1e-7
  p_clipped ← (epsilon ⌈ p) ⌊ (1 - epsilon)

  # BCE = -(y*log(p) + (1-y)*log(1-p))
  # Equivalent to: -(y*log(p_clipped) + (1-y)*log(1-p_clipped))
  -(+´ (y × (p_clipped⍟⊢)) + (1 - y) × ((1 - p_clipped)⍟⊢))
}

# Binary Cross Entropy Gradient (w.r.t. predictions)
# ∂BCE/∂p = (p - y) / (p*(1-p))
# Simplified numerically stable form
BinaryCrossEntropyGradient ← {
  y ← 𝕨
  p ← 𝕩

  epsilon ← 1e-7
  p_clipped ← (epsilon ⌈ p) ⌊ (1 - epsilon)

  # Gradient: (p - y) / (p*(1-p))
  (p_clipped - y) ÷ (p_clipped × (1 - p_clipped))
}

# Mean Squared Error Loss
MSE ← {
  y ← 𝕨  # true values
  p ← 𝕩  # predictions

  diff ← y - p
  (+´ diff × diff) ÷ (≠diff)
}

# Mean Squared Error Gradient
MSEGradient ← {
  y ← 𝕨
  p ← 𝕩

  # Gradient: 2*(p - y) / n
  diff ← p - y
  2 × diff ÷ (≠diff)
}

# Categorical Cross Entropy (for multi-class classification)
# CCE(y, p) = -sum(y_i * log(p_i))
CategoricalCrossEntropy ← {
  y ← 𝕨  # one-hot encoded labels
  p ← 𝕩  # predicted probabilities (softmax output)

  epsilon ← 1e-7
  p_clipped ← (epsilon ⌈ p) ⌊ (1 - epsilon)

  # CCE = -sum(y * log(p))
  -+´ y × (p_clipped⍟⊢)
}

# ============================================================================
# NUMERICAL STABILITY AND VALIDATION
# ============================================================================

# Check if scalar is finite (not NaN or Inf)
IsFinite ← {
  x ← 𝕩
  ¬ ((x ≠ x) ∨ (x = 1e1000) ∨ (x = ¯1e1000))
}

# Check for NaN (Not a Number)
IsNaN ← {
  x ← 𝕩
  x ≠ x
}

# Check for Infinity
IsInfinite ← {
  x ← 𝕩
  (x = 1e1000) ∨ (x = ¯1e1000)
}

# Validate entire vector (all elements finite)
ValidateVector ← {
  vec ← 𝕩
  ∧´ IsFinite¨ vec
}

# Validate entire matrix (all elements finite)
ValidateMatrix ← {
  mat ← 𝕩
  ∧´˘ ∧´ IsFinite¨¨ mat
}

# Clamp scalar to safe range [-1e10, 1e10]
ClampFiniteScalar ← {
  x ← 𝕩
  max_val ← 1e10
  min_val ← ¯1e10
  (min_val ⌈ x) ⌊ max_val
}

# Clamp vector elements to safe range
ClampFiniteVector ← {
  vec ← 𝕩
  max_val ← 1e10
  min_val ← ¯1e10
  (min_val ⌈ vec) ⌊ max_val
}

# Clamp matrix elements to safe range
ClampFiniteMatrix ← {
  mat ← 𝕩
  max_val ← 1e10
  min_val ← ¯1e10
  (min_val ⌈ mat) ⌊ max_val
}

# Replace NaN and Inf with fallback value
SanitizeVector ← {
  vec ← 𝕩
  fallback ← 0

  # Replace invalid values
  (IsFinite¨ vec) ⊸ / vec ∾ (≠vec) ↑ fallback
}

# Safe division: a/b with fallback when b ≈ 0
SafeDivide ← {
  a ← 𝕨
  b ← 𝕩
  epsilon ← 1e-10

  # If |b| < epsilon, return 0; else return a/b
  (|b) > epsilon ? (a ÷ b) ; 0
}

# ============================================================================
# COMPOSITE OPERATIONS FOR NEURAL NETWORKS
# ============================================================================

# Forward pass of neuron: y = activation(w·x + b)
NeuronForward ← {
  weights ← 𝕨.weights  # vector
  bias ← 𝕨.bias        # scalar
  x ← 𝕩                # input vector
  activation ← 𝕨.activation ⊣ Sigmoid

  z ← (VectorDot weights x) + bias
  activation z
}

# Batch normalization: (x - mean) / sqrt(var + epsilon)
BatchNormalize ← {
  x ← 𝕨  # batch of values (vector or matrix)
  epsilon ← 1e-5

  mean_val ← (x > 0) ? (+´ x ÷ ≠x) ; 0
  var_val ← (+´ (x - mean_val) × (x - mean_val)) ÷ ≠x

  (x - mean_val) ÷ ((var_val + epsilon) ^ 0.5)
}

# Layer normalization
LayerNormalize ← {
  x ← 𝕨
  gamma ← 𝕨.gamma ⊣ 1    # scale parameter
  beta ← 𝕨.beta ⊣ 0      # shift parameter
  epsilon ← 1e-5

  mean_val ← (+´ x) ÷ ≠x
  var_val ← (+´ (x - mean_val) × (x - mean_val)) ÷ ≠x

  x_norm ← (x - mean_val) ÷ ((var_val + epsilon) ^ 0.5)
  (gamma × x_norm) + beta
}

# Dropout: randomly zero out fraction p of elements
Dropout ← {
  x ← 𝕨
  p ← 𝕩  # dropout probability (e.g., 0.5)
  # Note: Pure BQN cannot generate random numbers without system interaction
  # This is a placeholder; in practice, random mask would be generated externally
  x  # Return unchanged in deterministic mode
}

# Gradient clipping: clip gradients to max norm
GradientClip ← {
  gradients ← 𝕨
  max_norm ← 𝕩 ⊣ 1.0

  norm ← VectorNorm gradients
  clip_scale ← (norm ≤ max_norm) ? 1 ; (max_norm ÷ norm)

  clip_scale × gradients
}

# ============================================================================
# WEIGHT INITIALIZATION SCHEMES
# ============================================================================

# Xavier/Glorot initialization: uniform in [-limit, limit]
# where limit = sqrt(6 / (fan_in + fan_out))
XavierInit ← {
  fan_in ← 𝕨
  fan_out ← 𝕩

  limit ← (6 ÷ (fan_in + fan_out)) ^ 0.5
  limit  # Return the limit; actual random values generated externally
}

# He initialization: normal distribution with std = sqrt(2 / fan_in)
HeInit ← {
  fan_in ← 𝕩

  stddev ← (2 ÷ fan_in) ^ 0.5
  stddev  # Return the standard deviation
}

# ============================================================================
# MATRIX DECOMPOSITION AND ANALYSIS
# ============================================================================

# Frobenius norm of matrix: sqrt(sum(a_ij^2))
FrobeniusNorm ← {
  mat ← 𝕩
  (+´ +´˘ mat × mat) ^ 0.5
}

# Trace of square matrix: sum of diagonal elements
Trace ← {
  mat ← 𝕩
  +´ (=≠mat) / ⊢ mat
}

# Determinant of 2×2 matrix (specialized for efficiency)
Determinant2x2 ← {
  mat ← 𝕩
  a ← mat.0.0
  b ← mat.0.1
  c ← mat.1.0
  d ← mat.1.1
  (a × d) - (b × c)
}

# ============================================================================
# VECTORIZED OPERATIONS (for batch processing)
# ============================================================================

# Apply scalar function element-wise to vector
VectorApply ← {
  f ← 𝕨
  vec ← 𝕩
  f¨ vec
}

# Apply activation function element-wise to matrix
MatrixApply ← {
  f ← 𝕨
  mat ← 𝕩
  f¨¨ mat
}

# Element-wise maximum of two vectors
VectorMaximum ← ⌈

# Element-wise minimum of two vectors
VectorMinimum ← ⌊

# ============================================================================
# TEST AND DEMONSTRATION FUNCTIONS
# ============================================================================

# Test scalar operations
TestScalarOps ← {
  x ← 5
  y ← 3
  ⟨
    "Add": Add x y,
    "Subtract": Subtract x y,
    "Multiply": Multiply x y,
    "Divide": Divide x y,
    "Power": Power x y,
    "Square": Square x,
    "SquareRoot": SquareRoot 16
  ⟩
}

# Test vector operations
TestVectorOps ← {
  v1 ← ⟨1, 2, 3⟩
  v2 ← ⟨4, 5, 6⟩
  ⟨
    "VectorAdd": VectorAdd v1 v2,
    "VectorDot": VectorDot v1 v2,
    "VectorNorm": VectorNorm v1,
    "VectorMean": VectorMean v1
  ⟩
}

# Test activation functions
TestActivations ← {
  x ← 0.5
  ⟨
    "Sigmoid": Sigmoid x,
    "SigmoidDerivative": SigmoidDerivative (Sigmoid x),
    "ReLU": ReLU x,
    "Tanh": Tanh x
  ⟩
}

# Test loss functions
TestLossFunctions ← {
  y ← ⟨0, 1, 1, 0⟩
  p ← ⟨0.1, 0.9, 0.8, 0.2⟩
  ⟨
    "BCE": BinaryCrossEntropy y p,
    "MSE": MSE y p
  ⟩
}

# Test numerical stability
TestNumericalStability ← {
  x ← 1e-10
  large ← 1e20
  invalid ← 0 ÷ 0
  ⟨
    "IsFiniteSmall": IsFinite x,
    "IsFiniteLarge": IsFinite large,
    "IsNaNInvalid": IsNaN invalid,
    "ValidateVector": ValidateVector ⟨1, 2, 3⟩
  ⟩
}

# ============================================================================
# EXPORTED PUBLIC API
# ============================================================================

{
  # Scalar Operations
  Add, Subtract, Multiply, Divide, Negate, Absolute,
  Minimum, Maximum, Clamp, Power, Exponential, Logarithm,
  Square, SquareRoot,

  # Vector Operations
  VectorAdd, VectorSubtract, VectorScale, VectorDot,
  VectorSum, VectorMean, VectorNorm, VectorZero, VectorFill,
  ElementWiseMultiply, ElementWiseDivide, VectorNormalize,

  # Matrix Operations
  MatrixZeros, MatrixIdentity, MatrixAdd, MatrixSubtract,
  MatrixScale, MatrixTranspose, MatrixMultiply,
  MatrixVectorMultiply, OuterProduct, MatrixSum, MatrixMean,
  MatrixRowSum, MatrixColumnSum, MatrixShape,

  # Activation Functions
  Sigmoid, SigmoidDerivative, SigmoidDerivativeViaInput,
  Tanh, TanhDerivativeViaInput, TanhDerivativeViaOutput,
  ReLU, ReLUDerivative, LeakyReLU, LeakyReLUDerivative,
  Softmax,

  # Loss Functions
  BinaryCrossEntropy, BinaryCrossEntropyGradient,
  MSE, MSEGradient, CategoricalCrossEntropy,

  # Numerical Stability
  IsFinite, IsNaN, IsInfinite, ValidateVector, ValidateMatrix,
  ClampFiniteScalar, ClampFiniteVector, ClampFiniteMatrix,
  SanitizeVector, SafeDivide,

  # Composite Operations
  NeuronForward, BatchNormalize, LayerNormalize, Dropout,
  GradientClip,

  # Weight Initialization
  XavierInit, HeInit,

  # Matrix Analysis
  FrobeniusNorm, Trace, Determinant2x2,

  # Vectorized Operations
  VectorApply, MatrixApply, VectorMaximum, VectorMinimum,

  # Test Functions
  TestScalarOps, TestVectorOps, TestActivations,
  TestLossFunctions, TestNumericalStability
}
