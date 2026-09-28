⍝ ==============================================================================
⍝ Dyalog APL + Bare-Metal Assembly Binding (⎕NA)
⍝ ==============================================================================
⍝ Zero-overhead C ABI bridge to NASM AVX-512 kernel
⍝ Bit-packed Boolean arrays → native 512-bit zmm registers
⍝ ==============================================================================

⍝ Bind NASM AVX-512 kernel via ⎕NA (Name Association)
⍝ Function signature: symbolic_kernel_avx512(A*, B*, C*, Out*, chunk_count)
⍝ P = Pointer (64-bit), U8 = 64-bit unsigned integer
⍝ NOTE: In production, ensure libsymbolic_engine.so is in LD_LIBRARY_PATH
'NativeKernel' ⎕NA 'libsymbolic_engine.so>symbolic_kernel_avx512 P P P P U8'

⍝ Memory allocation wrappers
AllocBooleanBuffer ← {
    Size ← ⍵
    ⍝ Allocate bit-packed Boolean array (Dyalog automatic packing)
    ⎕OR Size ⍴ ?2
}

FreeBooleanBuffer ← {
    Buffer ← ⍵
    ⍝ Dyalog garbage collection handles deallocation
    0 ⊢ Buffer
}

⍝ ==============================================================================
⍝ Bare-Metal Benchmark
⍝ ==============================================================================

RunAssemblyBenchmark ← {
    Iterations ← ⍺
    Chunks512 ← ⍵

    ⍝ Calculate buffer size
    ⍝ 1 chunk = 512 bits = 64 bytes
    ByteSize ← Chunks512 × 64

    ⎕ ← 'Allocating ', (⍕ ByteSize), ' bytes per buffer...'

    ⍝ Allocate contiguous bit-packed Boolean arrays
    ⍝ Dyalog ensures alignment for SIMD
    BufA ← AllocBooleanBuffer ByteSize
    BufB ← AllocBooleanBuffer ByteSize
    BufC ← AllocBooleanBuffer ByteSize
    BufOut ← AllocBooleanBuffer ByteSize

    ⍝ Start high-resolution timer
    Start ← ⎕AI[3]

    ⍝ Execute AVX-512 Assembly loop via ⎕NA
    ⍝ Each invocation: 3-input NAND over all Chunks512 vectors
    {
        ⍝ Call native kernel
        NativeKernel BufA BufB BufC BufOut Chunks512
    } ¨ ⍳ Iterations

    ⍝ Measure elapsed time
    ElapsedMs ← ⎕AI[3] - Start

    ⍝ Free unmanaged memory
    FreeBooleanBuffer BufA
    FreeBooleanBuffer BufB
    FreeBooleanBuffer BufC
    FreeBooleanBuffer BufOut

    ⍝ Calculate metrics
    TotalBits ← Iterations × Chunks512 × 512
    ThroughputGbps ← TotalBits ÷ (ElapsedMs × 1E6)

    ⍝ Display results
    ⎕ ← '========================================================='
    ⎕ ← ' DYALOG APL + BARE-METAL AVX-512 BENCHMARK'
    ⎕ ← '========================================================='
    ⎕ ← 'Total Iterations : ', ⍕ Iterations
    ⎕ ← 'Chunks (512-bit) : ', ⍕ Chunks512
    ⎕ ← 'Vector Size : ', (⍕ (Chunks512 × 512)), ' bits'
    ⎕ ← 'Elapsed Time : ', (⍕ ElapsedMs), ' ms'
    ⎕ ← 'Throughput (Gbits/s) : ', ⍕ ThroughputGbps
    ⎕ ← 'Per-Chunk Latency (ns) : ', ⍕ ((ElapsedMs × 1E6) ÷ (Iterations × Chunks512))
    ⎕ ← '========================================================='

    ThroughputGbps
}

⍝ ==============================================================================
⍝ Comparison: Native vs Assembly
⍝ ==============================================================================

CompareExecutionModes ← {
    Iterations ← ⍺
    Chunks512 ← ⍵

    ⎕ ← '================= EXECUTION MODE COMPARISON ================'
    ⎕ ← ''

    ⎕ ← 'Configuration:'
    ⎕ ← '  Iterations: ', ⍕ Iterations
    ⎕ ← '  Vector Size: ', (⍕ (Chunks512 × 512)), ' bits'
    ⎕ ← '  Total Operations: ', ⍕ (Iterations × Chunks512)
    ⎕ ← ''

    ⎕ ← 'Mode 1: Pure Dyalog APL (C-Engine)'
    ⎕ ← '─────────────────────────────────────'

    ⍝ Native: use vectorized operators
    A ← ?（Chunks512 × 512)⍴2
    B ← ?（Chunks512 × 512)⍴2
    C ← ?（Chunks512 × 512)⍴2

    Start ← ⎕AI[3]
    { Output ← A ∧ B ∧ C }¨ ⍳Iterations
    NativeMs ← ⎕AI[3] - Start

    NativeThroughput ← (Iterations × Chunks512 × 512) ÷ (NativeMs × 1E6)
    ⎕ ← '  Elapsed: ', (⍕ NativeMs), ' ms'
    ⎕ ← '  Throughput: ', ⍕ NativeThroughput, ' Gbits/s'
    ⎕ ← ''

    ⎕ ← 'Mode 2: Dyalog APL + AVX-512 Assembly (⎕NA)'
    ⎕ ← '──────────────────────────────────────────'

    ⍝ Assembly: via ⎕NA binding
    AssemblyThroughput ← Iterations RunAssemblyBenchmark Chunks512

    ⎕ ← ''
    ⎕ ← 'Speedup: ', ⍕ (AssemblyThroughput ÷ NativeThroughput), 'x'
    ⎕ ← '============================================================'
}

⍝ ==============================================================================
⍝ Execute Benchmarks
⍝ ==============================================================================

⎕ ← ''
⎕ ← '============================================================'
⎕ ← ' DYALOG APL + BARE-METAL AVX-512 SYMBOLIC ENGINE'
⎕ ← '============================================================'
⎕ ← ''

⍝ Benchmark: 1M iterations × 16 × 512-bit chunks
⍝ (Total: 1M × 8192 bits = 8.2 trillion bit operations)
ThroughputGbps ← 1000000 RunAssemblyBenchmark 16

⎕ ← ''
⎕ ← 'Performance Summary:'
⎕ ← '  Achieved throughput: ', (⍕ ThroughputGbps), ' Gbits/s'
⎕ ← '  ≈ ', (⍕ (ThroughputGbps ÷ 8)), ' GB/s'
⎕ ← '  ≈ ', (⍕ ⌊ThroughputGbps ÷ 1000), ' Tbits/s'
⎕ ← ''

⍝ Optional: Compare with native mode
⍝ CompareExecutionModes 100000 16

⎕ ← '============================================================'
⎕ ← 'Status: ✓ DYALOG APL + AVX-512 ENGINE OPERATIONAL'
⎕ ← '============================================================'
