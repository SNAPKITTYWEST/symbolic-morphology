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

using Sovereign.Engine;
using System.Diagnostics;

// ═══════════════════════════════════════════════════════════════════
//  SYMBOLIC LEARNING ENGINE — BENCHMARK SUITE
//  C# / .NET 8 — hand-rolled from mathematical primitives
// ═══════════════════════════════════════════════════════════════════

Console.WriteLine("═══════════════════════════════════════════════════════════════");
Console.WriteLine("  SYMBOLIC LEARNING ENGINE — BENCHMARK SUITE");
Console.WriteLine("  C# / .NET 8 — hand-rolled from mathematical primitives");
Console.WriteLine("═══════════════════════════════════════════════════════════════");
Console.WriteLine();

var corpus = Dataset.BuildCorpus();
var unseen = Dataset.UnseenWords();

const int EMBED_DIM = 16;
const int HIDDEN_DIM = 32;
const double LEARNING_RATE = 0.5;
const int EPOCHS = 3000;

var engine = new Engine(
    embedDim: EMBED_DIM,
    hiddenDim: HIDDEN_DIM,
    outputDim: Features.Count,
    learningRate: LEARNING_RATE,
    seed: 42
);

Console.WriteLine($"  Corpus: {corpus.Length} word forms, {Features.Count} features, {unseen.Length} unseen");
Console.WriteLine($"  Arch: {EMBED_DIM}-dim embed -> mean pool -> {HIDDEN_DIM}-dim hidden (tanh) -> {Features.Count}-dim output (sigmoid)");
Console.WriteLine();

// ═══════════════════════════════════════════════════════════════
//  BENCHMARK 1: TRAINING
// ═══════════════════════════════════════════════════════════════

Console.WriteLine("─────────────────────────────────────────────────────────────");
Console.WriteLine($"  BENCHMARK 1: TRAINING ({EPOCHS} epochs x {corpus.Length} examples)");
Console.WriteLine("─────────────────────────────────────────────────────────────");

var trainSw = Stopwatch.StartNew();
double finalLoss = 0;

for (int epoch = 0; epoch <= EPOCHS; epoch++)
{
    double totalLoss = 0;

    for (int s = 0; s < corpus.Length; s++)
    {
        var (word, target) = corpus[s];
        engine.Forward(word);
        totalLoss += engine.ComputeLoss(target);
        var (gW1, gB1, gW2, gB2, _) = engine.Backward(target);
        engine.GradientDescentStep(gW1, gB1, gW2, gB2);
    }

    finalLoss = totalLoss / corpus.Length;

    if (epoch % 500 == 0 || epoch == EPOCHS)
    {
        int totalCorrect = 0, totalFeatures = 0;
        for (int s = 0; s < corpus.Length; s++)
        {
            var (word, target) = corpus[s];
            var pred = engine.Forward(word);
            totalCorrect += Features.CountCorrect(pred, target);
            totalFeatures += Features.Count;
        }
        double accuracy = (double)totalCorrect / totalFeatures * 100;
        Console.WriteLine($"  Epoch {epoch,5} | Loss: {finalLoss,10:F6} | Acc: {accuracy,6:F1}% | {trainSw.Elapsed.TotalSeconds:F2}s");
    }
}

trainSw.Stop();
long totalExamples = (long)(EPOCHS + 1) * corpus.Length;
double examplesPerSec = totalExamples / trainSw.Elapsed.TotalSeconds;

Console.WriteLine();
Console.WriteLine("  TRAINING RESULTS:");
Console.WriteLine($"    Wall clock:       {trainSw.Elapsed.TotalSeconds:F3}s");
Console.WriteLine($"    Total examples:   {totalExamples}");
Console.WriteLine($"    Throughput:       {examplesPerSec:F0} examples/sec");
Console.WriteLine($"    Final loss:       {finalLoss:F6}");
Console.WriteLine();

// ═══════════════════════════════════════════════════════════════
//  BENCHMARK 2: INFERENCE LATENCY
// ═══════════════════════════════════════════════════════════════

Console.WriteLine("─────────────────────────────────────────────────────────────");
Console.WriteLine("  BENCHMARK 2: INFERENCE LATENCY");
Console.WriteLine("─────────────────────────────────────────────────────────────");

// Warm up
foreach (var (word, _) in corpus) engine.Forward(word);

int nInference = 1000;
var infSw = Stopwatch.StartNew();

for (int pass = 0; pass < nInference; pass++)
{
    foreach (var (word, _) in corpus)
    {
        engine.Forward(word);
    }
}

infSw.Stop();
long totalInferences = (long)nInference * corpus.Length;
double usPerInference = infSw.Elapsed.TotalMicroseconds / totalInferences;

Console.WriteLine($"  {totalInferences} inferences ({corpus.Length} words x {nInference} passes)");
Console.WriteLine($"    Wall clock:       {infSw.Elapsed.TotalSeconds:F3}s");
Console.WriteLine($"    Per inference:    {usPerInference:F2} µs");
Console.WriteLine($"    Throughput:       {totalInferences / infSw.Elapsed.TotalSeconds:F0} inferences/sec");
Console.WriteLine();

// ═══════════════════════════════════════════════════════════════
//  BENCHMARK 3: FULL FORWARD+BACKWARD LATENCY
// ═══════════════════════════════════════════════════════════════

Console.WriteLine("─────────────────────────────────────────────────────────────");
Console.WriteLine("  BENCHMARK 3: FORWARD + BACKWARD LATENCY");
Console.WriteLine("─────────────────────────────────────────────────────────────");

int nFb = 1000;
var fbSw = Stopwatch.StartNew();

for (int pass = 0; pass < nFb; pass++)
{
    foreach (var (word, target) in corpus)
    {
        engine.Forward(word);
        engine.ComputeLoss(target);
        engine.Backward(target);
    }
}

fbSw.Stop();
long totalFb = (long)nFb * corpus.Length;
double usPerFb = fbSw.Elapsed.TotalMicroseconds / totalFb;

Console.WriteLine($"  {totalFb} forward+backward passes ({corpus.Length} words x {nFb} passes)");
Console.WriteLine($"    Wall clock:       {fbSw.Elapsed.TotalSeconds:F3}s");
Console.WriteLine($"    Per pass:         {usPerFb:F2} µs");
Console.WriteLine($"    Throughput:       {totalFb / fbSw.Elapsed.TotalSeconds:F0} passes/sec");
Console.WriteLine();

// ═══════════════════════════════════════════════════════════════
//  BENCHMARK 4: GENERALIZATION QUALITY
// ═══════════════════════════════════════════════════════════════

Console.WriteLine("─────────────────────────────────────────────────────────────");
Console.WriteLine("  BENCHMARK 4: GENERALIZATION QUALITY");
Console.WriteLine("─────────────────────────────────────────────────────────────");

int trainCorrect = 0, trainTotal = 0, trainPerfect = 0;
foreach (var (word, target) in corpus)
{
    var pred = engine.Forward(word);
    int c = Features.CountCorrect(pred, target);
    trainCorrect += c;
    trainTotal += Features.Count;
    if (c == Features.Count) trainPerfect++;
}

int unseenCorrect = 0, unseenTotal = 0, unseenPerfect = 0;
foreach (var (word, target) in unseen)
{
    var pred = engine.Forward(word);
    int c = Features.CountCorrect(pred, target);
    unseenCorrect += c;
    unseenTotal += Features.Count;
    if (c == Features.Count) unseenPerfect++;
}

Console.WriteLine($"  Training set:  {trainPerfect}/{corpus.Length} perfect words, {(double)trainCorrect / trainTotal * 100:F1}% feature accuracy");
Console.WriteLine($"  Unseen set:    {unseenPerfect}/{unseen.Length} perfect words, {(double)unseenCorrect / unseenTotal * 100:F1}% feature accuracy");
Console.WriteLine();

// ═══════════════════════════════════════════════════════════════
//  SAMPLE INFERENCE
// ═══════════════════════════════════════════════════════════════

Console.WriteLine("─────────────────────────────────────────────────────────────");
Console.WriteLine("  SAMPLE INFERENCE");
Console.WriteLine("─────────────────────────────────────────────────────────────");

foreach (var word in new[] { "AMO", "AMABAT", "MONET", "REGIT", "AGUNT" })
{
    var target = corpus.First(c => c.Word == word).Target;
    var pred = engine.Forward(word);
    int errors = Features.Count - Features.CountCorrect(pred, target);
    Console.WriteLine();
    Console.WriteLine($"  INPUT: {word}");
    Console.WriteLine(Features.FormatPrediction(pred));
    Console.WriteLine($"  ERRORS: {errors}/{Features.Count}");
}

Console.WriteLine();
Console.WriteLine("═══════════════════════════════════════════════════════════════");
Console.WriteLine("  BENCHMARK COMPLETE");
Console.WriteLine("═══════════════════════════════════════════════════════════════");
