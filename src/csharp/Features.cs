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
/// SYMBOLIC LAYER — Boolean grammatical feature space.
///
/// 23 Boolean features organized into 6 grammatical categories.
/// The network outputs continuous values in [0,1]; these are thresholded
/// to produce discrete Boolean grammatical states.
/// </summary>
public static class Features
{
    public static readonly string[] Names =
    {
        "PERSON_1",   "PERSON_2",   "PERSON_3",
        "SINGULAR",   "PLURAL",
        "PRESENT",    "IMPERFECT",  "FUTURE",     "PERFECT",
        "INDICATIVE", "SUBJUNCTIVE","IMPERATIVE",
        "ACTIVE",     "PASSIVE",
        "CONJ_1",     "CONJ_2",     "CONJ_3",
    };

    public static readonly (string Category, string[] FeatureNames)[] Groups =
    {
        ("PERSON",      new[] { "PERSON_1", "PERSON_2", "PERSON_3" }),
        ("NUMBER",      new[] { "SINGULAR", "PLURAL" }),
        ("TENSE",       new[] { "PRESENT", "IMPERFECT", "FUTURE", "PERFECT" }),
        ("MOOD",        new[] { "INDICATIVE", "SUBJUNCTIVE", "IMPERATIVE" }),
        ("VOICE",       new[] { "ACTIVE", "PASSIVE" }),
        ("CONJUGATION", new[] { "CONJ_1", "CONJ_2", "CONJ_3" }),
    };

    public const int Count = 23;
    public const double BooleanThreshold = 0.5;

    static readonly Dictionary<string, int> Index = BuildIndex();

    static Dictionary<string, int> BuildIndex()
    {
        var idx = new Dictionary<string, int>();
        for (int i = 0; i < Names.Length; i++)
            idx[Names[i]] = i;
        return idx;
    }

    /// <summary>
    /// Build a target vector from named feature assignments.
    /// Each entry maps a category to either a feature name (string)
    /// or a positional index (int) within that category.
    /// </summary>
    public static double[] MakeTarget(params (string Category, object Value)[] specs)
    {
        var t = new double[Count];
        foreach (var (category, value) in specs)
        {
            var group = Groups.First(g => g.Category == category);
            int fi;
            if (value is string s)
                fi = Array.IndexOf(group.FeatureNames, s);
            else if (value is int i)
                fi = i;
            else
                throw new ArgumentException($"Value must be string or int, got {value.GetType()}");

            if (fi < 0 || fi >= group.FeatureNames.Length)
                throw new ArgumentException($"Invalid feature '{value}' in category '{category}'");

            t[Index[group.FeatureNames[fi]]] = 1.0;
        }
        return t;
    }

    public static string FormatPrediction(double[] values)
    {
        var lines = new List<string>();
        foreach (var (category, featureNames) in Groups)
        {
            lines.Add($"  {category}:");
            foreach (var name in featureNames)
            {
                int idx = Index[name];
                double v = values[idx];
                bool b = v >= BooleanThreshold;
                lines.Add($"    {name,-14} = {v,8:F5}  ->  {(b ? "TRUE" : "FALSE")}");
            }
        }
        return string.Join("\n", lines);
    }

    public static int CountCorrect(double[] predicted, double[] target)
    {
        int correct = 0;
        for (int i = 0; i < Count; i++)
        {
            bool pred = predicted[i] >= BooleanThreshold;
            bool tgt = target[i] >= 0.5;
            if (pred == tgt) correct++;
        }
        return correct;
    }
}
