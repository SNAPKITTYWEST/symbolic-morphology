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

/// SYMBOLIC LAYER — Boolean grammatical feature space.
///
/// 23 Boolean features organized into 6 grammatical categories.
/// The network outputs continuous values in [0,1]; these are thresholded
/// to produce discrete Boolean grammatical states.

pub const FEATURE_NAMES: &[&str] = &[
    "PERSON_1",   "PERSON_2",   "PERSON_3",
    "SINGULAR",   "PLURAL",
    "PRESENT",    "IMPERFECT",  "FUTURE",     "PERFECT",
    "INDICATIVE", "SUBJUNCTIVE","IMPERATIVE",
    "ACTIVE",     "PASSIVE",
    "CONJ_1",     "CONJ_2",     "CONJ_3",
];

pub const NUM_FEATURES: usize = 23;
pub const BOOLEAN_THRESHOLD: f64 = 0.5;

pub struct FeatureGroup {
    pub category: &'static str,
    pub names: &'static [&'static str],
}

pub const FEATURE_GROUPS: &[FeatureGroup] = &[
    FeatureGroup { category: "PERSON",      names: &["PERSON_1", "PERSON_2", "PERSON_3"] },
    FeatureGroup { category: "NUMBER",      names: &["SINGULAR", "PLURAL"] },
    FeatureGroup { category: "TENSE",       names: &["PRESENT", "IMPERFECT", "FUTURE", "PERFECT"] },
    FeatureGroup { category: "MOOD",        names: &["INDICATIVE", "SUBJUNCTIVE", "IMPERATIVE"] },
    FeatureGroup { category: "VOICE",       names: &["ACTIVE", "PASSIVE"] },
    FeatureGroup { category: "CONJUGATION", names: &["CONJ_1", "CONJ_2", "CONJ_3"] },
];

fn feature_index(name: &str) -> usize {
    FEATURE_NAMES.iter().position(|&n| n == name)
        .unwrap_or_else(|| panic!("Unknown feature: {name}"))
}

/// Build a target vector from (category, feature_name) pairs.
pub fn make_target(spec: &[(&str, &str)]) -> [f64; NUM_FEATURES] {
    let mut target = [0.0; NUM_FEATURES];
    for &(category, feature_name) in spec {
        let group = FEATURE_GROUPS.iter()
            .find(|g| g.category == category)
            .unwrap_or_else(|| panic!("Unknown category: {category}"));
        let _ = group.names.iter()
            .position(|&n| n == feature_name)
            .unwrap_or_else(|| panic!("Feature '{feature_name}' not in category '{category}'"));
        target[feature_index(feature_name)] = 1.0;
    }
    target
}

pub fn count_correct(predicted: &[f64; NUM_FEATURES], target: &[f64; NUM_FEATURES]) -> usize {
    let mut correct = 0;
    for i in 0..NUM_FEATURES {
        let pred = predicted[i] >= BOOLEAN_THRESHOLD;
        let tgt = target[i] >= 0.5;
        if pred == tgt { correct += 1; }
    }
    correct
}

pub fn format_prediction(values: &[f64; NUM_FEATURES]) -> String {
    let mut lines = Vec::new();
    for group in FEATURE_GROUPS {
        lines.push(format!("  {}:", group.category));
        for &name in group.names {
            let idx = feature_index(name);
            let v = values[idx];
            let b = if v >= BOOLEAN_THRESHOLD { "TRUE" } else { "FALSE" };
            lines.push(format!("    {:<14} = {:8.5}  ->  {}", name, v, b));
        }
    }
    lines.join("\n")
}
