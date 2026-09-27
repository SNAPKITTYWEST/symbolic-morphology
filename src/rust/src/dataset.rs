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

use crate::features::{self, NUM_FEATURES};

pub type TrainingExample = (String, [f64; NUM_FEATURES]);

fn make(spec: &[(&str, &str)]) -> [f64; NUM_FEATURES] {
    features::make_target(spec)
}

fn p1() -> (&'static str, &'static str) { ("PERSON", "PERSON_1") }
fn p2() -> (&'static str, &'static str) { ("PERSON", "PERSON_2") }
fn p3() -> (&'static str, &'static str) { ("PERSON", "PERSON_3") }
fn sg() -> (&'static str, &'static str) { ("NUMBER", "SINGULAR") }
fn pl() -> (&'static str, &'static str) { ("NUMBER", "PLURAL") }
fn pres() -> (&'static str, &'static str) { ("TENSE", "PRESENT") }
fn impf() -> (&'static str, &'static str) { ("TENSE", "IMPERFECT") }
fn fut() -> (&'static str, &'static str) { ("TENSE", "FUTURE") }
fn perf() -> (&'static str, &'static str) { ("TENSE", "PERFECT") }
fn ind() -> (&'static str, &'static str) { ("MOOD", "INDICATIVE") }
fn act() -> (&'static str, &'static str) { ("VOICE", "ACTIVE") }
fn c1() -> (&'static str, &'static str) { ("CONJUGATION", "CONJ_1") }
fn c2() -> (&'static str, &'static str) { ("CONJUGATION", "CONJ_2") }
fn c3() -> (&'static str, &'static str) { ("CONJUGATION", "CONJ_3") }

fn t(specs: &[(&str, &str)]) -> [f64; NUM_FEATURES] { make(specs) }

pub fn build_corpus() -> Vec<TrainingExample> {
    vec![
        // ─── 1st Conjugation: amare (to love) ──────────────────────
        // Present indicative active
        ("AMO".into(),    t(&[p1(), sg(), pres(), ind(), act(), c1()])),
        ("AMAS".into(),   t(&[p2(), sg(), pres(), ind(), act(), c1()])),
        ("AMAT".into(),   t(&[p3(), sg(), pres(), ind(), act(), c1()])),
        ("AMAMUS".into(), t(&[p1(), pl(), pres(), ind(), act(), c1()])),
        ("AMATIS".into(), t(&[p2(), pl(), pres(), ind(), act(), c1()])),
        ("AMANT".into(),  t(&[p3(), pl(), pres(), ind(), act(), c1()])),
        // Imperfect indicative active
        ("AMABAM".into(),   t(&[p1(), sg(), impf(), ind(), act(), c1()])),
        ("AMABAS".into(),   t(&[p2(), sg(), impf(), ind(), act(), c1()])),
        ("AMABAT".into(),   t(&[p3(), sg(), impf(), ind(), act(), c1()])),
        ("AMABAMUS".into(), t(&[p1(), pl(), impf(), ind(), act(), c1()])),
        ("AMABATIS".into(), t(&[p2(), pl(), impf(), ind(), act(), c1()])),
        ("AMABANT".into(),  t(&[p3(), pl(), impf(), ind(), act(), c1()])),
        // Future indicative active
        ("AMABO".into(),    t(&[p1(), sg(), fut(), ind(), act(), c1()])),
        ("AMABIS".into(),   t(&[p2(), sg(), fut(), ind(), act(), c1()])),
        ("AMABIT".into(),   t(&[p3(), sg(), fut(), ind(), act(), c1()])),
        ("AMABIMUS".into(), t(&[p1(), pl(), fut(), ind(), act(), c1()])),
        ("AMABITIS".into(), t(&[p2(), pl(), fut(), ind(), act(), c1()])),
        ("AMABUNT".into(),  t(&[p3(), pl(), fut(), ind(), act(), c1()])),
        // Perfect indicative active
        ("AMAVI".into(),     t(&[p1(), sg(), perf(), ind(), act(), c1()])),
        ("AMAVISTI".into(),  t(&[p2(), sg(), perf(), ind(), act(), c1()])),
        ("AMAVIT".into(),    t(&[p3(), sg(), perf(), ind(), act(), c1()])),
        ("AMAVIMUS".into(),  t(&[p1(), pl(), perf(), ind(), act(), c1()])),
        ("AMAVISTIS".into(), t(&[p2(), pl(), perf(), ind(), act(), c1()])),
        ("AMAVERUNT".into(), t(&[p3(), pl(), perf(), ind(), act(), c1()])),

        // ─── 1st Conjugation: laudare (to praise) ──────────────────
        ("LAUDO".into(),      t(&[p1(), sg(), pres(), ind(), act(), c1()])),
        ("LAUDAS".into(),     t(&[p2(), sg(), pres(), ind(), act(), c1()])),
        ("LAUDAT".into(),     t(&[p3(), sg(), pres(), ind(), act(), c1()])),
        ("LAUDAMUS".into(),   t(&[p1(), pl(), pres(), ind(), act(), c1()])),
        ("LAUDATIS".into(),   t(&[p2(), pl(), pres(), ind(), act(), c1()])),
        ("LAUDANT".into(),    t(&[p3(), pl(), pres(), ind(), act(), c1()])),
        ("LAUDABAM".into(),   t(&[p1(), sg(), impf(), ind(), act(), c1()])),
        ("LAUDABAS".into(),   t(&[p2(), sg(), impf(), ind(), act(), c1()])),
        ("LAUDABAT".into(),   t(&[p3(), sg(), impf(), ind(), act(), c1()])),
        ("LAUDABAMUS".into(), t(&[p1(), pl(), impf(), ind(), act(), c1()])),
        ("LAUDABATIS".into(), t(&[p2(), pl(), impf(), ind(), act(), c1()])),
        ("LAUDABANT".into(),  t(&[p3(), pl(), impf(), ind(), act(), c1()])),
        ("LAUDAVI".into(),    t(&[p1(), sg(), perf(), ind(), act(), c1()])),
        ("LAUDAVISTI".into(), t(&[p2(), sg(), perf(), ind(), act(), c1()])),
        ("LAUDAVIT".into(),   t(&[p3(), sg(), perf(), ind(), act(), c1()])),

        // ─── 2nd Conjugation: monere (to warn) ─────────────────────
        ("MONEO".into(),     t(&[p1(), sg(), pres(), ind(), act(), c2()])),
        ("MONES".into(),     t(&[p2(), sg(), pres(), ind(), act(), c2()])),
        ("MONET".into(),     t(&[p3(), sg(), pres(), ind(), act(), c2()])),
        ("MONEMUS".into(),   t(&[p1(), pl(), pres(), ind(), act(), c2()])),
        ("MONETIS".into(),   t(&[p2(), pl(), pres(), ind(), act(), c2()])),
        ("MONENT".into(),    t(&[p3(), pl(), pres(), ind(), act(), c2()])),
        ("MONEBAM".into(),   t(&[p1(), sg(), impf(), ind(), act(), c2()])),
        ("MONEBAS".into(),   t(&[p2(), sg(), impf(), ind(), act(), c2()])),
        ("MONEBAT".into(),   t(&[p3(), sg(), impf(), ind(), act(), c2()])),
        ("MONEBAMUS".into(), t(&[p1(), pl(), impf(), ind(), act(), c2()])),
        ("MONEBATIS".into(), t(&[p2(), pl(), impf(), ind(), act(), c2()])),
        ("MONEBANT".into(),  t(&[p3(), pl(), impf(), ind(), act(), c2()])),
        ("MONUI".into(),     t(&[p1(), sg(), perf(), ind(), act(), c2()])),
        ("MONUISTI".into(),  t(&[p2(), sg(), perf(), ind(), act(), c2()])),
        ("MONUIT".into(),    t(&[p3(), sg(), perf(), ind(), act(), c2()])),

        // ─── 2nd Conjugation: habere (to have) ─────────────────────
        ("HABEO".into(),    t(&[p1(), sg(), pres(), ind(), act(), c2()])),
        ("HABES".into(),    t(&[p2(), sg(), pres(), ind(), act(), c2()])),
        ("HABET".into(),    t(&[p3(), sg(), pres(), ind(), act(), c2()])),
        ("HABEMUS".into(),  t(&[p1(), pl(), pres(), ind(), act(), c2()])),
        ("HABETIS".into(),  t(&[p2(), pl(), pres(), ind(), act(), c2()])),
        ("HABENT".into(),   t(&[p3(), pl(), pres(), ind(), act(), c2()])),
        ("HABUI".into(),    t(&[p1(), sg(), perf(), ind(), act(), c2()])),
        ("HABUISTI".into(), t(&[p2(), sg(), perf(), ind(), act(), c2()])),
        ("HABUIT".into(),   t(&[p3(), sg(), perf(), ind(), act(), c2()])),

        // ─── 3rd Conjugation: regere (to rule) ─────────────────────
        ("REGO".into(),     t(&[p1(), sg(), pres(), ind(), act(), c3()])),
        ("REGIS".into(),    t(&[p2(), sg(), pres(), ind(), act(), c3()])),
        ("REGIT".into(),    t(&[p3(), sg(), pres(), ind(), act(), c3()])),
        ("REGIMUS".into(),  t(&[p1(), pl(), pres(), ind(), act(), c3()])),
        ("REGITIS".into(),  t(&[p2(), pl(), pres(), ind(), act(), c3()])),
        ("REGUNT".into(),   t(&[p3(), pl(), pres(), ind(), act(), c3()])),
        ("REGEBAM".into(),  t(&[p1(), sg(), impf(), ind(), act(), c3()])),
        ("REGEBAS".into(),  t(&[p2(), sg(), impf(), ind(), act(), c3()])),
        ("REGEBAT".into(),  t(&[p3(), sg(), impf(), ind(), act(), c3()])),
        ("REGEBAMUS".into(),t(&[p1(), pl(), impf(), ind(), act(), c3()])),
        ("REGEBATIS".into(),t(&[p2(), pl(), impf(), ind(), act(), c3()])),
        ("REGEBANT".into(), t(&[p3(), pl(), impf(), ind(), act(), c3()])),
        ("REXI".into(),     t(&[p1(), sg(), perf(), ind(), act(), c3()])),
        ("REXISTI".into(),  t(&[p2(), sg(), perf(), ind(), act(), c3()])),
        ("REXIT".into(),    t(&[p3(), sg(), perf(), ind(), act(), c3()])),

        // ─── 3rd Conjugation: agere (to do/drive) ──────────────────
        ("AGO".into(),     t(&[p1(), sg(), pres(), ind(), act(), c3()])),
        ("AGIS".into(),    t(&[p2(), sg(), pres(), ind(), act(), c3()])),
        ("AGIT".into(),    t(&[p3(), sg(), pres(), ind(), act(), c3()])),
        ("AGIMUS".into(),  t(&[p1(), pl(), pres(), ind(), act(), c3()])),
        ("AGITIS".into(),  t(&[p2(), pl(), pres(), ind(), act(), c3()])),
        ("AGUNT".into(),   t(&[p3(), pl(), pres(), ind(), act(), c3()])),
        ("EGI".into(),     t(&[p1(), sg(), perf(), ind(), act(), c3()])),
        ("EGISTI".into(),  t(&[p2(), sg(), perf(), ind(), act(), c3()])),
        ("EGIT".into(),    t(&[p3(), sg(), perf(), ind(), act(), c3()])),

        // ─── 3rd Conjugation: ducere (to lead) ─────────────────────
        ("DUCO".into(),     t(&[p1(), sg(), pres(), ind(), act(), c3()])),
        ("DUCIS".into(),    t(&[p2(), sg(), pres(), ind(), act(), c3()])),
        ("DUCIT".into(),    t(&[p3(), sg(), pres(), ind(), act(), c3()])),
        ("DUCIMUS".into(),  t(&[p1(), pl(), pres(), ind(), act(), c3()])),
        ("DUCITIS".into(),  t(&[p2(), pl(), pres(), ind(), act(), c3()])),
        ("DUCUNT".into(),   t(&[p3(), pl(), pres(), ind(), act(), c3()])),
        ("DUXI".into(),     t(&[p1(), sg(), perf(), ind(), act(), c3()])),
        ("DUXISTI".into(),  t(&[p2(), sg(), perf(), ind(), act(), c3()])),
        ("DUXIT".into(),    t(&[p3(), sg(), perf(), ind(), act(), c3()])),
    ]
}

/// Unseen words for generalization testing — these do NOT appear in training.
pub fn unseen_words() -> Vec<TrainingExample> {
    vec![
        ("NARRAT".into(),    make(&[p3(), sg(), pres(), ind(), act(), c1()])),
        ("NARRANT".into(),   make(&[p3(), pl(), pres(), ind(), act(), c1()])),
        ("NARRABAT".into(),  make(&[p3(), sg(), impf(), ind(), act(), c1()])),
        ("VIDET".into(),     make(&[p3(), sg(), pres(), ind(), act(), c2()])),
        ("VIDENT".into(),    make(&[p3(), pl(), pres(), ind(), act(), c2()])),
        ("VIDEBAT".into(),   make(&[p3(), sg(), impf(), ind(), act(), c2()])),
        ("SCRIBIT".into(),   make(&[p3(), sg(), pres(), ind(), act(), c3()])),
        ("SCRIBUNT".into(),  make(&[p3(), pl(), pres(), ind(), act(), c3()])),
    ]
}
