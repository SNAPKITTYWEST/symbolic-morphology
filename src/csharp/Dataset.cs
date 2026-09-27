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
/// Latin verb morphology corpus.
/// Each entry maps a word form to its Boolean grammatical feature vector.
/// </summary>
public static class Dataset
{
    public static (string Word, double[] Target)[] BuildCorpus()
    {
        var corpus = new List<(string, double[])>();

        // ─── 1st Conjugation: amare (to love) ──────────────────────
        // Present indicative active
        Add(corpus, "amo",    "PERSON_1", "SINGULAR", "PRESENT",    "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amas",   "PERSON_2", "SINGULAR", "PRESENT",    "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amat",   "PERSON_3", "SINGULAR", "PRESENT",    "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amamus", "PERSON_1", "PLURAL",   "PRESENT",    "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amatis", "PERSON_2", "PLURAL",   "PRESENT",    "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amant",  "PERSON_3", "PLURAL",   "PRESENT",    "INDICATIVE", "ACTIVE", "CONJ_1");

        // Imperfect indicative active
        Add(corpus, "amabam",  "PERSON_1", "SINGULAR", "IMPERFECT", "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amabas",  "PERSON_2", "SINGULAR", "IMPERFECT", "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amabat",  "PERSON_3", "SINGULAR", "IMPERFECT", "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amabamus","PERSON_1", "PLURAL",   "IMPERFECT", "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amabatis","PERSON_2", "PLURAL",   "IMPERFECT", "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amabant", "PERSON_3", "PLURAL",   "IMPERFECT", "INDICATIVE", "ACTIVE", "CONJ_1");

        // Future indicative active
        Add(corpus, "amabo",  "PERSON_1", "SINGULAR", "FUTURE",    "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amabis", "PERSON_2", "SINGULAR", "FUTURE",    "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amabit", "PERSON_3", "SINGULAR", "FUTURE",    "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amabimus","PERSON_1","PLURAL",   "FUTURE",    "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amabitis","PERSON_2","PLURAL",   "FUTURE",    "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amabunt", "PERSON_3","PLURAL",   "FUTURE",    "INDICATIVE", "ACTIVE", "CONJ_1");

        // Perfect indicative active
        Add(corpus, "amavi",    "PERSON_1", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amavisti", "PERSON_2", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amavit",   "PERSON_3", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amavimus", "PERSON_1", "PLURAL",   "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amavistis","PERSON_2", "PLURAL",   "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "amaverunt","PERSON_3", "PLURAL",   "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_1");

        // ─── 1st Conjugation: laudare (to praise) ──────────────────
        Add(corpus, "laudo",      "PERSON_1", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudas",     "PERSON_2", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudat",     "PERSON_3", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudamus",   "PERSON_1", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudatis",   "PERSON_2", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudant",    "PERSON_3", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudabam",   "PERSON_1", "SINGULAR", "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudabas",   "PERSON_2", "SINGULAR", "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudabat",   "PERSON_3", "SINGULAR", "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudabamus", "PERSON_1", "PLURAL",   "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudabatis", "PERSON_2", "PLURAL",   "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudabant",  "PERSON_3", "PLURAL",   "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudavi",    "PERSON_1", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudavisti", "PERSON_2", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_1");
        Add(corpus, "laudavit",   "PERSON_3", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_1");

        // ─── 2nd Conjugation: monere (to warn) ─────────────────────
        Add(corpus, "moneo",    "PERSON_1", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "mones",    "PERSON_2", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "monet",    "PERSON_3", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "monemus",  "PERSON_1", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "monetis",  "PERSON_2", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "monent",   "PERSON_3", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "monebam",  "PERSON_1", "SINGULAR", "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "monebas",  "PERSON_2", "SINGULAR", "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "monebat",  "PERSON_3", "SINGULAR", "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "monebamus","PERSON_1", "PLURAL",   "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "monebatis","PERSON_2", "PLURAL",   "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "monebant", "PERSON_3", "PLURAL",   "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "monui",    "PERSON_1", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "monuisti", "PERSON_2", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "monuit",   "PERSON_3", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_2");

        // ─── 2nd Conjugation: habere (to have) ─────────────────────
        Add(corpus, "habeo",    "PERSON_1", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "habes",    "PERSON_2", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "habet",    "PERSON_3", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "habemus",  "PERSON_1", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "habetis",  "PERSON_2", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "habent",   "PERSON_3", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "habui",    "PERSON_1", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "habuisti", "PERSON_2", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_2");
        Add(corpus, "habuit",   "PERSON_3", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_2");

        // ─── 3rd Conjugation: regere (to rule) ─────────────────────
        Add(corpus, "rego",     "PERSON_1", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "regis",    "PERSON_2", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "regit",    "PERSON_3", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "regimus",  "PERSON_1", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "regitis",  "PERSON_2", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "regunt",   "PERSON_3", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "regebam",  "PERSON_1", "SINGULAR", "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "reuebas",  "PERSON_2", "SINGULAR", "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "regebat",  "PERSON_3", "SINGULAR", "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "regebamus","PERSON_1", "PLURAL",   "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "regebatis","PERSON_2", "PLURAL",   "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "regebant", "PERSON_3", "PLURAL",   "IMPERFECT","INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "rexi",     "PERSON_1", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "rexisti",  "PERSON_2", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "rexit",    "PERSON_3", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_3");

        // ─── 3rd Conjugation: agere (to do/drive) ──────────────────
        Add(corpus, "ago",      "PERSON_1", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "agis",     "PERSON_2", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "agit",     "PERSON_3", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "agimus",   "PERSON_1", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "agitis",   "PERSON_2", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "agunt",    "PERSON_3", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "egi",      "PERSON_1", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "egisti",   "PERSON_2", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "egit",     "PERSON_3", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_3");

        // ─── 3rd Conjugation: ducere (to lead) ─────────────────────
        Add(corpus, "duco",     "PERSON_1", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "ducis",    "PERSON_2", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "ducit",    "PERSON_3", "SINGULAR", "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "ducimus",  "PERSON_1", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "ducitis",  "PERSON_2", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "ducunt",   "PERSON_3", "PLURAL",   "PRESENT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "duxi",     "PERSON_1", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "duxisti",  "PERSON_2", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_3");
        Add(corpus, "duxit",    "PERSON_3", "SINGULAR", "PERFECT",  "INDICATIVE", "ACTIVE", "CONJ_3");

        return corpus.ToArray();
    }

    static void Add(List<(string, double[])> corpus, string word,
        string person, string number, string tense, string mood, string voice, string conjugation)
    {
        corpus.Add((word.ToUpperInvariant(), Features.MakeTarget(
            ("PERSON", person),
            ("NUMBER", number),
            ("TENSE", tense),
            ("MOOD", mood),
            ("VOICE", voice),
            ("CONJUGATION", conjugation)
        )));
    }

    /// <summary>
    /// Unseen words for generalization testing.
    /// These verbs do NOT appear in the training corpus.
    /// </summary>
    public static (string Word, double[] Target)[] UnseenWords()
    {
        return new (string, double[])[]
        {
            ("NARRAT", Features.MakeTarget(("PERSON","PERSON_3"),("NUMBER","SINGULAR"),("TENSE","PRESENT"),("MOOD","INDICATIVE"),("VOICE","ACTIVE"),("CONJUGATION","CONJ_1"))),
            ("NARRANT", Features.MakeTarget(("PERSON","PERSON_3"),("NUMBER","PLURAL"),("TENSE","PRESENT"),("MOOD","INDICATIVE"),("VOICE","ACTIVE"),("CONJUGATION","CONJ_1"))),
            ("NARRABAT", Features.MakeTarget(("PERSON","PERSON_3"),("NUMBER","SINGULAR"),("TENSE","IMPERFECT"),("MOOD","INDICATIVE"),("VOICE","ACTIVE"),("CONJUGATION","CONJ_1"))),
            ("VIDET", Features.MakeTarget(("PERSON","PERSON_3"),("NUMBER","SINGULAR"),("TENSE","PRESENT"),("MOOD","INDICATIVE"),("VOICE","ACTIVE"),("CONJUGATION","CONJ_2"))),
            ("VIDENT", Features.MakeTarget(("PERSON","PERSON_3"),("NUMBER","PLURAL"),("TENSE","PRESENT"),("MOOD","INDICATIVE"),("VOICE","ACTIVE"),("CONJUGATION","CONJ_2"))),
            ("VIDEBAT", Features.MakeTarget(("PERSON","PERSON_3"),("NUMBER","SINGULAR"),("TENSE","IMPERFECT"),("MOOD","INDICATIVE"),("VOICE","ACTIVE"),("CONJUGATION","CONJ_2"))),
            ("SCRIBIT", Features.MakeTarget(("PERSON","PERSON_3"),("NUMBER","SINGULAR"),("TENSE","PRESENT"),("MOOD","INDICATIVE"),("VOICE","ACTIVE"),("CONJUGATION","CONJ_3"))),
            ("SCRIBUNT", Features.MakeTarget(("PERSON","PERSON_3"),("NUMBER","PLURAL"),("TENSE","PRESENT"),("MOOD","INDICATIVE"),("VOICE","ACTIVE"),("CONJUGATION","CONJ_3"))),
        };
    }
}
