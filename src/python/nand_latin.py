#!/usr/bin/env python3
# Symbolic Morphology Engine — Latin verb morphology from raw letters to Boolean grammar
# Copyright (C) 2026 Ahmad Ali Parr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# -*- coding: utf-8 -*-
"""
nand_latin.py -- a Latin morphology learner with a single primitive: NAND.
"""

import math
import random
import time

# ============================================================================
# LEVEL 0 -- THE ONLY PRIMITIVE
# ============================================================================

def NAND(a, b):
    """a, b in {0,1} -> {0,1}. The single irreducible leaf of the graph."""
    return 0 if (a == 1 and b == 1) else 1

def NOT(a): return NAND(a, a)
def AND(a, b): return NOT(NAND(a, b))
def OR(a, b): return NAND(NOT(a), NOT(b))

def XOR(a, b):
    t = NAND(a, b)
    return NAND(NAND(a, t), NAND(b, t))

def MUX(sel, a, b):
    """sel=1 -> a, sel=0 -> b."""
    return OR(AND(sel, a), AND(NOT(sel), b))

# ============================================================================
# LEVEL 1 -- BIT-PARALLEL GATES
# ============================================================================

W = 16
F = 8
_M = (1 << W) - 1

def _mask(w): return (1 << w) - 1

def NAND_W(a, b, w=W):
    return (~(a & b)) & _mask(w)
def NOT_W(a, w=W): return NAND_W(a, a, w)
def AND_W(a, b, w=W): return NOT_W(NAND_W(a, b, w), w)
def OR_W(a, b, w=W): return NAND_W(NOT_W(a, w), NOT_W(b, w), w)
def XOR_W(a, b, w=W):
    t = NAND_W(a, b, w)
    return NAND_W(NAND_W(a, t, w), NAND_W(b, t, w), w)

def MUX_W(sel, a, b, w=W):
    m = _mask(w)
    s = (-sel) & m
    return ((a & s) | (b & (~s) & m)) & m

# ============================================================================
# LEVEL 2 -- ARITHMETIC, BUILT FROM THE GATES ABOVE
# ============================================================================

def ADD_W(a, b, w=W):
    """Kogge-Stone carry-lookahead adder."""
    m = _mask(w)
    g = AND_W(a, b, w)
    p = XOR_W(a, b, w)
    d = 1
    while d < w:
        g = OR_W(g, AND_W(p, (g << d) & m, w), w)
        p = AND_W(p, (p << d) & m, w)
        d <<= 1
    carry = (g << 1) & m
    return XOR_W(XOR_W(a, b, w), carry, w)

def NEG_W(a, w=W):
    """two's complement: -a = ~a + 1"""
    return ADD_W(NOT_W(a, w), 1, w)

def SUB_W(a, b, w=W):
    return ADD_W(a, NEG_W(b, w), w)

def UMUL_W(a, b, w=W):
    """Unsigned w x w -> 2w shift-add multiplier, built from ADD_W."""
    m2 = (1 << (2 * w)) - 1
    res = 0
    i = 0
    bb = b
    while bb:
        if bb & 1:
            res = ADD_W(res, (a << i) & m2, 2 * w)
        bb >>= 1
        i += 1
    return res

# ---- fixed-point Q(16-8).8 -------------------------------------------------

def FP(x):
    """float -> Q(16.8) two's-complement int."""
    v = int(round(x * (1 << F)))
    lim = 1 << (W - 1)
    if v >= lim: v = lim - 1
    if v < -lim: v = -lim
    return v & _M

def to_signed(a):
    return a - (1 << W) if (a >> (W - 1)) & 1 else a

def to_float(a):
    return to_signed(a) / float(1 << F)

def SAR_W(a, n):
    """arithmetic shift right (signed semantics)."""
    return (to_signed(a) >> n) & _M

def MUL_W(a, b):
    """signed Q(16.8) multiply."""
    sa = to_signed(a)
    sb = to_signed(b)
    neg = (sa < 0) != (sb < 0)
    ua = -sa if sa < 0 else sa
    ub = -sb if sb < 0 else sb
    p = (UMUL_W(ua, ub) >> F) & _M
    return NEG_W(p) if neg else p

# ============================================================================
# LEVEL 3 -- TRANSCENDENTALS (comparators + MUX interpolation, from NAND)
# ============================================================================

_TX = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
_TY = [math.tanh(x) for x in _TX]
_TXf = [FP(x) for x in _TX]
_TYf = [FP(y) for y in _TY]
_TS = [FP((_TY[i + 1] - _TY[i]) / (_TX[i + 1] - _TX[i]))
        for i in range(len(_TX) - 1)]

def SLT_W(a, b):
    """signed a < b -- sign bit of (a - b)."""
    return (SUB_W(a, b) >> (W - 1)) & 1

def tanh_fp(x):
    if to_signed(x) < to_signed(_TXf[0]): x = _TXf[0]
    elif to_signed(x) > to_signed(_TXf[-1]): x = _TXf[-1]
    idx = 0
    for i in range(len(_TXf) - 1):
        if to_signed(x) >= to_signed(_TXf[i]):
            idx = i
    dx = SUB_W(x, _TXf[idx])
    return ADD_W(_TYf[idx], MUL_W(_TS[idx], dx))

_HALF = FP(0.5)

def sigmoid_fp(z):
    """sigmoid(z) = 0.5 + 0.5 * tanh(z/2)"""
    return ADD_W(_HALF, SAR_W(tanh_fp(SAR_W(z, 1)), 1))

# ============================================================================
# LEVEL 4 -- SYMBOLIC OUTPUT SPACE (10 Boolean grammatical bits)
# ============================================================================

FEATURES = [
    "PERSON_B0", "PERSON_B1",
    "NUMBER",
    "TENSE_B0", "TENSE_B1",
    "MOOD_B0", "MOOD_B1",
    "VOICE",
    "CONJ_B0", "CONJ_B1",
]
K = len(FEATURES)

_PERSON = {1: (0, 0), 2: (0, 1), 3: (1, 0)}
_TENSE = {'PRESENT': (0, 0), 'IMPERFECT': (0, 1),
           'FUTURE': (1, 0), 'PERFECT': (1, 1)}
_MOOD = {'INDICATIVE': (0, 0), 'SUBJUNCTIVE': (0, 1), 'IMPERATIVE': (1, 0)}
_CONJ = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}

def target_bits(person, number, tense, mood, voice, conj):
    pb = _PERSON[person]
    tb = _TENSE[tense]
    mb = _MOOD[mood]
    cb = _CONJ[conj]
    return [
        pb[0], pb[1],
        1 if number == 'PL' else 0,
        tb[0], tb[1],
        mb[0], mb[1],
        1 if voice == 'PASSIVE' else 0,
        cb[0], cb[1],
    ]

def decode(p):
    b = [1 if to_float(v) >= 0.5 else 0 for v in p]
    person = {(0, 0): 1, (0, 1): 2, (1, 0): 3}.get((b[0], b[1]), '?')
    number = 'PL' if b[2] else 'SG'
    tense = {(0, 0): 'PRES', (0, 1): 'IMPF',
              (1, 0): 'FUT', (1, 1): 'PERF'}.get((b[3], b[4]), '?')
    mood = {(0, 0): 'IND', (0, 1): 'SUBJ',
              (1, 0): 'IMP'}.get((b[5], b[6]), '?')
    voice = 'PASS' if b[7] else 'ACT'
    conj = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}.get((b[8], b[9]), '?')
    return person, number, tense, mood, voice, conj

# ============================================================================
# LEVEL 5 -- CORPUS
# ============================================================================

PARADIGMS = {
    1: ("am", {"PRESENT": ["o", "as", "at", "amus", "atis", "ant"],
                "IMPERFECT": ["abam", "abas", "abat", "abamus", "abatis", "abant"]}),
    2: ("mon", {"PRESENT": ["eo", "es", "et", "emus", "etis", "ent"],
                "IMPERFECT": ["ebam", "ebas", "ebat", "ebamus", "ebatis", "ebant"]}),
    3: ("reg", {"PRESENT": ["o", "is", "it", "imus", "itis", "unt"],
                "IMPERFECT": ["ebam", "ebas", "ebat", "ebamus", "ebatis", "ebant"]}),
    4: ("aud", {"PRESENT": ["io", "is", "it", "imus", "itis", "iunt"],
                "IMPERFECT": ["iebam","iebas","iebat","iebamus","iebatis","iebant"]}),
}

def pn(i):
    return (i % 3) + 1, ('SG' if i < 3 else 'PL')

def build_corpus():
    corpus = []
    for conj, (stem, tenses) in PARADIGMS.items():
        for tense, endings in tenses.items():
            for i, end in enumerate(endings):
                p, n = pn(i)
                w = stem + end
                corpus.append({
                    "word": w,
                    "y": target_bits(p, n, tense, 'INDICATIVE', 'ACTIVE', conj),
                    "person": p, "number": n,
                    "tense": tense,
                    "mood": 'INDICATIVE', "voice": 'ACTIVE', "conj": conj,
                })
    return corpus

# ============================================================================
# LEVEL 6 -- MODEL (embeddings + Elman RNN + linear output)
# ============================================================================

def rand_fp(rng, scale):
    return FP(rng.uniform(-scale, scale))

def init_model(vocab, D=3, H=4, seed=7):
    rng = random.Random(seed)
    V = len(vocab)
    def mat(r, c, s): return [[rand_fp(rng, s) for _ in range(c)] for _ in range(r)]
    def vec(n, s): return [rand_fp(rng, s) for _ in range(n)]
    return {
        'E': mat(V, D, 0.8),
        'Wxh': mat(H, D, 0.8),
        'Whh': mat(H, H, 0.5),
        'bh': vec(H, 0.15),
        'Why': mat(K, H, 0.8),
        'by': vec(K, 0.15),
        'vocab': vocab, 'D': D, 'H': H, 'K': K,
    }

def forward(m, word):
    E, Wxh, Whh, bh = m['E'], m['Wxh'], m['Whh'], m['bh']
    Why, by = m['Why'], m['by']
    H, D, K, vocab = m['H'], m['D'], m['K'], m['vocab']

    h = [0] * H
    cache = []
    for ch in word:
        ci = vocab[ch]
        x = E[ci]
        pre = []
        for j in range(H):
            s = bh[j]
            for d in range(D): s = ADD_W(s, MUL_W(Wxh[j][d], x[d]))
            for k in range(H): s = ADD_W(s, MUL_W(Whh[j][k], h[k]))
            pre.append(s)
        h_new = [tanh_fp(p) for p in pre]
        cache.append((ci, x, h, h_new))
        h = h_new

    logits = []
    for k in range(K):
        s = by[k]
        for j in range(H):
            s = ADD_W(s, MUL_W(Why[k][j], h[j]))
        logits.append(s)
    return h, logits, cache

def predict(m, word):
    _, logits, _ = forward(m, word)
    return [sigmoid_fp(z) for z in logits]

# ============================================================================
# LEVEL 7 -- HAND-ROLLED BACKPROPAGATION THROUGH TIME
# ============================================================================

def backward(m, word, y):
    H, D, K = m['H'], m['D'], m['K']
    h_final, logits, cache = forward(m, word)
    p = [sigmoid_fp(z) for z in logits]

    invK = FP(1.0 / K)
    dlogit = []
    for k in range(K):
        diff = SUB_W(p[k], FP(float(y[k])))
        dlogit.append(MUL_W(diff, invK))

    V = len(m['vocab'])
    gE = [[0] * D for _ in range(V)]
    gWxh = [[0] * D for _ in range(H)]
    gWhh = [[0] * H for _ in range(H)]
    gbh = [0] * H
    gWhy = [[0] * H for _ in range(K)]
    gby = [0] * K

    for k in range(K):
        dk = dlogit[k]
        for j in range(H):
            gWhy[k][j] = ADD_W(gWhy[k][j], MUL_W(dk, h_final[j]))
        gby[k] = ADD_W(gby[k], dk)

    dh = [0] * H
    for j in range(H):
        acc = 0
        for k in range(K):
            acc = ADD_W(acc, MUL_W(m['Why'][k][j], dlogit[k]))
        dh[j] = acc

    for t in range(len(cache) - 1, -1, -1):
        ci, x, h_prev, h_new = cache[t]

        dz = []
        for j in range(H):
            t2 = MUL_W(h_new[j], h_new[j])
            dz.append(MUL_W(dh[j], SUB_W(FP(1.0), t2)))

        for j in range(H):
            dj = dz[j]
            for d in range(D):
                gWxh[j][d] = ADD_W(gWxh[j][d], MUL_W(dj, x[d]))
            for kk in range(H):
                gWhh[j][kk] = ADD_W(gWhh[j][kk], MUL_W(dj, h_prev[kk]))
            gbh[j] = ADD_W(gbh[j], dj)

        new_dh = [0] * H
        for kk in range(H):
            acc = 0
            for j in range(H):
                acc = ADD_W(acc, MUL_W(m['Whh'][j][kk], dz[j]))
            new_dh[kk] = acc
        dh = new_dh

        for d in range(D):
            acc = 0
            for j in range(H):
                acc = ADD_W(acc, MUL_W(m['Wxh'][j][d], dz[j]))
            gE[ci][d] = ADD_W(gE[ci][d], acc)

    return p, (gE, gWxh, gWhh, gbh, gWhy, gby)

def sgd_step(m, grads, lr_fp):
    gE, gWxh, gWhh, gbh, gWhy, gby = grads
    def upd(P, G):
        for i in range(len(P)):
            if isinstance(P[i], list):
                for j in range(len(P[i])):
                    P[i][j] = SUB_W(P[i][j], MUL_W(lr_fp, G[i][j]))
            else:
                P[i] = SUB_W(P[i], MUL_W(lr_fp, G[i]))
    upd(m['E'], gE)
    upd(m['Wxh'], gWxh)
    upd(m['Whh'], gWhh)
    upd(m['bh'], gbh)
    upd(m['Why'], gWhy)
    upd(m['by'], gby)

# ============================================================================
# LEVEL 8 -- LOSS, TRAINING, EVALUATION, DISPLAY
# ============================================================================

def loss_of(p, y):
    s = 0.0
    for k in range(K):
        pk = max(1e-7, min(1 - 1e-7, to_float(p[k])))
        s += -(y[k] * math.log(pk) + (1 - y[k]) * math.log(1 - pk))
    return s / K

def train(m, corpus, epochs, lr_start=1.0, lr_end=0.05,
          log_every=5, seed=1):
    rng = random.Random(seed)
    data = list(corpus)
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        rng.shuffle(data)
        f = (ep - 1) / max(1, epochs - 1)
        lr = lr_start * (1 - f) + lr_end * f
        lr_fp = FP(lr)
        tot = 0.0
        for ex in data:
            p, grads = backward(m, ex['word'], ex['y'])
            tot += loss_of(p, ex['y'])
            sgd_step(m, grads, lr_fp)
        avg = tot / len(data)
        hist.append(avg)
        if ep == 1 or ep % log_every == 0 or ep == epochs:
            print(" epoch %3d | lr=%.3f | mean BCE=%.5f | %.1fs"
                  % (ep, lr, avg, time.time() - t0))
    return hist

def evaluate(m, data):
    if not data:
        return {"n": 0}
    exact = bit_ok = bit_tot = 0
    L = 0.0
    crisp = 0
    cells = 0
    for ex in data:
        p = predict(m, ex['word'])
        L += loss_of(p, ex['y'])
        ok = True
        for k in range(K):
            v = to_float(p[k])
            b = 1 if v >= 0.5 else 0
            if b == int(ex['y'][k]): bit_ok += 1
            else: ok = False
            bit_tot += 1
            cells += 1
            if abs(v - 0.5) > 0.45: crisp += 1
        if ok: exact += 1
    return {"n": len(data),
            "exact": exact / len(data),
            "bit": bit_ok / bit_tot,
            "loss": L / len(data),
            "crisp": crisp / cells}

def show(m, word, target=None, title=None):
    p = predict(m, word)
    if title:
        print("\n" + "=" * 76)
        print(title)
        print("=" * 76)
    print("INPUT: %s" % word.upper())
    print()
    print(" %-12s %-10s %-7s%s" % ("FEATURE", "RAW", "BOOL",
          " TARGET ERR" if target is not None else ""))
    print(" " + "-" * (44 if target is not None else 30))
    for k, f in enumerate(FEATURES):
        v = to_float(p[k])
        line = " %-12s %-10.4f %-7s" % (f, v, "TRUE" if v >= 0.5 else "FALSE")
        if target is not None:
            tv = "TRUE" if target[k] >= 0.5 else "FALSE"
            line += " %-6s %+.4f" % (tv, v - target[k])
        print(line)
    if target is not None:
        L = loss_of(p, target)
        print("\n LOSS (mean BCE) : %.6f" % L)
        _, grads = backward(m, word, target)
        s = 0.0
        for G in grads:
            for row in G:
                if isinstance(row, list):
                    for v in row: s += to_float(v) ** 2
                else:
                    s += to_float(row) ** 2
        print(" GRADIENT ||g||_2 : %.6f" % math.sqrt(s))
    p_, n_, t_, mo, vo, cj = decode(p)
    print("\n READ AS: person=%s number=%s tense=%s mood=%s voice=%s conj=%s"
          % (p_, n_, t_, mo, vo, cj))

# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 76)
    print("NAND-RECURSIVE LATIN MORPHOLOGY LEARNER")
    print("Every computation bottoms out at a single NAND gate.")
    print("=" * 76)

    corpus = build_corpus()
    vocab = {c: i for i, c in enumerate(sorted(set("".join(e['word'] for e in corpus))))}

    print("\nCORPUS")
    print(" examples : %d" % len(corpus))
    print(" vocab : %s" % "".join(sorted(vocab)))
    print(" features : %d Boolean grammatical bits" % K)

    rng = random.Random(23)
    words = sorted({e['word'] for e in corpus})
    unseen_words = set(rng.sample(words, max(2, len(words) // 6)))
    train_data = [e for e in corpus if e['word'] not in unseen_words]
    test_data = [e for e in corpus if e['word'] in unseen_words]

    print(" train : %d" % len(train_data))
    print(" test : %d (unseen inflected forms)" % len(test_data))

    m = init_model(vocab, D=3, H=4, seed=7)
    npar = sum(len(r) if isinstance(r[0], list) else 1
               for r in m.values() if isinstance(r, list))
    print("\nMODEL")
    print(" D=%d H=%d K=%d params=%d fixed-point Q(16.8)"
          % (m['D'], m['H'], m['K'], npar))

    print("\nTRAINING")
    print("-" * 76)
    hist = train(m, train_data, epochs=40, lr_start=1.0, lr_end=0.05, log_every=5)
    print("-" * 76)
    print(" final train BCE : %.6f" % hist[-1])
    print(" wall clock : %.1f s" % (time.time() - t0))

    by_word = {e['word']: e for e in corpus}
    for w, ttl in [("amo", "EXAMPLE 1 (1sg present indicative active, 1st conj)"),
                   ("amat", "EXAMPLE 2 (3sg present indicative active, 1st conj)"),
                   ("regebat", "EXAMPLE 3 (3sg imperfect indicative active, 3rd conj)"),
                   ("audiunt", "EXAMPLE 4 (3pl present indicative active, 4th conj)")]:
        show(m, w, by_word[w]['y'], ttl)

    print("\n" + "=" * 76)
    print("UNSEEN INFLECTED FORMS")
    print("=" * 76)
    for e in test_data:
        show(m, e['word'], e['y'])

    print("\n" + "=" * 76)
    print("EVALUATION")
    print("=" * 76)
    for name, data in [("seen (train)", train_data), ("unseen words", test_data)]:
        r = evaluate(m, data)
        print(" %-16s n=%2d | exact=%5.1f%% | bit-acc=%5.1f%% | BCE=%.5f | crisp=%5.1f%%"
              % (name, r['n'], 100 * r['exact'], 100 * r['bit'],
                 r['loss'], 100 * r['crisp']))

    print("\n" + "=" * 76)
    print("BOOLEAN CONVERGENCE")
    print("=" * 76)
    buckets = [0] * 10
    tot = 0
    for e in corpus:
        p = predict(m, e['word'])
        for v in p:
            d = abs(to_float(v) - 0.5)
            buckets[min(9, int(d * 20))] += 1
            tot += 1
    for i, c in enumerate(buckets):
        bar = "#" * int(60 * c / max(1, tot))
        print(" %.2f-%.2f : %5.2f%% %s"
              % (i * 0.05, (i + 1) * 0.05, 100 * c / tot, bar))

    print("\nDONE in %.1f s" % (time.time() - t0))

if __name__ == "__main__":
    main()
