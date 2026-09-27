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

"""
Burt's symbolic Latin morphology learner — float-based Elman RNN.
Benchmark version: measures training time, inference latency, quality.
"""
import math, random, time

# ── Primitives ──────────────────────────────────────────────────────
def zeros(n): return [0.0] * n
def zeros2(r, c): return [[0.0] * c for _ in range(r)]

def sigmoid(z):
    if z >= 0.0:
        e = math.exp(-z); return 1.0 / (1.0 + e)
    e = math.exp(z); return e / (1.0 + e)

def matTvec(M, v):
    c = len(M[0]); out = [0.0] * c
    for i in range(len(M)):
        vi = v[i]
        if vi != 0.0:
            row = M[i]
            for j in range(c): out[j] += row[j] * vi
    return out

def outer_add(M, a, b):
    for i in range(len(a)):
        ai = a[i]
        if ai != 0.0:
            row = M[i]
            for j in range(len(b)): row[j] += ai * b[j]

# ── Features ────────────────────────────────────────────────────────
FEATURES = ["PERSON_1","PERSON_2","PERSON_3","SINGULAR","PLURAL",
            "PRESENT","IMPERFECT","FUTURE","PERFECT",
            "INDICATIVE","SUBJUNCTIVE","IMPERATIVE","ACTIVE","PASSIVE"]
K = len(FEATURES)
FIDX = {f: i for i, f in enumerate(FEATURES)}

def make_target(person, number, tense, mood, voice):
    y = [0.0] * K
    y[FIDX["PERSON_%d" % person]] = 1.0
    y[FIDX["SINGULAR" if number == "SG" else "PLURAL"]] = 1.0
    y[FIDX[tense]] = 1.0; y[FIDX[mood]] = 1.0; y[FIDX[voice]] = 1.0
    return y

# ── Corpus (same structure as Burt's) ───────────────────────────────
PRES_IND_ACT = {
    1: ["o","as","at","amus","atis","ant"],
    2: ["eo","es","et","emus","etis","ent"],
    3: ["o","is","it","imus","itis","unt"],
    4: ["io","is","it","imus","itis","iunt"],
}
IMPF_IND_ACT = {
    1: ["abam","abas","abat","abamus","abatis","abant"],
    2: ["ebam","ebas","ebat","ebamus","ebatis","ebant"],
    3: ["ebam","ebas","ebat","ebamus","ebatis","ebant"],
    4: ["iebam","iebas","iebat","iebamus","iebatis","iebant"],
}
PERF_IND_ACT = ["i","isti","it","imus","istis","erunt"]

VERBS = [
    (1,"am","amav"), (1,"laud","laudav"), (1,"port","portav"), (1,"voc","vocav"),
    (2,"mon","monu"), (2,"vid","vid"),
    (3,"reg","rex"), (3,"duc","dux"), (3,"mitt","mis"),
    (4,"aud","audiv"), (4,"ven","ven"),
]

def pn(i): return (i % 3) + 1, ("SG" if i < 3 else "PL")

def build_corpus():
    corpus = []
    for conj, stem, pstem in VERBS:
        for i, e in enumerate(PRES_IND_ACT[conj]):
            p, n = pn(i)
            corpus.append((stem+e, make_target(p,n,"PRESENT","INDICATIVE","ACTIVE")))
        for i, e in enumerate(IMPF_IND_ACT[conj]):
            p, n = pn(i)
            corpus.append((stem+e, make_target(p,n,"IMPERFECT","INDICATIVE","ACTIVE")))
        for i, e in enumerate(PERF_IND_ACT):
            p, n = pn(i)
            corpus.append((pstem+e, make_target(p,n,"PERFECT","INDICATIVE","ACTIVE")))
    return corpus

# ── Model: Elman RNN ────────────────────────────────────────────────
class MorphRNN:
    def __init__(self, vocab, D=8, H=16, seed=7):
        rng = random.Random(seed)
        self.vocab = vocab; self.D = D; self.H = H
        s = lambda: rng.gauss(0, 0.4)
        self.E = [[s() for _ in range(D)] for _ in range(len(vocab))]
        sc_x = 1.0/math.sqrt(D); sc_h = 1.0/math.sqrt(H)
        self.Wxh = [[rng.gauss(0,sc_x) for _ in range(D)] for _ in range(H)]
        self.Whh = [[rng.gauss(0,sc_h) for _ in range(H)] for _ in range(H)]
        self.bh = zeros(H)
        self.Why = [[rng.gauss(0,sc_h) for _ in range(H)] for _ in range(K)]
        self.by = zeros(K)

    def forward(self, word):
        E,Wxh,Whh,bh,Why,by = self.E,self.Wxh,self.Whh,self.bh,self.Why,self.by
        H,D = self.H,self.D
        h = zeros(H); cache = []
        for ch in word:
            ci = self.vocab[ch]; x = E[ci]
            pre = [0.0]*H
            for j in range(H):
                s = bh[j]
                for d in range(D): s += Wxh[j][d]*x[d]
                for k in range(H): s += Whh[j][k]*h[k]
                pre[j] = s
            h_new = [math.tanh(v) for v in pre]
            cache.append((ci,x,h,h_new)); h = h_new
        logits = [by[k] + sum(Why[k][j]*h[j] for j in range(H)) for k in range(K)]
        return h, logits, cache

    def backward(self, word, y, grads):
        H,D = self.H,self.D
        h,logits,cache = self.forward(word)
        p = [sigmoid(z) for z in logits]
        loss = sum(-(y[k]*math.log(max(1e-12,p[k]))+(1-y[k])*math.log(max(1e-12,1-p[k]))) for k in range(K))/K
        dlogits = [(p[k]-y[k])/K for k in range(K)]
        outer_add(grads["Why"], dlogits, h)
        for k in range(K): grads["by"][k] += dlogits[k]
        dh = matTvec(self.Why, dlogits)
        gE,gWxh,gWhh,gbh = grads["E"],grads["Wxh"],grads["Whh"],grads["bh"]
        for t in range(len(cache)-1,-1,-1):
            ci,x,h_prev,h_new = cache[t]
            dz = [dh[j]*(1.0-h_new[j]*h_new[j]) for j in range(H)]
            for j in range(H):
                d = dz[j]
                if d == 0.0: continue
                for dd in range(D): gWxh[j][dd] += d*x[dd]
                for k in range(H): gWhh[j][k] += d*h_prev[k]
                gbh[j] += d
            dh = matTvec(self.Whh, dz)
            dE = matTvec(self.Wxh, dz)
            for dd in range(D): gE[ci][dd] += dE[dd]
        return loss, p

    def step(self, grads, lr):
        for P,G in [(self.E,grads["E"]),(self.Wxh,grads["Wxh"]),(self.Whh,grads["Whh"]),
                     (self.Why,grads["Why"])]:
            for i in range(len(P)):
                for j in range(len(P[i])): P[i][j] -= lr*G[i][j]
        for i in range(len(self.bh)): self.bh[i] -= lr*grads["bh"][i]
        for i in range(len(self.by)): self.by[i] -= lr*grads["by"][i]

    def zero_grads(self):
        return {"E":zeros2(len(self.vocab),self.D),"Wxh":zeros2(self.H,self.D),
                "Whh":zeros2(self.H,self.H),"bh":zeros(self.H),
                "Why":zeros2(K,self.H),"by":zeros(K)}

# ── Benchmark ───────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 65)
    print("PYTHON FLOAT-BASED RNN — BENCHMARK")
    print("=" * 65)

    corpus = build_corpus()
    all_chars = sorted(set("".join(w for w,_ in corpus)))
    vocab = {c:i for i,c in enumerate(all_chars)}
    print(f"  Corpus: {len(corpus)} words, {K} features, vocab: {len(vocab)} chars")

    model = MorphRNN(vocab, D=8, H=16, seed=7)
    nparam = sum(len(r) for row in model.E for r in [row]) + \
             sum(len(r) for row in model.Wxh for r in [row]) + \
             sum(len(r) for row in model.Whh for r in [row]) + \
             len(model.bh) + \
             sum(len(r) for row in model.Why for r in [row]) + \
             len(model.by)
    print(f"  Params: D=8, H=16, K={K}, total={nparam}")
    print()

    # BENCHMARK 1: Training
    EPOCHS = 3000
    print(f"  BENCHMARK 1: TRAINING ({EPOCHS} epochs x {len(corpus)} examples)")
    print("  " + "-" * 55)

    for epoch in range(EPOCHS + 1):
        total_loss = 0.0
        for word, target in corpus:
            grads = model.zero_grads()
            loss, _ = model.backward(word, target, grads)
            total_loss += loss
            model.step(grads, 0.3 * (1 - epoch/EPOCHS) + 0.02 * (epoch/EPOCHS))

        if epoch % 500 == 0 or epoch == EPOCHS:
            correct = sum(1 for w,t in corpus
                         if all((1 if sigmoid(z)>=0.5 else 0)==int(t[k])
                                for k,z in enumerate(
                                    [model.by[k]+sum(model.Why[k][j]*model.forward(w)[0][j]
                                     for j in range(model.H)) for k in range(K)])))
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:>5} | Loss: {total_loss/len(corpus):.6f} | {elapsed:.2f}s")

    train_time = time.time() - t0
    total_ex = (EPOCHS+1)*len(corpus)
    print(f"\n  Training: {train_time:.2f}s, {total_ex} examples, {total_ex/train_time:.0f} ex/sec")

    # BENCHMARK 2: Inference
    print(f"\n  BENCHMARK 2: INFERENCE LATENCY")
    N_INF = 500
    inf_t0 = time.time()
    for _ in range(N_INF):
        for word, _ in corpus:
            model.forward(word)
    inf_time = time.time() - inf_t0
    total_inf = N_INF * len(corpus)
    print(f"  {total_inf} inferences in {inf_time:.2f}s")
    print(f"  Per inference: {inf_time*1e6/total_inf:.1f} µs")
    print(f"  Throughput: {total_inf/inf_time:.0f} inferences/sec")

    # BENCHMARK 3: Forward+Backward
    print(f"\n  BENCHMARK 3: FORWARD+BACKWARD LATENCY")
    N_FB = 500
    fb_t0 = time.time()
    for _ in range(N_FB):
        for word, target in corpus:
            grads = model.zero_grads()
            model.backward(word, target, grads)
    fb_time = time.time() - fb_t0
    total_fb = N_FB * len(corpus)
    print(f"  {total_fb} passes in {fb_time:.2f}s")
    print(f"  Per pass: {fb_time*1e6/total_fb:.1f} µs")
    print(f"  Throughput: {total_fb/fb_time:.0f} passes/sec")

    # BENCHMARK 4: Quality
    print(f"\n  BENCHMARK 4: GENERALIZATION QUALITY")
    train_ok = 0
    for word, target in corpus:
        _,logits,_ = model.forward(word)
        p = [sigmoid(z) for z in logits]
        if all((1 if p[k]>=0.5 else 0)==int(target[k]) for k in range(K)):
            train_ok += 1
    print(f"  Training: {train_ok}/{len(corpus)} perfect words")

    print(f"\n  TOTAL TIME: {time.time()-t0:.1f}s")
    print("=" * 65)

if __name__ == "__main__":
    main()
