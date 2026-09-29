# Symbolic morphology learner — BQN port. Same architecture, init and online SGD as src/rust.
# embed(16) → mean-pool → tanh(32) → sigmoid(23), BCE, manual backprop, lr 0.5, 3001 epochs.
# Run from this directory (CBQN):  BQN morphology.bqn

ed‿hd‿fd ← 16‿32‿23
lr ← 0.5

# ── data ─────────────────────────────────────────────────────────────────────
Fix ← {𝕩 + (𝕩='-') × '¯'-'-'}
Num ← {•BQN Fix 𝕩}
Split ← (⊢-˜+`×¬)∘=⟜⊢⊔⊢

v ← Num¨ •file.Lines "../bench/shared/init.txt"
Take ← {r ← 𝕨↑v ⋄ v ↩ 𝕨↓v ⋄ r}
emb ← (128‿ed) ⥊ (128×ed) Take @
w1  ← (hd‿ed)   ⥊ (hd×ed) Take @
b1  ← hd Take @
w2  ← (fd‿hd)   ⥊ (fd×hd) Take @
b2  ← fd Take @

rows ← ' ' Split¨ •file.Lines "../bench/shared/corpus.txt"
words ← {-⟜@ ⊑𝕩}¨ rows                 # character codes of each word
tgt   ← {Num¨ 1↓𝕩}¨ rows               # 23 flags per word
n ← ≠words

# ── model ────────────────────────────────────────────────────────────────────
MV ← {+´˘ 𝕨 ×⎉1 𝕩}                      # matrix · vector
Sigmoid ← {÷1+⋆-𝕩}
Tanh ← {e ← ⋆2×𝕩 ⋄ (e-1)÷e+1}

# One online-SGD step on a sample; returns its BCE loss (computed before the update).
Step ← {w‿t:
  len ← ≠w
  h ← (+˝ w⊏emb) ÷ len                    # mean-pool: sum the rows
  a1 ← Tanh b1 + w1 MV h
  y ← Sigmoid b2 + w2 MV a1
  p ← 1e¯12 ⌈ y ⌊ 1-1e¯12
  l ← (+´ -(t×⋆⁼p) + (1-t)×⋆⁼1-p) ÷ fd
  dz2 ← (y-t) ÷ fd
  dz1 ← (+˝ dz2 × w2) × 1-a1×a1
  dh ← +˝ dz1 × w1
  { emb ↩ (-⟜(lr×dh÷len))⌾(𝕩⊸⊏) emb }¨ w
  b1 -↩ lr×dz1
  w1 -↩ lr×dz1 ×⌜ h
  b2 -↩ lr×dz2
  w2 -↩ lr×dz2 ×⌜ a1
  l
}

Epoch ← {𝕩 ⋄ (+´ Step¨ words ⋈¨ tgt) ÷ n}

t0 ← •MonoTime @
last ← 0
{𝕩 ⋄ last ↩ Epoch @}¨ ↕3001
dt ← (•MonoTime @) - t0
•Out "bqn   examples=" ∾ (•Fmt n) ∾ " epochs=3001 final_loss=" ∾ (•Repr last) ∾ " time=" ∾ (•Repr dt) ∾ "s ex/s=" ∾ •Repr (n×3001)÷dt
