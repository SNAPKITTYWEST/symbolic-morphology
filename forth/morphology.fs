\ Symbolic morphology learner — Forth (gforth) port. Same architecture, init and online SGD as src/rust.
\ embed(16) -> mean-pool -> tanh(32) -> sigmoid(23), BCE, manual backprop, lr 0.5, 3001 epochs.
\ Run from this directory:  gforth morphology.fs -e bye

128 constant NCH  16 constant E  32 constant H  23 constant F  96 constant NS  16 constant MAXL
0.5e fconstant LR
17 set-precision

: mkarr ( n "name" -- ) create floats allot ;

NCH E * mkarr emb    H E * mkarr w1     H mkarr b1
F H * mkarr w2       F mkarr b2
E mkarr pooled       H mkarr a1         F mkarr y
F mkarr dz2          H mkarr dz1        E mkarr dh
NS F * mkarr tgt     3351 mkarr tmp
create wch NS MAXL * cells allot
create wln NS cells allot

variable cs   variable ln   variable nsm
fvariable p   fvariable tg  fvariable invn  fvariable total  fvariable lastl

\ ---------------------------------------------------------------- data loading
create lbuf 256 allot
variable fid
: open-in ( a u -- fid ) r/o open-file throw ;

: load-init
  s" ../bench/shared/init.txt" open-in fid !
  3351 0 do
    lbuf 256 fid @ read-line throw drop   ( u )
    lbuf swap >float 0= abort" bad float"
    i floats tmp + f!
  loop
  fid @ close-file throw
  tmp emb  NCH E * floats move
  tmp NCH E * floats +  w1  H E * floats move
  tmp NCH E * H E * + floats +  b1  H floats move
  tmp NCH E * H E * + H + floats +  w2  F H * floats move
  tmp NCH E * H E * + H + F H * + floats +  b2  F floats move ;

: (ld-line)   \ parses the current input line: WORD then 23 single-digit 0/1 flags
  parse-name ( a u )  dup wln nsm @ cells + !
  0 do  dup i + c@ wch nsm @ MAXL * i + cells + !  loop  drop
  F 0 do  parse-name drop c@ [char] 0 - s>f  nsm @ F * i + floats tgt + f!  loop ;

: load-corpus
  0 nsm !
  s" ../bench/shared/corpus.txt" open-in fid !
  begin  lbuf 256 fid @ read-line throw  while
    lbuf swap ['] (ld-line) execute-parsing  1 nsm +!
  repeat drop  fid @ close-file throw ;

\ ---------------------------------------------------------------- model
: sample-c ( t -- c ) cs @ MAXL * + cells wch + @ ;
: sigm ( f: x -- y )
  fdup 0e f< if fexp fdup 1e f+ f/
  else fnegate fexp 1e f+ 1e fswap f/ then ;

: forward
  cs @ cells wln + @ ln !
  E 0 do
    0e  ln @ 0 do  i sample-c E * j + floats emb + f@ f+  loop
    ln @ s>f f/  i floats pooled + f!
  loop
  H 0 do
    i floats b1 + f@
    E 0 do  j E * i + floats w1 + f@  i floats pooled + f@ f*  f+  loop
    ftanh  i floats a1 + f!
  loop
  F 0 do
    i floats b2 + f@
    H 0 do  j H * i + floats w2 + f@  i floats a1 + f@ f*  f+  loop
    sigm  i floats y + f!
  loop ;

: loss ( f: -- l )
  0e
  F 0 do
    i floats y + f@  1e-12 fmax  1e 1e-12 f- fmin  p f!
    cs @ F * i + floats tgt + f@ tg f!
    tg f@ p f@ fln f*   1e tg f@ f- 1e p f@ f- fln f*  f+  f-
  loop
  F s>f f/ ;

: backward
  F 0 do  i floats y + f@  cs @ F * i + floats tgt + f@ f-  F s>f f/  i floats dz2 + f!  loop
  H 0 do
    0e  F 0 do  i H * j + floats w2 + f@  i floats dz2 + f@ f*  f+  loop
    i floats a1 + f@ fdup f* 1e fswap f-  f*  i floats dz1 + f!
  loop
  E 0 do
    0e  H 0 do  i E * j + floats w1 + f@  i floats dz1 + f@ f*  f+  loop
    i floats dh + f!
  loop
  1e ln @ s>f f/ invn f!
  ln @ 0 do
    i sample-c
    E 0 do  dup E * i + floats emb + dup f@  invn f@ i floats dh + f@ f* LR f* f-  f!  loop
    drop
  loop
  H 0 do
    i floats b1 + dup f@  LR i floats dz1 + f@ f* f-  f!
    E 0 do  j E * i + floats w1 + dup f@  j floats dz1 + f@ i floats pooled + f@ f*  LR f*  f-  f!  loop
  loop
  F 0 do
    i floats b2 + dup f@  LR i floats dz2 + f@ f* f-  f!
    H 0 do  j H * i + floats w2 + dup f@  j floats dz2 + f@ i floats a1 + f@ f*  LR f*  f-  f!  loop
  loop ;

: train ( epochs -- )
  1+ 0 do
    0e total f!
    nsm @ 0 do  i cs !  forward  loss total f@ f+ total f!  backward  loop
    total f@ nsm @ s>f f/ lastl f!
  loop ;

: bench
  load-init load-corpus
  utime
  3000 train
  utime 2swap d- d>f 1e6 f/   ( f: seconds )
  ." forth  examples=" nsm @ . ." epochs=3001 final_loss=" lastl f@ f.
  ." time=" fdup f. ." s ex/s=" nsm @ 3001 * s>f fswap f/ f. cr ;

bench
