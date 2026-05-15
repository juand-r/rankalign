I was being a bit terse — let me unpack it.

"Eval direction" here = which typicality reference is subtracted from the generator score at evaluation time. We have several:

self eval-ref: subtract $\log P_\theta(y)$ — the trained model's own marginal probability of the answer.
neg eval-ref: subtract $\log P_\theta(y \mid \mathrm{neg}(x))$ — what the trained model thinks of $y$ given the negated prompt.
basetyp / basetypneg: same but under the frozen base model.
Each one defines a different scoring rule on the same items, and so a different gen-ROC.

The asymmetry I was pointing at:

sig wins for TC vs RankAlign	sig losses
self side (everything under self/basetyp)
6/10 (offline) and 3/10 (online)
0 / 0
neg side (everything under neg/basetypneg)
2/10 (offline) and 3/10 (online)
6/10 (offline) and 2/10 (online)
If TC were a "fix" that improves the generator regardless of how you measure it, you'd expect roughly the same picture on both sides. Instead:

Self-direction: TC reliably outperforms the no-TC baseline.
Neg-direction: TC mostly loses to the no-TC baseline (especially offline neg-TC). RankAlign actually does better on neg eval than on self eval for many tasks.
That's the asymmetry. The benefit of TC is direction-specific, not generic.

Why might this happen? I'm speculating — you'd need a separate diagnostic to confirm any of these — but candidates:

The neg prompt is noisier. "Do you think a robin is NOT a bird?" is a less natural format and the model's $P_\theta(y \mid \mathrm{neg}(x))$ may be more chaotic than $P_\theta(y)$. If the neg-direction signal is itself unreliable, training to correct against it can amplify rather than cancel noise.

Training objective and eval direction interact. When we train neg-TC, we subtract $\log P_\theta(y \mid \mathrm{neg}(x))$ during training; at eval we subtract that same quantity from the post-training model. If the training pushes the model in a way that makes the neg-prompt typicality less informative (e.g. flatter), the neg-eval metric gets worse even if the underlying generator improved.

RankAlign has a hidden advantage on neg eval. RankAlign trains without TC. At eval time under the neg eval-ref, the no-TC-trained model's neg-prompt distribution might happen to be a more useful normalizer than for the TC-trained models. In that case the asymmetry tells us about the evaluation metric, not about which method generates better.

The cleanest way to disentangle these would be: at training time, vary which TC reference is used; at eval time, hold the eval-ref fixed at one canonical thing (e.g. always basetyp) and see whether the asymmetry persists. If it does, it's a generation-quality story. If it disappears, it's a metric story.

Worth noting on the experimental side: I'm pairing each TC variant against RankAlign using the variant's canonical eval ref (offline TC → basetyp[neg], online TC and RankAlign → self/neg). So part of what's being attributed to "neg side" might actually be "neg + basetypneg vs self + self," which is a more compound comparison. If you want, the same paired-bootstrap script can re-render the deltas under matched eval refs (e.g. all under self vs all under basetyp) to peel that apart — it's a one-line change.
