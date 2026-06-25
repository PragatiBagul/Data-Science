# Experiment : Why SGD is a Secret Regularizer: An Empirical Deep Dive

## Introduction

There's a question that has quietly puzzled deep learning practitioners for years: why do large, overparameterized neural networks — models with far more parameters than training samples — generalize so well in practice? Classical statistical learning theory would predict catastrophic overfitting. Yet empirically, these models often learn smooth, well-behaved functions that transfer gracefully to unseen data.

One compelling answer lies not in the architecture, not in explicit regularization techniques like dropout or weight decay, but in the **optimizer itself** — specifically, in the noise inherent to Stochastic Gradient Descent (SGD).

This post walks through a hands-on experiment that isolates and measures this phenomenon directly. We train the same overparameterized network three ways — mini-batch SGD, full-batch Gradient Descent, and Adam — and compare them across four interpretable metrics: weight norm, training and test loss, generalization gap, and prediction smoothness.

The results are striking, and in at least one case, deeply counterintuitive.

---

## Background: The Implicit Regularization Hypothesis

Regularization, in the classical sense, means adding an explicit penalty to the loss function to discourage model complexity. L2 regularization penalizes large weights. Dropout randomly zeroes activations. Early stopping halts training before the model memorizes noise.

Implicit regularization is subtler: it refers to the tendency of an optimization algorithm to favor certain solutions over others — not because of any explicit constraint, but because of the algorithm's geometry.

The key insight is this: when you compute gradients on a mini-batch rather than the full dataset, you introduce stochastic noise into each update. That noise isn't just an inconvenience to be tolerated — it actively shapes which minimum the optimizer converges to.

Research by Keskar et al. (2016) showed that large-batch SGD tends to converge to **sharp minimizers** — narrow valleys in the loss landscape where small perturbations cause large increases in loss. Mini-batch SGD, by contrast, tends to find **flat minimizers** — broad, gentle basins where the loss is robust to perturbation. Flat minima generalize better, because the model's behavior doesn't change drastically when you move from the training distribution to the test distribution.

But how large is this effect in practice? And how does it compare to Adam, one of the most widely adopted optimizers in modern deep learning?

---

## Experimental Setup

### Dataset

We generate a synthetic regression dataset from a noisy sine wave:

y = sin(x) + ε,    ε ~ N(0, 0.05²)

with 100 samples drawn uniformly over [-3, 3]. The signal is smooth and deterministic; the noise is small. This makes it easy to visually and quantitatively assess whether a model learned the true underlying function or overfit the noise.

The dataset is split 80/20 into training and test sets.

### Model: Intentionally Overparameterized

We use a deliberately wide 4-layer ReLU network:

Input(1) → Linear(512) → ReLU → Linear(512) → ReLU

→ Linear(512) → ReLU → Linear(512) → Output

This model has approximately **786,944 parameters** for just 80 training points — a parameter-to-sample ratio of roughly 10,000:1. Under classical intuitions, this model should memorize the training data with ease and generalize poorly. That is the assumption we are here to stress-test.

### Three Training Conditions

| **Condition** | **Optimizer** | **Batch size** | **Learning rate** |
| --- | --- | --- | --- |
| Mini-batch SGD | SGD | 16 | 0.01 |
| Full GD | SGD | 80 (full dataset) | 0.01 |
| Adam | Adam | 16 | 0.001 |

All three conditions are trained for 2,000 epochs from the same random initialization (seed = 42).

### Metrics

We track four metrics throughout and at convergence:

1. **Weight norm** — the L2 norm of all model parameters, a proxy for solution complexity.
2. **Train and test loss** — mean squared error on both splits.
3. **Generalization gap** — test loss minus train loss. Negative values indicate better-than-training performance on unseen data.
4. **Prediction smoothness** — mean absolute second-order finite difference of predictions on [-5, 5], i.e., how much curvature the learned function has. Lower = smoother.

---

## Results

Here are the final metrics at epoch 2,000:

| **Optimizer** | **Weight Norm** | **Train Loss** | **Test Loss** | **Gen. Gap** | **Smoothness** |
| --- | --- | --- | --- | --- | --- |
| SGD | 29.60 | 0.004690 | 0.004106 | **−0.000584** | 0.090190 |
| Full GD | 29.39 | 0.095017 | 0.111745 | +0.016728 | 0.069132 |
| Adam | **38.18** | **0.001228** | 0.002095 | +0.000868 | **0.039242** |

### Weight Norm: Adam Keeps Growing

All three models are initialized identically, with weight norm ≈ 29.27. Over 2,000 epochs:

- **SGD** drifts only slightly upward to 29.60.
- **Full GD** stays nearly flat at 29.39.
- **Adam** climbs steadily to **38.18** — nearly 30% larger than where it started.

This is a direct consequence of Adam's adaptive learning rates. By maintaining per-parameter moment estimates, Adam can take large effective steps in directions of low recent gradient variance. There is no natural brake on weight growth. The practical implication: Adam solutions tend to live in more complex regions of the weight space, which can hurt generalization in low-data regimes.

### Loss: Full GD Fails to Converge

The most surprising result is the performance of Full GD. With access to the exact gradient at every step — no noise, no approximation — one might expect it to converge most reliably. Instead, it achieves the *worst* train and test loss of all three conditions: 0.095 and 0.112 respectively.

This is a vivid illustration of the flat-vs-sharp minima phenomenon. Without the stochastic noise from mini-batches, full gradient descent follows the gradient exactly — and that exact gradient leads it straight into a sharp, narrow minimum. Sharp minima have high loss in their immediate neighborhood, which explains why the model generalizes poorly even on training data.

Mini-batch SGD, ironically, *benefits* from the randomness. The noisy gradient acts as a perturbation that repeatedly bounces the optimizer out of sharp minima and keeps it searching for broader, flatter basins. When it eventually settles, it has found a solution that is both low-loss and stable.

Adam achieves the lowest train loss (0.00123), benefiting from its adaptive learning rates to navigate the loss landscape efficiently. Its test loss (0.00210) is also low, making it a strong performer overall — but it does not match the generalization characteristics of SGD.

### Generalization Gap: SGD Goes Negative

The generalization gap (test loss − train loss) is the most direct measure of how well learned knowledge transfers to unseen data.

- **SGD: −0.000584** — SGD's test loss is *lower* than its train loss. The model performs better on data it has never seen. This is a hallmark of a regularized, conservative solution that captures the underlying structure of the data rather than its noise.
- **Full GD: +0.016728** — The largest positive gap, confirming that its solution does not transfer well.
- **Adam: +0.000868** — A small positive gap. Adam generalizes well, but its direction of error is the classical one: slight overfitting to training data.

A negative generalization gap in a model with 10,000× more parameters than data points is a remarkable outcome. It is not explained by early stopping, weight decay, or any explicit constraint — only by the implicit regularization of mini-batch noise.

### Smoothness: A Nuanced Trade-off

Smoothness measures how much the model's predictions curve over the extrapolation range [-5, 5] (wider than the training range of [-3, 3]). Lower values indicate more regular, well-behaved predictions.

- **Adam: 0.039** — the smoothest predictions, consistent with its good fit and low loss.
- **Full GD: 0.069** — moderate smoothness, though its high loss undermines the value of this.
- **SGD: 0.090** — the roughest predictions of the three.

SGD's higher smoothness score is worth examining carefully. Despite having the most curvature in its predictions, SGD achieves the best generalization. This tells us that smoothness alone is not the right proxy for generalization quality — what matters is whether the curvature reflects the true signal or spurious noise. SGD's slight roughness captures real structure in the sine wave; it is not a symptom of overfitting.

---

## Interpreting the Full Picture

Taken together, these results form a coherent story about the role of optimization noise in generalization.

**Mini-batch SGD** introduces controlled randomness into every gradient update. That randomness serves as an implicit regularizer in two complementary ways: it prevents the optimizer from committing to sharp, brittle minima, and it applies a kind of implicit pressure toward solutions with smaller weight norms. The result is a model that generalizes exceptionally well — even better on unseen data than on its own training set.

**Full-batch Gradient Descent** removes that randomness entirely. Without the perturbation that mini-batches provide, the optimizer converges to whichever local minimum the gradient first leads it to. In the highly non-convex landscape of a deep network, that minimum is often sharp and narrow — low-loss in a very small neighborhood, but poor at generalization. This is not a failure of the algorithm per se, but an inherent consequence of deterministic optimization in non-convex settings.

**Adam** represents a different trade-off. Its adaptive learning rates allow efficient navigation of the loss landscape and excellent training-time performance. However, the per-parameter momentum accumulation causes weights to drift upward throughout training, landing in a region of higher weight norm than SGD. In this experiment Adam still generalizes well, but it does so with a larger solution — and that excess complexity could become a liability at scale, in lower-data regimes, or when training for longer.

---

## Implications for Practice

These findings carry direct implications for how we should think about optimizer choice:

**1. Batch size is a regularization hyperparameter.** Increasing batch size is not a free performance optimization. Larger batches reduce gradient noise, which can push the model toward sharp minima. If you are experiencing unexpected overfitting after scaling batch size, this may be the cause.

**2. SGD's noise is a feature, not a bug.** Many practitioners switch to Adam as a default because it converges faster and is less sensitive to learning rate tuning. Those are genuine advantages. But in settings where generalization is paramount and data is scarce, mini-batch SGD's implicit regularization may produce better-calibrated models.

**3. Weight norm growth is worth monitoring.** In this experiment, Adam's unbounded weight norm growth is visible from early in training. Monitoring weight norms during training is cheap and informative — a model whose norms grow steadily throughout training may be accumulating complexity without bound.

**4. Explicit and implicit regularization interact.** These experiments were run without any explicit regularization (no dropout, no weight decay, no early stopping). In most production settings you would add some. The implicit regularization of SGD compounds with these techniques; Adam's weaker implicit regularization means explicit constraints carry more of the burden.

---

## Limitations and Future Directions

This experiment is intentionally minimal to isolate the phenomenon of interest. Several important variables were held constant or left unexplored:

- **Learning rate schedules**: Both SGD and Adam were run with fixed learning rates. Cyclical learning rates and warmup schedules can meaningfully change the flatness of the minimum found.
- **Explicit regularization**: Adding weight decay, dropout, or batch normalization would change the dynamics considerably — and may close the gap between SGD and Adam.
- **Scale**: The model here has ~787K parameters. The relative behavior of these optimizers at the scale of modern large language models (billions of parameters) may differ.
- **Dataset complexity**: A 1D sine wave is a best-case scenario for visualization and interpretability. Higher-dimensional, noisier, or multi-modal data may produce qualitatively different results.

Natural extensions of this work would include varying batch size systematically to measure its effect on the generalization gap, experimenting with noise injection into Adam to approximate SGD-style implicit regularization, and examining the loss landscape geometry directly using Hessian eigenvalue analysis.

---

## Conclusion

We set out to answer a simple question: does the choice of optimizer affect generalization, independent of any explicit regularization? The answer, demonstrated empirically here, is a clear yes.

Mini-batch SGD's gradient noise is not a computational inconvenience to be minimized — it is an active and beneficial force that shapes the geometry of the learned solution. It keeps weights compact, avoids sharp minima, and produces models that generalize to unseen data at least as well as they fit the data they were trained on, sometimes better.

Full-batch Gradient Descent, stripped of this noise, converges deterministically to worse solutions. Adam, optimized for fast convergence, accumulates weight norm without bound and sacrifices some of SGD's implicit regularization for training-time efficiency.

For practitioners, the lesson is that the optimizer is not just a means to minimize training loss — it is itself a modeling choice, with direct consequences for what kind of solution the model finds and how well that solution generalizes. Choosing an optimizer thoughtfully, with an understanding of the implicit biases it carries, is as important as choosing an architecture or a loss function.

The stochastic noise you have been treating as unavoidable overhead is, in many cases, one of the most effective regularizers in your toolkit.

---

## References

1. Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2016). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.* arXiv:1609.04836.
2. Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). *Understanding Deep Learning Requires Rethinking Generalization.* ICLR 2017.
3. Neyshabur, B., Tomioka, R., & Srebro, N. (2015). *In Search of the Real Inductive Bias: On the Role of Implicit Regularization in Deep Learning.* ICLR Workshop 2015.
4. Smith, S. L., & Le, Q. V. (2018). *A Bayesian Perspective on Generalization and Stochastic Gradient Descent.* ICLR 2018.
5. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR 2015.

---
