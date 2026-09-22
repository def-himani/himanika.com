# Attention Mechanisms and Transformer Architectures

## 1. Introduction and Architectural Motivation

Traditional sequence-to-sequence architectures relied primarily on Recurrent Neural Networks (RNNs), such as Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs). These models process tokens sequentially along the time dimension:

$$h_t = f(h_{t-1}, x_t)$$

This sequential dependency introduces two primary structural bottlenecks:

1. **Inability to Parallelize:** Computation at step $t$ strictly requires the hidden state $h_{t-1}$ from the prior step, preventing full hardware utilization during training across long contexts.
2. **Vanishing and Exploding Gradients:** Information traversing long temporal paths suffers from signal degradation, limiting the effective context window despite gating mechanisms.

The Transformer architecture, introduced by Vaswani et al. (2017), bypasses recurrence entirely by utilizing multi-head self-attention, establishing constant $O(1)$ path lengths between any pair of input tokens regardless of their sequence distance.

---

## 2. Mathematical Formulation of Scaled Dot-Product Attention

The core building block of the model is Scaled Dot-Product Attention. Given an input sequence representation matrix $X \in \mathbb{R}^{n \times d_{\text{model}}}$, where $n$ is the sequence length and $d_{\text{model}}$ is the hidden dimension, the representations are linearly projected into three distinct spaces:

$$\mathbf{Q} = XW^Q, \quad \mathbf{K} = XW^K, \quad \mathbf{V} = XW^V$$

Where the projection parameter matrices satisfy:

$$W^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$$

### 2.1 Attention Map Computation

The similarity score between queries and keys is computed via matrix inner products. To prevent gradient saturation in the softmax function for large inner-product dimensions $d_k$, the dot products are scaled by $\frac{1}{\sqrt{d_k}}$:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

For a single query vector $q_i \in \mathbb{R}^{d_k}$ interacting with a sequence of key vectors $\{k_1, k_2, \dots, k_n\}$ and values $\{v_1, v_2, \dots, v_n\}$, the scalar attention weight assigned to position $j$ is:

$$\alpha_{ij} = \frac{\exp\left(\frac{q_i^T k_j}{\sqrt{d_k}}\right)}{\sum_{l=1}^{n} \exp\left(\frac{q_i^T k_l}{\sqrt{d_k}}\right)}$$

The resulting contextual representation vector $z_i \in \mathbb{R}^{d_v}$ is expressed as a convex combination of the value vectors:

$$z_i = \sum_{j=1}^{n} \alpha_{ij} v_j$$

### 2.2 Necessity of the Variance Scaling Factor

Assume the components of $q \in \mathbb{R}^{d_k}$ and $k \in \mathbb{R}^{d_k}$ are independent and identically distributed random variables with zero mean and unit variance:

$$\mathbb{E}[q^{(r)}] = 0, \quad \text{Var}(q^{(r)}) = 1, \quad \mathbb{E}[k^{(r)}] = 0, \quad \text{Var}(k^{(r)}) = 1$$

The inner product scalar is:

$$S = q^T k = \sum_{r=1}^{d_k} q^{(r)} k^{(r)}$$

Applying expectations and variance arithmetic:

$$\mathbb{E}[S] = \sum_{r=1}^{d_k} \mathbb{E}[q^{(r)}] \mathbb{E}[k^{(r)}] = 0$$

$$\text{Var}(S) = \sum_{r=1}^{d_k} \text{Var}(q^{(r)} k^{(r)}) = \sum_{r=1}^{d_k} \left(\mathbb{E}[(q^{(r)})^2]\mathbb{E}[(k^{(r)})^2] - (\mathbb{E}[q^{(r)}]\mathbb{E}[k^{(r)}])^2\right) = d_k$$

As $d_k$ grows large, the variance of the dot products scales proportionally with $d_k$. Large-magnitude inputs push the softmax function into regions with near-zero gradients:

$$\frac{\partial \, \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i (\delta_{ij} - \text{softmax}(z)_j) \approx 0 \quad \text{for } |z_i| \gg 0$$

Dividing by $\sqrt{d_k}$ normalizes the variance back to $\text{Var}\left(\frac{S}{\sqrt{d_k}}\right) = 1$, preserving stable gradient backpropagation during early training phases.

---

## 3. Multi-Head Attention Mechanisms

Instead of performing a single attention function with dimension $d_{\text{model}}$, Multi-Head Attention projects queries, keys, and values $h$ times with independently learned linear transformations:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

Where each individual head computation is defined as:

$$\text{head}_i = \text{Attention}(\mathbf{Q}W_i^Q, \, \mathbf{K}W_i^K, \, \mathbf{V}W_i^V)$$

With parameter projections:

$$W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}, \quad W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$$

Typically, dimensions are chosen such that:

$$d_k = d_v = \frac{d_{\text{model}}}{h}$$

This guarantees that the total computational cost remains equivalent to standard single-head attention with full dimensionality.

---

## 4. Position-Wise Feed-Forward Networks

Following the attention layer in each Transformer sub-layer, an identical feed-forward network is applied independently to each position:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Where the parameter spaces satisfy:

$$W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}, \quad W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}, \quad b_1 \in \mathbb{R}^{d_{ff}}, \quad b_2 \in \mathbb{R}^{d_{\text{model}}}$$

In standard configurations, the inner dimension is four times the model dimension:

$$d_{ff} = 4 \times d_{\text{model}}$$

More recent implementations replace the ReLU activation with parameterized variants such as Gaussian Error Linear Units (GELU) or SwiGLU:

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0, 1)$$

$$\text{SwiGLU}(x, W, V, b, c) = \text{Swish}(xW + b) \otimes (xV + c)$$

---

## 5. Positional Encoding Methods

Because the attention operator is permutation-equivariant:

$$\text{Attention}(P\mathbf{Q}, P\mathbf{K}, P\mathbf{V}) = P \cdot \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$$

The network does not have an intrinsic representation of token order. Positional information must therefore be explicitly injected into the input embeddings.

### 5.1 Sinusoidal Positional Encoding

Vaswani et al. proposed deterministic fixed sinusoidal encodings across dimensions $i \in \{0, \dots, \frac{d_{\text{model}}}{2}-1\}$:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

This allows the model to extrapolate to sequence lengths not observed during training, as for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear transformation of $PE_{pos}$:

$$\begin{bmatrix} \sin(\omega (pos + k)) \\ \cos(\omega (pos + k)) \end{bmatrix} = \begin{bmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{bmatrix} \begin{bmatrix} \sin(\omega pos) \\ \cos(\omega pos) \end{bmatrix}$$

### 5.2 Rotary Position Embedding (RoPE)

Modern decoder-only architectures (e.g., LLaMA, Mistral) frequently deploy Rotary Position Embeddings. RoPE encodes relative position directly into the inner product of queries and keys by rotating the vectors in the 2D plane:

$$R_{\Theta, m}^{2d} = \text{diag}\left(R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m}\right)$$

Where each $2 \times 2$ block rotation matrix is defined as:

$$R_{\theta_i, m} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix}$$

This structure guarantees that the attention dot product depends strictly on relative offset $(m - n)$:

$$\langle R_{\Theta, m} q, \, R_{\Theta, n} k \rangle = q^T R_{\Theta, n - m} k$$

---

## 6. Layer Normalization Formulations

Residual connections surround each sub-layer to preserve forward signal and backward gradients:

$$y = \text{LayerNorm}(x + \text{SubLayer}(x))$$

### 6.1 Post-LN vs. Pre-LN Architectures

* **Post-LN (Original Transformer):** Normalization is computed after the residual addition:
  $$x_{l+1} = \text{LN}(x_l + \text{SubLayer}(x_l))$$
  Post-LN requires warm-up scheduling during optimization because gradient scales at initialization are sensitive to network depth.
* **Pre-LN (Modern Standard):** Normalization is placed on the input branch:
  $$x_{l+1} = x_l + \text{SubLayer}(\text{LN}(x_l))$$
  Pre-LN enables gradient flow directly through the identity branch, supporting stable training without extensive learning-rate warm-up.

### 6.2 Root Mean Square Normalization (RMSNorm)

Standard Layer Normalization calculates both sample mean and variance:

$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta, \quad \mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$$

RMSNorm simplifies this computation by omitting the mean centering step, reducing computational overhead without loss of model quality:

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

---

## 7. Complexity Analysis: Recurrence vs. Self-Attention

| Layer Type | Computational Complexity per Layer | Sequential Operations | Maximum Path Length |
| :--- | :--- | :--- | :--- |
| **Self-Attention** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| **Recurrent (RNN)** | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| **Convolutional** | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |

When sequence length $n$ is smaller than representation dimensionality $d$, self-attention layers exhibit lower computational cost than recurrent layers. For long-context regimes where $n > d$, quadratic complexity $O(n^2)$ becomes the primary memory and compute constraint, motivating FlashAttention and linear approximation variants.
