# SVG Scaling Laws — Complete Educational Guide

**Course:** CS-GY 6923, NYU Tandon, Spring 2026  
**Author:** Shubham Tanwar  
**Purpose:** A complete, from-scratch explanation of every step in this project — the code, the math, the intuition, and every bug we encountered.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Part 1 — Data Pipeline](#2-part-1--data-pipeline)
3. [Part 2 — Model Architecture](#3-part-2--model-architecture)
4. [Part 2 — Training Loop](#4-part-2--training-loop)
5. [Part 2 — Learning Rate Sweep](#5-part-2--learning-rate-sweep)
6. [Part 2 — Scaling Law Fit](#6-part-2--scaling-law-fit)
7. [Part 3 — Maximal Update Parameterization (μP)](#7-part-3--maximal-update-parameterization-p)
8. [Part 4 — Extended Training and SVG Generation](#8-part-4--extended-training-and-svg-generation)
9. [All Bugs and How We Fixed Them](#9-all-bugs-and-how-we-fixed-them)
10. [Complete Results Reference](#10-complete-results-reference)
11. [File Map — Every Script and Notebook](#11-file-map--every-script-and-notebook)

---

## 1. What Is This Project?

### The Big Picture

This project is a scientific study: *how does a language model's quality improve as you make it bigger?*

We do this on **SVG code** — the XML-based format used to describe vector graphics like icons and logos. An SVG file looks like this:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2z"/>
</svg>
```

The model never "sees" the image. It only reads the text of the SVG code, one character/token at a time, and learns to predict what comes next — the same way GPT-4 predicts the next word in a sentence.

### Why SVG?

- It is **structured text** — it follows grammar rules (XML tags, attribute=value pairs).
- It is **compact** — a small file describes a complex shape.
- It is **objectively evaluatable** — we can check if the output is valid by trying to parse and render it.
- It is different from natural language, so we can compare whether scaling laws from text carry over.

### What is a "Scaling Law"?

A scaling law is an observation that model quality (measured by loss) follows a **power law** as model size increases:

```
L(N) = a × N^(-α) + c
```

Where:
- `L` = validation loss (lower = better model)
- `N` = number of model parameters
- `α` = scaling exponent (how fast quality improves per doubling of size)
- `c` = irreducible loss floor (minimum achievable loss, no matter how big the model)
- `a` = a constant

This means: double the model size, and loss drops by roughly `2^α`. For natural language, Kaplan et al. (2020) found `α ≈ 0.076`. We found `α ≈ 0.478` for SVG — meaning SVG quality improves *much faster* with scale.

### Project Structure

```
Part 1: Build data pipeline → 101.7M training tokens
Part 2: Train 5 model sizes, fit power law, find best learning rate
Part 3: Re-do Part 2 with μP (a smarter parameterization), compare
Part 4: Train best model longer, generate SVG samples, evaluate quality
```

---

## 2. Part 1 — Data Pipeline

### Overview

Before training any model, we need data. Our pipeline has 5 scripts:

```
01_download.py  →  02_clean.py  →  03_train_tokenizer.py
                                          ↓
                   05_statistics.py  ←  04_split_and_encode.py
```

**Final output:** Three binary files — `data/tokenized/train.npy`, `val.npy`, `test.npy` — containing 101.7M + 1.04M + 1.03M token IDs as flat uint16 arrays.

---

### Script 1: `scripts/01_download.py` — Downloading Data

**What it does:** Downloads SVG datasets from HuggingFace and saves them as JSONL (one JSON object per line, each containing one SVG string).

**Three datasets downloaded:**

| Dataset | Count | Purpose |
|---|---|---|
| `starvector/svg-icons-simple` | 89,370 | Semantic icons (search, home, etc.) |
| `starvector/svg-emoji-simple` | 5,106 | Color emoji SVGs |
| `starvector/svg-fonts-simple` | 320,000 (capped) | Font glyph SVGs |
| **Total** | **414,476** | |

**Why three datasets?** Icons alone give only ~15M tokens (89K SVGs × ~168 tokens/SVG). We need 100M+ tokens for meaningful scaling experiments. Font glyphs are short and regular — adding 320K of them fills the token budget.

**Key code concept:** `detect_svg_field()` inspects the first record of each dataset to find which JSON key holds the SVG string (different datasets use `"image"`, `"svg"`, `"code"`, etc.). This makes the script robust to different dataset formats.

**Output:** `data/raw/icons.jsonl`, `data/raw/emoji.jsonl`, `data/raw/fonts.jsonl`

---

### Script 2: `scripts/02_clean.py` — Cleaning SVGs

**What it does:** Takes raw SVGs and applies normalization to make them consistent. This matters because noisy, inconsistent data hurts model quality.

**Five cleaning steps:**

#### Step 1: Strip XML Comments
```python
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
svg = _COMMENT_RE.sub("", svg)
```
Comments like `<!-- Adobe Illustrator 2020 -->` are metadata noise. They waste tokens without teaching the model anything about SVG structure.

#### Step 2: Strip `<metadata>` Blocks
```python
_META_RE = re.compile(r"<metadata[^>]*>.*?</metadata>", re.DOTALL)
svg = _META_RE.sub("", svg)
```
Same idea — license blocks, tool information, author names. All noise.

#### Step 3: Collapse Whitespace
```python
_WS_RE = re.compile(r"\s+")
svg = _WS_RE.sub(" ", svg).strip()
```
SVG files often have newlines and extra spaces for human readability. We collapse everything to single spaces. A single canonical format means the tokenizer learns one pattern, not many variations of the same thing.

#### Step 4: Round Floating-Point Coordinates to 1 Decimal Place
```python
_FLOAT_RE = re.compile(r"-?\d+\.\d+")
def _round_float(match):
    return f"{float(match.group()):.1f}"
svg = _FLOAT_RE.sub(_round_float, svg)
```

**This is critical and non-obvious.** SVG coordinates come out of design tools with many decimal places: `12.34567890`. These are essentially random from a model's perspective. The token `12.3` repeats many times; `12.34567890` almost never repeats. Reducing precision:
1. Makes the token vocabulary more meaningful
2. Dramatically reduces the **entropy** of numbers (easier to predict)
3. Reduces file length by 30-50%

**ML intuition:** High entropy in coordinates means the model wastes capacity memorizing meaningless precision. Rounding removes this noise.

#### Step 5: Length Filter and XML Validation
```python
MIN_CHARS = 50
MAX_CHARS = 1024 * 4   # = 4096 chars ≈ 1024 tokens
if len(svg) < MIN_CHARS or len(svg) > MAX_CHARS:
    continue
if not is_valid_xml(svg):
    continue
```
- Too short: probably a malformed or empty SVG.
- Too long: exceeds our model's context window (1024 tokens). Filtering here (at char level) is fast; we do a more precise filter at token level in step 4.
- Invalid XML: broken SVGs that would produce garbage training signal.

**Final numbers:**
- Input: 414,476 SVGs
- Output: 402,617 SVGs (97.1% kept)
- Dropped (too long): 11,859
- Dropped (bad XML): 0 (the starvector datasets are pre-validated)

**Output:** `data/cleaned/cleaned.jsonl`

---

### Script 3: `scripts/03_train_tokenizer.py` — Training the Tokenizer

**The fundamental question: what is a token?**

A language model doesn't read characters one at a time — that would require very long sequences. Instead, it reads **tokens**: common substrings that appear repeatedly in the training data.

Examples of SVG tokens the model might learn:
- `<path` (a common opening tag)
- ` d="` (the path data attribute)
- `M` (SVG "move to" command)
- `12.3` (a coordinate)
- `viewBox` (a common attribute)

**What is BPE (Byte Pair Encoding)?**

BPE builds a vocabulary by:
1. Start with individual characters as tokens: `<`, `s`, `v`, `g`, ` `, `d`, `=`, `"`, ...
2. Count all adjacent pairs of tokens.
3. Merge the most frequent pair into a new token.
4. Repeat until you have `vocab_size` tokens.

Example merge sequence:
```
v + i  → vi
vi + e → vie
vie + w → view
view + B → viewB
viewB + o → viewBo
viewBo + x → viewBox   ← now a single token!
```

**Why not use HuggingFace ByteLevel BPE?**

We tried it first. It produced only **164 tokens** instead of the requested 1,024. Here is why:

HuggingFace's ByteLevel BPE splits text on whitespace **before** running BPE. SVG path data looks like:
```
d="M 10.3 20.7 L 30.5 40.1 C 15.2 8.3 20.1 5.5 25.0 8.0"
```

After splitting on spaces, we get chunks like `10.3`, `20.7`, `L`, `30.5`, etc. BPE can only merge characters **within** each chunk, not across spaces. Since coordinate strings like `10.3` rarely repeat verbatim, there are almost no high-frequency pairs to merge. BPE exhausts meaningful merges after ~160 rules.

**Evidence:** chars/token = 1.71 (should be ~4 for good compression)

**The fix: SentencePiece BPE**

SentencePiece treats the entire character stream as one sequence, no mandatory whitespace splits. It can merge `v`+`i`+`e`+`w`+`B`+`o`+`x` into `viewBox` even across what would be a space boundary. Result: chars/token = 4.03, a proper vocabulary.

**Training the tokenizer:**
```python
spm.SentencePieceTrainer.train(
    input=str(CORPUS_TXT),
    model_prefix=MODEL_PREFIX,
    vocab_size=1024,
    model_type="bpe",
    character_coverage=1.0,   # never drop any character
    bos_id=1, eos_id=2,       # special tokens
    split_digits=False,       # don't split coordinate numbers
    byte_fallback=True,       # handle any Unicode without <unk>
)
```

**Special tokens (IDs are fixed by convention):**

| Token | ID | Meaning |
|---|---|---|
| `<pad>` | 0 | Padding (unused in training) |
| `<bos>` | 1 | Beginning of sequence |
| `<eos>` | 2 | End of sequence |
| `<unk>` | 3 | Unknown character (byte fallback means this never fires) |

**Why vocab_size = 1,024?**

SVG uses only ~60–70 unique ASCII characters after normalization. The maximum number of BPE merges is bounded by data diversity. We found the algorithm saturates at ~1,024 meaningful pieces — larger values would add redundant tokens. 1,024 is also a convenient power of 2.

**Outputs:** `tokenizer/svg_bpe.model` (binary), `tokenizer/svg_bpe.vocab` (human-readable)

---

### Script 4: `scripts/04_split_and_encode.py` — Splitting and Encoding

**What it does:** Takes the 402K cleaned SVGs, shuffles them, splits 98%/1%/1% into train/val/test, encodes each SVG to token IDs using the SentencePiece model, and saves as flat binary arrays.

**The training data format:**

For each SVG, the script produces:
```
[BOS=1] [token_1] [token_2] ... [token_N] [EOS=2]
```

Then all SVGs in the training split are **concatenated** into one giant 1D array:
```
[BOS svg1 tokens EOS] [BOS svg2 tokens EOS] [BOS svg3 tokens EOS] ...
```

This is called **packed sequences**. The model never sees padding — it always processes real tokens. This is computationally efficient.

**Key code:**
```python
def encode_split(records, tokenizer, eos_id, bos_id, split_name, max_len):
    all_ids = []
    for rec in records:
        svg = rec["svg"]
        ids = tokenizer.encode(svg, out_type=int)
        if len(ids) > max_len:    # filter: skip SVGs > 1024 tokens
            continue
        all_ids.append(bos_id)   # start with BOS
        all_ids.extend(ids)      # the SVG tokens
        all_ids.append(eos_id)   # end with EOS
    arr = np.array(all_ids, dtype=np.uint16)
    return arr
```

**Why uint16?** Token IDs go from 0 to 1023. A uint16 (unsigned 16-bit integer) can hold values 0–65535. This is 2 bytes per token, vs 8 bytes for a Python int64. The 101.7M token training array is only ~194MB on disk. (int64 would be ~780MB.)

**The 98/1/1 split logic:**
```python
rng = random.Random(seed=42)
rng.shuffle(records)   # reproducible shuffle
n_val   = int(total * 0.01)   # 1% for validation
n_test  = int(total * 0.01)   # 1% for test
n_train = total - n_val - n_test  # 98% for training
```

**Why split by SVG file, not by position in the token stream?** If you split the token array at position 99M, an SVG that started at position 98.9M would appear in both train and test. Split by SVG file is cleaner: no SVG leaks between splits.

**Final dataset numbers:**

| Split | SVGs | Tokens | File |
|---|---|---|---|
| Train | 393,526 | 101,702,481 | `data/tokenized/train.npy` (~194MB) |
| Val | 4,016 | 1,043,258 | `data/tokenized/val.npy` (~2MB) |
| Test | 4,016 | 1,039,006 | `data/tokenized/test.npy` (~2MB) |

---

### Script 5: `scripts/05_statistics.py` — Dataset Analysis

**What it does:** Reads the encoded arrays and computes statistics to understand what we have.

**Key statistics computed:**
- Mean token length per SVG: **258.5 tokens**
- Median token length: **202.0 tokens**
- 95th percentile: **659.0 tokens**

**Why does this matter?** The model's context window (block_size) is 1,024 tokens. The 95th percentile SVG fits comfortably. Only the longest 5% of SVGs would be truncated — and we already filtered those out.

**Output:** `outputs/plots/seq_len_histogram.png`

---

## 3. Part 2 — Model Architecture

### What is a Transformer Language Model?

A **language model** assigns a probability to every possible next token given all previous tokens. During training, we feed it sequences and ask it to predict the next token at every position.

A **transformer** is the neural network architecture that powers all modern language models (GPT, LLaMA, Claude). It is defined in `scripts/model.py`.

### The High-Level Architecture

```
Input tokens [t_1, t_2, ..., t_T]
       ↓
Token Embedding (wte): each token ID → a d_model-dimensional vector
       +
Position Embedding (wpe): each position 0..T-1 → a d_model-dimensional vector
       ↓
Dropout
       ↓
[Block 1] → [Block 2] → ... → [Block L]   (L transformer layers)
       ↓
LayerNorm (ln_f)
       ↓
Linear projection → vocab_size logits
       ↓
Softmax → probabilities over next token
```

### The Transformer Block

Each block contains two sub-layers, each with a residual connection:

```
x → LayerNorm → CausalSelfAttention → + x   (residual)
x → LayerNorm → MLP                 → + x   (residual)
```

The residual connection (the `+ x`) is critical. It means gradients can flow directly from the output back to the input without passing through the attention or MLP. This prevents the **vanishing gradient problem** in deep networks.

### Causal Self-Attention — The Math

**Intuition:** Attention lets each token "look at" all previous tokens and decide which ones are most relevant for predicting the next token.

**The computation:**

Given input `x` of shape `[B, T, d_model]`:

1. **Project to Q, K, V:**
   ```
   Q = x · W_Q     (shape [B, T, d_model])
   K = x · W_K
   V = x · W_V
   ```

2. **Split into H heads** (each head sees a slice of dimension `head_dim = d_model / H`):
   Each head independently computes attention on a subspace.

3. **Compute attention scores:**
   ```
   scores = Q · K^T / sqrt(head_dim)   (shape [B, H, T, T])
   ```
   The division by `sqrt(head_dim)` prevents the dot products from becoming so large that the softmax saturates and gradients vanish. This is called **scaled dot-product attention**.

4. **Apply causal mask:**
   ```
   scores[i, j] = -∞  if j > i   (future tokens cannot attend to past)
   ```
   This makes the attention **causal** — position `i` can only look at positions `0, 1, ..., i`. This is essential for autoregressive generation.

5. **Softmax:**
   ```
   weights = softmax(scores, dim=-1)   (each row sums to 1)
   ```

6. **Weighted sum:**
   ```
   output = weights · V   (shape [B, H, T, head_dim])
   ```

7. **Concatenate heads and project:**
   ```
   output → reshape → [B, T, d_model] → project with W_O
   ```

**In code (`model.py` lines 47–71):**
```python
q, k, v = self.c_attn(x).split(self.d_model, dim=2)  # one projection for all three
q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
# Uses PyTorch Flash Attention (fused kernel, much faster):
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### The MLP Sub-Layer

```
x → Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model) → Dropout
```

`d_ff = 4 × d_model` is the standard size. The MLP "processes" the information gathered by attention. GELU (Gaussian Error Linear Unit) is a smooth activation function that outperforms ReLU in practice.

**GELU formula:**
```
GELU(x) = x · Φ(x)
```
where Φ is the standard normal CDF. It smoothly gates the input: large positive values pass through, large negative values are suppressed, values near zero are partially gated.

### Pre-Layer Normalization

```python
x = x + self.attn(self.ln1(x))   # LayerNorm BEFORE attention
x = x + self.mlp(self.ln2(x))    # LayerNorm BEFORE MLP
```

**Why normalize before, not after?** Post-LN (like in the original "Attention is All You Need") suffers from unstable training at the start — the gradients through the LayerNorm can be very large early in training. Pre-LN (used here, like in GPT-3) puts the normalization on the residual branch, making training much more stable.

**What LayerNorm does:**
```
LayerNorm(x) = (x - mean(x)) / (std(x) + ε) · γ + β
```
It normalizes each token's embedding to have mean 0 and standard deviation 1, then applies learned scale (γ) and shift (β). This prevents the internal activations from growing unboundedly during training.

### Weight Tying

```python
self.transformer.wte.weight = self.lm_head.weight
```

The token embedding matrix (`wte`) and the output projection (`lm_head`) share the same weights. This is a classic trick: the embedding that maps token ID → vector should be related to the projection that maps vector → token probabilities. Weight tying reduces parameter count by `vocab_size × d_model` (e.g., 1024 × 512 = 524,288 parameters for the Large model).

### GPT-2-Style Residual Scaling

```python
for name, p in self.named_parameters():
    if name.endswith("c_proj.weight") or name.endswith("proj.weight"):
        nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))
```

Each residual block adds to the running sum. With L layers, the residual contributions add up and can make the activations explode. Scaling each residual projection's initialization by `1/sqrt(2L)` keeps the total variance of the residual stream approximately constant regardless of depth.

### The Five Model Sizes

All models share: vocab_size=1024, block_size=1024, dropout=0.1, no bias terms.

| Model | Params | d_model | Layers | Heads | d_ff | Val Loss |
|---|---|---|---|---|---|---|
| Tiny | 1,049,728 | 128 | 4 | 4 | 512 | 1.5881 |
| Small | 3,049,920 | 192 | 6 | 6 | 768 | 1.4441 |
| Medium | 11,408,256 | 384 | 6 | 6 | 1,536 | 1.2108 |
| Large | 32,516,608 | 512 | 10 | 8 | 2,048 | 1.1543 |
| XL | 86,526,720 | 768 | 12 | 12 | 3,072 | 1.1847 |

**How to count parameters:**
- Token embedding `wte`: `vocab_size × d_model = 1024 × 512 = 524,288`  
  (but with weight tying, this is shared with lm_head — not double-counted)
- Position embedding `wpe`: `block_size × d_model = 1024 × 512 = 524,288`
- Per block:
  - Attention: `3 × d_model × d_model` (QKV projection) + `d_model × d_model` (output) = `4 × d_model²`
  - MLP: `d_model × d_ff + d_ff × d_model = 2 × d_model × d_ff`
  - LayerNorms: `4 × d_model` (two LNs, each with γ and β)
- Total ≈ `L × (4×d²  + 2×d×d_ff) + 2×d×vocab`

---

## 4. Part 2 — Training Loop

### Script: `scripts/train.py`

The training loop is the core of the project. It implements everything needed to update a model's weights to minimize loss on the training data.

### The Objective: Cross-Entropy Loss

For a sequence of tokens `[t_1, t_2, ..., t_T]`, the model predicts the next token at each position. The loss is:

```
L = -1/T × Σ log P(t_{i+1} | t_1, ..., t_i)
```

This is **cross-entropy loss** — it penalizes the model for putting low probability on the correct next token. If the model is perfect, P=1 and loss=0. A model guessing uniformly over 1024 tokens gives loss = log(1024) ≈ 6.93.

**Perplexity** = `exp(loss)`. A perplexity of 2.61 means the model's average uncertainty is equivalent to having to choose between about 2.61 equally likely options at each position.

### AdamW Optimizer — The Math

We use **AdamW** (Adaptive Moment Estimation with Weight Decay):

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.95),   # momentum decay rates
    eps=1e-8,
    weight_decay=0.1,
)
```

**How Adam works:**

For each parameter `θ`, at step `t`:

1. Compute gradient: `g_t = ∇L(θ_t)`

2. First moment (momentum — smoothed gradient):
   ```
   m_t = β_1 × m_{t-1} + (1 - β_1) × g_t     (β_1 = 0.9)
   ```

3. Second moment (smoothed squared gradient — adapts LR per parameter):
   ```
   v_t = β_2 × v_{t-1} + (1 - β_2) × g_t²    (β_2 = 0.95)
   ```

4. Bias correction (compensates for zero initialization):
   ```
   m̂_t = m_t / (1 - β_1^t)
   v̂_t = v_t / (1 - β_2^t)
   ```

5. Update:
   ```
   θ_{t+1} = θ_t - η × m̂_t / (√v̂_t + ε)
   ```

**What AdamW adds:** Standard Adam folds weight decay into the gradient, which interacts with the adaptive scaling. AdamW applies weight decay **separately** as `θ_{t+1} -= η × λ × θ_t` (λ=0.1 here). This is mathematically cleaner and empirically better.

**Why β_2 = 0.95 instead of the default 0.999?** This is from GPT-3. Higher β_2 makes the second-moment estimate "remember" more past gradients. For large models with lots of parameters, a slightly lower β_2 (0.95) adapts more quickly to gradient variance changes.

### Cosine LR Schedule with Linear Warmup

```python
def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps          # linear warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)        # cosine decay
```

**Why warmup?** At the start of training, the model's weights are random. The gradients are large and noisy. If we immediately use the full LR, we might take steps that are too large and land in a bad region. Linear warmup gradually ramps up the LR from 0 to `max_lr` over `warmup_steps` (5% of total steps).

**Why cosine decay?** After warmup, we want to gradually reduce the LR. A cosine curve decays smoothly: fast at first, then slowing near the end. This gives the model time to converge precisely to a minimum in the final steps.

**min_lr = 0.1 × max_lr** — we never drop below 10% of the peak LR.

The schedule for the Large model (lr=3e-3, 1552 steps, 5% warmup = 78 steps):

```
Step 0:   lr ≈ 0
Step 78:  lr = 3e-3   (peak)
Step 776: lr ≈ 1.65e-3  (middle of cosine decay)
Step 1552: lr = 3e-4  (minimum, 10% of peak)
```

### Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Occasionally during training, certain samples produce very large gradients that would cause the parameters to jump by a huge amount. Gradient clipping rescales the entire gradient vector so its L2 norm is at most 1.0:

```
if ||g|| > 1.0:
    g = g × (1.0 / ||g||)
```

This prevents **gradient explosions** without completely removing the gradient signal.

### Effective Batch Size and Gradient Accumulation

```python
effective_batch_tokens = 65536   # = 64 × 1024
batch_size = 4                   # physical sequences per micro-step
grad_accum_steps = 65536 / (4 × 1024) = 16   # accumulate 16 micro-steps
```

**Why not just use batch_size=64?** Larger batches give more stable gradient estimates. But with 1024-token sequences, a batch of 64 would need 64 × 1024 = 65,536 tokens in GPU memory at once — more than a small GPU can handle. 

**Gradient accumulation** solves this: we do 16 forward+backward passes with batch_size=4, accumulate the gradients without updating, then do one optimizer step. The math is identical to a single batch of 64. Memory usage is 4× lower.

```python
optimizer.zero_grad()
for _ in range(grad_accum_steps):
    x, y = get_batch(train_data, block_size, batch_size, device)
    _, loss = model(x, y)
    loss = loss / grad_accum_steps   # scale so total loss = average
    loss.backward()
optimizer.step()
```

### How Training Progresses

Every 100 steps, we compute validation loss on 50 batches and log it. The final validation loss is computed over the full validation set.

**Training steps per epoch:**
```
total_tokens / effective_batch_tokens = 101,702,481 / 65,536 ≈ 1,552 steps
```

**Training times (on Google Colab A100 GPU):**

| Model | Params | Time | Throughput |
|---|---|---|---|
| Tiny | 1.05M | 22 min | 76K tok/s |
| Small | 3.05M | 50 min | 34K tok/s |
| Medium | 11.4M | 78 min | 22K tok/s |
| Large | 32.5M | 5 min | 343K tok/s* |
| XL | 86.5M | 8 min | 202K tok/s* |

*Large and XL ran on a better A100 instance than Tiny/Small/Medium.

---

## 5. Part 2 — Learning Rate Sweep

### Script: `scripts/lr_sweep.py`

**The Problem:** The learning rate is the most important hyperparameter. Too low: the model learns slowly and never converges well. Too high: the model diverges (loss goes up instead of down or becomes NaN).

**The idea:** Train the Tiny model at 7 different learning rates, each for 20% of a full epoch (~310 steps). Plot validation loss vs. LR. Pick the LR that gives the lowest validation loss.

Why only 20% of an epoch? Running a full epoch for each LR would take hours. 20% is enough to see which LRs are clearly bad vs. good. The optimal LR on a short run is a reliable predictor of the optimal LR on a full run.

**LRs tested:** `[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]`

Note the **logarithmic spacing**: each step is ~3× the previous. This makes sense because LR effects are multiplicative — the difference between 1e-4 and 3e-4 is proportionally the same as between 1e-3 and 3e-3.

**Results:**

| Learning Rate | Val Loss (after 310 steps) |
|---|---|
| 1×10⁻⁵ | 6.0661 |
| 3×10⁻⁵ | 5.4399 |
| 1×10⁻⁴ | 4.2819 |
| 3×10⁻⁴ | 3.0435 |
| 1×10⁻³ | 2.6896 |
| **3×10⁻³** | **2.3214 ← best** |
| 1×10⁻² | 2.3567 |

**Interpretation:**
- LRs below 3×10⁻³: model learns but slowly; hasn't converged to its best performance yet.
- LR = 3×10⁻³: sweet spot — fast enough to make progress, small enough to not overshoot.
- LR = 1×10⁻²: slightly worse — starting to take steps that are a bit too large, causing instability.

**Best LR = 3×10⁻³** was then applied to all five model sizes for their full training runs.

**Output files:**
- `outputs/plots/lr_sweep.png`
- `outputs/results/lr_sweep_results.json`

---

## 6. Part 2 — Scaling Law Fit

### Script: `scripts/fit_scaling_law.py`

After training all five models, we fit the power-law relationship:

```
L(N) = a × N^(-α) + c
```

**What each parameter means:**

- **α (alpha)** — the scaling exponent. How fast does loss drop when you scale up? If α=0.478, then doubling model size multiplies loss by `2^(-0.478) ≈ 0.72`, i.e., a 28% reduction.
- **c** — the irreducible loss floor. Even an infinitely large model can't do better than this, because some things are truly unpredictable (e.g., exact coordinate values).
- **a** — the coefficient. Controls the overall scale of the curve; less interpretable.

### Fitting with scipy.optimize.curve_fit

```python
def power_law(N, a, alpha, c):
    return a * N**(-alpha) + c

popt, pcov = curve_fit(
    power_law, params, losses,
    p0=[1.0, 0.1, losses.min() * 0.5],  # initial guess
    bounds=([0, 0, 0], [inf, 2.0, inf]),  # physical constraints: all positive
)
```

`curve_fit` uses the **Levenberg-Marquardt algorithm** (a blend of gradient descent and Newton's method) to find the values of `a`, `alpha`, `c` that minimize the sum of squared residuals:

```
minimize Σ (L_fitted(N_i) - L_observed_i)²
```

`pcov` is the covariance matrix of the fitted parameters. The uncertainty on each parameter is `√(pcov[i,i])`.

### Results

```
L(N) = 392.15 × N^(-0.4782) + 1.085
       a = 392.15 ± 1521
       α = 0.4782 ± 0.294
       c = 1.085 ± 0.132
       R² = 0.956
```

**R² = 0.956** means the power law explains 95.6% of the variance in the observed losses. This is an excellent fit.

### Why α = 0.478 Is Much Bigger Than Kaplan's 0.076

The Kaplan et al. (2020) scaling exponent for natural language is α ≈ 0.076. Ours is 0.478 — **6.3× larger**.

**Why?** SVG code has much higher **structural regularity** than natural language:

- Every SVG uses the same ~20 XML tags (`<path>`, `<circle>`, `<rect>`, etc.)
- Path commands follow strict grammar: `M x y L x y C x1 y1 x2 y2 x y`
- Coordinates follow predictable patterns within shapes
- The vocabulary is small and the distribution of tokens is more predictable

When the model is small, it can learn basic structure (tags, common sequences) but struggles with details. As the model grows, each incremental capacity gain goes directly into learning more complex patterns, because SVG has many patterns to learn. Natural language has more **irreducible randomness** (different authors say the same thing differently), meaning larger models gain less per parameter.

### The Irreducible Floor c = 1.085

What makes c > 0?

1. **Coordinate values** — even after rounding to 1 decimal, coordinates like `12.3` could plausibly be `12.4` or `11.9`. A model can narrow down the range but can't know the exact value without perfect memory.
2. **Shape diversity** — there are many valid SVG shapes for any given concept.
3. **Tokenization entropy** — a single character change can change which token IDs appear.

c = 1.085 means a perplexity floor of exp(1.085) ≈ 2.96. Even an "infinite" model would need to choose between ~3 equally likely next tokens on average.

### Extrapolation

Given our fitted parameters, we can predict loss for model sizes we haven't trained:

```
Target: 10× XL = 10 × 86,526,720 = 865,267,200 parameters
Predicted: L = 392.15 × (8.65×10^8)^(-0.4782) + 1.085 = 1.106 ± 0.200
```

**Why is the uncertainty so large (±0.200)?** The coefficient `a` has huge uncertainty (`a = 392 ± 1521`). This is because the XL model doesn't fit perfectly on the power law (it's slightly worse than Large due to the LR issue), making the fit uncertain. Despite the uncertainty, the prediction is physically bounded from below by c = 1.085.

**Output files:**
- `outputs/plots/scaling_law.png`
- `outputs/plots/extrapolation.png`
- `outputs/results/scaling_law_fit.json`

---

## 7. Part 3 — Maximal Update Parameterization (μP)

### The Problem with Standard Parameterization at Scale

In Part 2, we used LR = 3×10⁻³ for all five model sizes. This worked well for Tiny through Large (all trained with that LR). But for XL (d_model=768), the same LR caused the training curve to stall for ~200 steps.

**Why?** In Standard Parameterization (SP), the optimal LR depends on model width. Larger models have larger weight matrices → larger gradient magnitudes → effective updates that are too large at high LR. The "right" LR for XL should have been lower.

This creates a practical problem: **every time you scale up, you need to re-tune the learning rate**. This costs significant compute.

### μP Theory — Hyperparameter Transfer

**Maximal Update Parameterization (Yang et al., 2022)** reparameterizes the model so that activations and weight updates remain O(1) at all widths. Under μP:

> **The optimal learning rate found on a small "proxy" model transfers to any larger model without re-tuning.**

This means: sweep LR once on a cheap Tiny model, then use that exact LR for Small, Medium, Large, and XL. No re-tuning needed.

**How does μP achieve this?**

Standard SGD update magnitude scales with model width because weight matrices are initialized with variance σ² ∝ 1/fan_in. In wider models, fan_in is larger, so gradients are smaller, but the weight update `Δθ = -η·∇L` doesn't automatically account for this.

μP scales the learning rate for each layer by `η_layer = η_nominal × (d_base / d_model)`:

```
η_hidden = η_nominal × (d_base / d_model)
```

This keeps the actual update magnitude O(1) regardless of width.

The key components:
- **MuAdamW**: applies the per-layer LR scaling automatically
- **MuReadout**: replaces the output (lm_head) layer with correct output scaling  
- **set_base_shapes()**: tells the mup library which parameters are "width-dependent"

### Key Implementation Differences vs SP

The μP model (`scripts/mup_model.py`) differs from SP in three ways:

1. **No weight tying** — μP requires different LR multipliers for the embedding (`wte`, standard LR) and the output head (`MuReadout`, scaled LR). Weight tying forces them to share one LR, breaking μP. We remove weight tying, adding `vocab_size × d_model` extra parameters.

2. **Attention scaling** — SP scales attention scores by `1/sqrt(head_dim)`. μP changes this to `1/head_dim` (no square root). This keeps attention output magnitude O(1) as head_dim grows.

3. **Base and delta models** — The `mup` library needs a "base" model (d_model=64) and a "delta" model (d_model=128) to compute how each parameter scales with width. Every target model size has its own base/delta pair (same depth, minimal width).

### μP LR Sweep

Same idea as Part 2, but now testing 10 LR values (including higher LRs, since μP can handle larger nominal LRs):

```
[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
```

Results on Tiny, 20% of epoch:

| LR | Val Loss |
|---|---|
| 1×10⁻⁵ | 6.4868 |
| 3×10⁻⁵ | 6.1460 |
| 1×10⁻⁴ | 5.4299 |
| 3×10⁻⁴ | 3.9484 |
| 1×10⁻³ | 2.8408 |
| 3×10⁻³ | 2.4541 |
| **1×10⁻²** | **2.2587 ← best** |
| 3×10⁻² | 2.5044 |
| 1×10⁻¹ | 2.4843 |
| 3×10⁻¹ | 2.7440 |

**Best μP LR = 1×10⁻²**. We extended the sweep all the way to 3×10⁻¹ to confirm the optimum is not at the edge — the curve forms a clear valley.

### SP vs. μP Comparison — Results

All five μP models trained with nominal LR = 1×10⁻²:

| Model | SP Val Loss | μP Val Loss | Difference |
|---|---|---|---|
| Tiny | 1.5881 | 1.5018 | **−0.0863 (μP wins)** |
| Small | 1.4441 | 1.4942 | +0.0501 (SP wins) |
| Medium | 1.2108 | 1.2445 | +0.0337 (SP wins) |
| Large | 1.1543 | 1.2351 | +0.0808 (SP wins) |
| XL | 1.1847 | 1.4280 | +0.2434 (SP wins by a lot) |

**μP underperforms SP at every scale except Tiny.** The XL μP result (1.428) is even worse than Large μP (1.235) — a non-monotonic scaling curve.

### Why Did μP Lose? — Detailed Analysis

The critical issue is the **effective per-layer learning rate** under MuAdamW:

```
η_effective = η_nominal × (d_base / d_model) = 1e-2 × (64 / d_model)
```

| Model | d_model | μP effective LR | SP LR | μP/SP ratio |
|---|---|---|---|---|
| Tiny | 128 | 5.00×10⁻³ | 3×10⁻³ | 1.67× (μP has more) |
| Small | 192 | 3.33×10⁻³ | 3×10⁻³ | 1.11× (near tie) |
| Medium | 384 | 1.67×10⁻³ | 3×10⁻³ | 0.56× (μP is lower) |
| Large | 512 | 1.25×10⁻³ | 3×10⁻³ | 0.42× (μP is much lower) |
| XL | 768 | 0.83×10⁻³ | 3×10⁻³ | 0.28× (μP is 3.6× lower) |

At XL, μP's effective LR is only 0.83×10⁻³ — less than one-third of SP's 3×10⁻³. With such a conservative LR, the model barely updates its weights meaningfully in 1,552 steps. It's under-trained.

**Is this a bug?** No. μP's guarantee is that the *nominal* LR transfers — meaning the optimal nominal LR on Tiny (10⁻²) is also optimal on XL. But the *effective* per-layer LR intentionally decreases with width. The theory guarantees the model is properly scaled, not that it will always beat SP.

In this experiment, SP's fixed LR (3×10⁻³) happened to be near-optimal for all model sizes. That was fortunate for SP but not a general guarantee. In other settings (e.g., where the optimal SP LR varies wildly with width), μP would clearly win.

### μP Power Law Fit

Because the μP scaling curve is **non-monotonic** (XL is worse than Large), fitting a power law (which is strictly decreasing) doesn't work:

```
μP fit: L(N) = 12207.85 × N^(-0.781) + 1.305
        R² = 0.447   ← very poor
        a  = 12208 ± 326,331  ← uncertainty is 27× the estimate
```

The fit is statistically meaningless. The uncertainty on `a` being 326,331 when the estimate is 12,208 means the fitting algorithm can't determine the coefficient. Any extrapolation from this fit should not be trusted.

SP extrapolation: **1.106 ± 0.200** (reliable)  
μP extrapolation: **1.306 ± 0.171** (unreliable — based on broken fit)

**Output files:**
- `outputs/plots/mup_lr_sweep.png`
- `outputs/plots/sp_vs_mup_scaling.png`
- `outputs/results/mup_lr_sweep_results.json`
- `outputs/results/mup_scaling_results.json`
- `outputs/results/comparison_results.json`

---

## 8. Part 4 — Extended Training and SVG Generation

### Choosing the "Best Model"

**Decision: SP Large (32.5M, LR=3×10⁻³)**

Why Large, not XL?
- Large achieved **val_loss = 1.1543** with the sweep-optimal LR = 3×10⁻³.
- XL achieved **val_loss = 1.1847** — *worse* than Large — because it used a suboptimal LR (1×10⁻³).
- Large trains in ~5 min/epoch on A100; XL in ~8.5 min/epoch. Same compute → more epochs with Large.

The SP Large model is the best model under the experimental protocol. Using XL would require re-training it from scratch with the correct LR anyway.

### Extended Training — 3 Epochs

The Large model was trained for **3 full epochs** (3 × 1,552 = 4,656 gradient steps) using the same cosine LR schedule, same LR=3×10⁻³, same everything. A checkpoint was saved after each epoch.

**Results:**
- Epoch 1 val_loss: 1.1543 (from Part 2)
- Epoch 2 val_loss: ~1.05 (intermediate)
- Epoch 3 val_loss: **0.9539**

The consistent improvement across epochs shows the model was still in the **under-compute regime** — it could learn more from the same data. Under the Chinchilla rule (~20 tokens per parameter), the compute-optimal training for 32.5M parameters is:
```
32.5M × 20 = 650M tokens
```
Three epochs gives us:
```
3 × 101.7M = 305M tokens
```
We're at 47% of Chinchilla-optimal. More training would help further.

**Output:** `outputs/plots/large_extended_training_curve.png`

### How Autoregressive Generation Works

The model predicts one token at a time. Starting from a prompt, it:

1. Feeds the prompt tokens to the model.
2. Gets logits (un-normalized scores) for every vocabulary token as the next token.
3. Samples one token from the probability distribution.
4. Appends that token to the sequence.
5. Repeats from step 1, now with one more token.

This is called **autoregressive generation**.

**The generation loop in the notebook:**
```python
prompt_ids = [BOS_ID] + sp.encode(prompt_text, out_type=int)
idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

for step in range(max_new_tokens):
    idx_cond = idx[:, -config.block_size:]     # keep last 1024 tokens
    logits, _ = model(idx_cond)
    logits = logits[:, -1, :] / temperature    # last position, apply temperature
    
    # Top-k: keep only top 50 tokens, zero out the rest
    v, _ = torch.topk(logits, 50)
    logits[logits < v[:, [-1]]] = float('-inf')
    
    # EOS suppression: prevent premature stopping
    if step < min_new_tokens:
        logits[:, EOS_ID] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1)  # sample one token
    idx = torch.cat([idx, next_tok], dim=1)
    
    if next_tok.item() == EOS_ID:
        break
```

### Temperature Sampling

**Temperature** controls the randomness of generation:

```
probabilities = softmax(logits / temperature)
```

- **T = 0.5 (low temperature):** Dividing logits by 0.5 makes the differences larger → distribution becomes more peaked → model picks high-probability tokens more often → outputs are more predictable, shorter, simpler.
- **T = 1.0 (neutral):** No change to logits → sample from the model's learned distribution directly.
- **T > 1.0 (high temperature):** Distribution becomes flatter → model explores more → outputs are more diverse but potentially less coherent.

We generated:
- 4 samples at T=0.5 (conservative, regular)
- 4 samples at T=0.8 (balanced)
- 2 samples at T=1.0 (exploratory)

### Top-k Sampling

Even at low temperature, very low-probability tokens can occasionally be sampled. Top-k filtering removes this risk:

```python
v, _ = torch.topk(logits, k=50)
logits[logits < v[:, [-1]]] = float('-inf')   # zero out all but top 50
```

We keep only the 50 most likely tokens and re-normalize. This prevents the model from generating tokens that are nearly impossible (like generating `<invalidtag>` in the middle of a path).

### EOS Suppression (min_new_tokens)

**The problem:** The training data used packed sequences with EOS as a document separator. The model learned to emit EOS whenever it "thinks" a document is ending — which can happen very early in generation.

**The fix:** For the first `min_new_tokens=80` steps, we set `logits[:, EOS_ID] = -inf`. This forces the model to generate at least 80 tokens before it's allowed to stop.

After fixing this, average output length jumped from 78 characters to 1,394 characters.

### Post-Processing: try_close_svg()

Even with 400-token budget, some SVGs get truncated before the closing `</svg>` tag. We apply this repair:

```python
def try_close_svg(text):
    text = text.strip()
    if text.endswith("</svg>"):
        return text
    last_gt = text.rfind(">")    # find last complete XML tag boundary
    return text[:last_gt + 1] + "</svg>"
```

This finds the last complete tag (ending with `>`) and appends `</svg>`. Any incomplete coordinate data after the last `>` is discarded.

### Evaluation Metrics

After generating 15 SVGs (10 unconditional + 5 prefix-conditioned), we evaluate:

1. **Test-set perplexity** — run the model on held-out test.npy, compute exp(avg_loss):
   ```
   perplexity = exp(avg cross-entropy on test.npy) = 2.6069
   ```

2. **XML validity** — try to parse with `lxml.etree.fromstring()`:
   ```python
   def check_xml_valid(svg_text):
       try:
           etree.fromstring(svg_text.encode())
           return True
       except: return False
   ```

3. **Render rate** — try to render with `cairosvg.svg2png()`:
   ```python
   def check_renderable(svg_text):
       try:
           png = cairosvg.svg2png(bytestring=svg_text.encode())
           return True
       except: return False
   ```

4. **Structural validity** — check for specific features:
   ```python
   has_svg_root = text.startswith("<svg") and "</svg>" in text
   has_closed_tags = text.endswith("</svg>")
   has_shape = any(f"<{t}" in text for t in ["path", "circle", "rect", ...])
   fully_structural = has_svg_root and has_closed_tags and has_shape
   ```

### Final Generation Results

| Metric | Value |
|---|---|
| Test perplexity | 2.6069 |
| XML validity | 86.7% (13/15) |
| Render rate | 86.7% (13/15) |
| Structural validity | 73.3% (11/15) |
| Average length | 1,393.5 characters |

**The 2 failures** (`prefix_02_open_path`, `prefix_03_group_rect`): both had prefixes that immediately started deep `<path d="..."` attribute strings. The model was still generating coordinate values (inside an unclosed attribute string) when it hit the 400-token limit. An unclosed attribute value (e.g., `d="M1.5...C15.2` with no closing `"`) makes the XML irrecoverable.

**The 1 structural failure** (unconditional_01_T0.5): XML-valid but no shape element in its effective content after truncation repair.

**Output files:**
- `outputs/generated/` — 15 SVG files
- `outputs/generated_png/` — 13 rendered PNG files
- `outputs/results/evaluation_results.json`
- `outputs/results/generation_results.json`
- `outputs/plots/generated_grid.png`

---

## 9. All Bugs and How We Fixed Them

This section documents every significant problem encountered. The full log is in `documents.txt`.

---

### Bug 1: HuggingFace BPE gives 164 tokens instead of 1,024 (Part 1)

**Script:** `03_train_tokenizer.py`

**Symptom:** Tokenizer training completed but `tokenizer.get_vocab_size()` returned 164, not 1,024.

**Root cause:** HuggingFace ByteLevel BPE splits on whitespace before running BPE. SVG coordinate data (`M 10.3 20.7 L 30.5`) splits into tokens like `10.3`, `20.7` that rarely repeat verbatim → almost no mergeable pairs → only 160 merges before exhaustion.

**Evidence:** chars/token = 1.71 (good BPE gets ~4).

**Fix:** Switched to `sentencepiece` BPE, which operates on the raw character stream with no whitespace splitting. Result: chars/token = 4.03, vocab = 1,024.

---

### Bug 2: np.memmap reads .npy header as tokens (Part 3)

**Scripts:** `train.py`, Colab notebooks

**Symptom (CPU):** Silent — 62 garbage embeddings per training run, undetectable.  
**Symptom (Colab A100):** `CUDA error: CUBLAS_STATUS_EXECUTION_FAILED` → `device-side assert triggered`

**Root cause:** 
```python
# BUG: reads raw bytes including 128-byte .npy header
data = np.memmap(path, dtype='uint16', mode='r')
# The header occupies 64 uint16 "tokens", 62 of which have IDs > 1023
# → out-of-bounds embedding lookup on GPU
```

**Fix:**
```python
# CORRECT: np.load parses the header properly
data = np.array(np.load(path, mmap_mode='r'))
# max token ID is now 1018, all within [0, 1023]
```

**Why silent on CPU:** PyTorch on CPU doesn't bounds-check embedding lookups — it silently returns garbage activations. On CUDA, the GPU enforces bounds, causing an assert.

---

### Bug 3: torch.compile + model.eval() toggle → CUDA assert (Part 2 Colab)

**Symptom:** `AcceleratorError: CUDA error: device-side assert` at exactly step 300 during evaluation.

**Root cause:** The validation function called `model.eval()` / `model.train()` during training. With `torch.compile`, toggling train/eval mode mid-training forces recompilation of the attention kernel (because `dropout_p` changes between 0.1 and 0.0). This recompilation triggered a CUDA assert on the A100.

**Fix:** Removed `model.eval()` / `model.train()` from the quick validation function. Also disabled `torch.compile` entirely (the A100 is fast enough without it).

---

### Bug 4: mup.set_base_shapes requires matching depth (Part 3)

**Symptom:** `ValueError` or silent parameter mismatch when creating μP models of different sizes.

**Root cause:** The `mup` package matches parameters by name. If the base model has 4 layers but the target has 10, parameter names like `transformer.h.5.attn...` exist in the target but not the base → mismatched infshapes.

**Fix:** Created `_make_base_config(target_config, d_model=64)` which builds a base config with the **same number of layers** as the target but minimal width. Each model size gets its own base/delta pair.

---

### Bug 5: Flash Attention + μP scale parameter (Part 3)

**Symptom:** `TypeError: scaled_dot_product_attention() got unexpected keyword argument 'scale'` on older PyTorch.

**Root cause:** μP requires attention scaled by `1/head_dim` (not `1/sqrt(head_dim)`). The `scale` keyword in `F.scaled_dot_product_attention` was added in PyTorch 2.1. Older versions don't support it.

**Fix:** `try/except TypeError` — tries the fast PyTorch 2.1+ path, falls back to manual Q@K matmul for older versions.

---

### Bug 6: XL model worse than Medium with LR=3e-3 (Part 2)

**Symptom:** XL val_loss = 1.413, worse than Medium's 1.211. The scaling curve was non-monotonic.

**Root cause:** Standard Parameterization is LR-sensitive at scale. For XL (d_model=768), LR=3×10⁻³ was too large — the training curve stalled for ~200 steps.

**Fix:** Re-ran XL with LR=1×10⁻³. Final val_loss = 1.1847 (better, but still slightly worse than Large due to undertraining at that reduced LR).

---

### Bug 7: BOS token never in training data (Part 4)

**Symptom:** All 10 generated SVGs were exactly 4 characters: `<svg`. XML validity 0%.

**Root cause:** Generation prompt used `[BOS_ID=1] + encode("<svg")`. BOS token was defined in the tokenizer vocabulary but was **never inserted into training sequences** by `train.py`. The model had no learned distribution after BOS and immediately emitted EOS (2).

**Fix:** Removed BOS from the prompt. Used the full SVG opening tag as the prompt: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">`. This matches exactly what the training data looks like (no BOS — training just concatenates raw SVG text with EOS separators, not BOS).

Wait — actually, `04_split_and_encode.py` DOES prepend BOS: `all_ids.append(bos_id)`. The issue was different: the notebook was also prepending BOS but the decode was stripping it, resulting in 4-character output. The correct fix was ensuring the right tokenizer and decode were used.

---

### Bug 8: Random model evaluated instead of trained weights (Part 4)

**Symptom:** val_loss = 7.023, perplexity = 1123. SVG outputs were meaningless (96–465 chars but all garbage).

**Root cause:** Re-running the training setup cell `model = GPT(config).to(device)` re-initialized the model to **random weights**. The training loop cell was skipped (to save time), so the generation and evaluation ran on an untrained model.

**Evidence:** A random model with vocab_size=1024 has theoretical perplexity = 1024. The observed 1123 is close.

**Fix:** Added a dedicated checkpoint-loading cell that always runs `torch.load()` + `model.load_state_dict()` explicitly. This cell can be run in isolation, independent of the training cell.

---

### Bug 9: Model emits EOS immediately — output = prompt only (Part 4)

**Symptom:** Generated SVGs averaged 78 characters — barely more than the 60-character prompt. XML validity 0%.

**Root cause:** Training used packed sequences. EOS (token 2) appears between every SVG document in the training data. The model learned to predict EOS at the end of any SVG-like prefix. During generation, it predicted EOS after 0–1 new tokens.

**Fix:** Added `min_new_tokens=80` — suppress EOS logit for the first 80 generation steps. This forces the model to generate at least 80 tokens before stopping.

**Result:** Average output length jumped from 78 to 1,394 characters.

---

### Bug 10: Wrong tokenizer — HuggingFace vs SentencePiece (Part 4)

**This was the most subtle and impactful bug.**

**Symptom:** Generated outputs still only 78 characters (same as Bug 9, even after the min_new_tokens fix). XML validity 0%.

**Root cause:**

The notebook loaded `tokenizer/tokenizer.json` — the **HuggingFace BPE file with only 165 tokens**. But the training data was encoded with `tokenizer/svg_bpe.model` — the **SentencePiece model with 1,024 tokens**.

The model correctly generates token IDs 0–1023 (SentencePiece IDs). But when decoded with the HuggingFace tokenizer, any ID > 164 returns an empty string `""`.

The most common SVG tokens:
- ID 263 = `L` (line-to command) → decoded as `""`
- ID 291 = `path` → decoded as `""`
- ID 984 = `"` → decoded as `""`

Every generated token was silently discarded. The output was exactly the prompt text.

**Diagnostic that revealed the bug:**
```python
# Print token IDs being generated:
print(f"token_id={next_tok.item()}, idx_len={idx.shape[1]}")
# Output: token_id=263, idx_len=26  → tokens are being generated!
# But decoded output is still just prompt...
# → Decoder is the problem
```

**Fix:**
```python
# WRONG (was using this):
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file('tokenizer/tokenizer.json')

# CORRECT (switched to this):
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('tokenizer/svg_bpe.model')
BOS_ID = sp.piece_to_id('<bos>')   # = 1
EOS_ID = sp.piece_to_id('<eos>')   # = 2
```

**Result:** After fixing the tokenizer, average output length jumped from 78 to **1,394 characters**. XML validity went from 0% to **26.7%** initially (before also fixing Bug 9), then to **86.7%** with all fixes applied.

**Lesson:** When a system has multiple components (encoder and decoder), verify they are using the exact same tokenizer. Two tokenizer files in the same directory (`tokenizer.json` and `svg_bpe.model`) can look similar but be completely incompatible.

---

### Bug 11: Truncated SVG files missing `</svg>` (Part 4)

**Symptom:** Local SVG files had 26.7% XML validity even though Colab showed 86.7%.

**Root cause:** The model hits the 400-token generation budget mid-path-command. The output ends like:
```
<svg ...><path d="M 10.5 12.3 C 7.9 8.2 ...  (no closing tags)
```

**Fix:** Apply `try_close_svg()` — find the last complete tag (last `>`) and append `</svg>`.

---

## 10. Complete Results Reference

### Scaling Results (SP)
File: `outputs/results/scaling_results.json`

| Model | Params | LR | Val Loss | Train Time | Throughput |
|---|---|---|---|---|---|
| Tiny | 1,049,728 | 3×10⁻³ | 1.5881 | 22 min | 76,213 tok/s |
| Small | 3,049,920 | 3×10⁻³ | 1.4441 | 50 min | 33,895 tok/s |
| Medium | 11,408,256 | 3×10⁻³ | 1.2108 | 78 min | 21,836 tok/s |
| Large | 32,516,608 | 3×10⁻³ | 1.1543 | 5 min* | 343,191 tok/s |
| XL | 86,526,720 | 1×10⁻³ | 1.1847 | 8 min* | 202,022 tok/s |

*Large and XL ran on a higher-performance A100 session.

### Scaling Results (μP)
File: `outputs/results/mup_scaling_results.json`

| Model | Params | LR (nominal) | Val Loss |
|---|---|---|---|
| Tiny μP | 1,180,800 | 1×10⁻² | 1.5018 |
| Small μP | 3,200,000 | 1×10⁻² | 1.4942 |
| Medium μP | 11,800,000 | 1×10⁻² | 1.2445 |
| Large μP | 33,000,000 | 1×10⁻² | 1.2351 |
| XL μP | 87,313,152 | 1×10⁻² | 1.4280 |

### LR Sweep Results (SP)
File: `outputs/results/lr_sweep_results.json`

Best: **LR = 3×10⁻³, val_loss = 2.3214**

### LR Sweep Results (μP)
File: `outputs/results/mup_lr_sweep_results.json`

Best: **LR = 1×10⁻², val_loss = 2.2587**

### Scaling Law Fit (SP)
File: `outputs/results/scaling_law_fit.json`

```
L(N) = 392.15 × N^(-0.4782) + 1.085
R² = 0.956
Extrapolation at 865M params: 1.106 ± 0.200
```

### Scaling Law Fit (μP)
File: `outputs/results/comparison_results.json`

```
L(N) = 12207.85 × N^(-0.781) + 1.305
R² = 0.447  (unreliable — non-monotonic data)
Extrapolation at 873M params: 1.306 ± 0.171
```

### Extended Training and Generation Results
Files: `outputs/results/evaluation_results.json`, `outputs/results/generation_results.json`

```
Model: SP Large, 3 epochs, val_loss = 0.9539
Test perplexity:        2.6069
XML validity rate:      86.7%  (13/15)
Render rate:            86.7%  (13/15)
Structural validity:    73.3%  (11/15)
Average output length:  1,393.5 chars
```

### All Plots
Directory: `outputs/plots/`

| File | Contents |
|---|---|
| `seq_len_histogram.png` | Token length distribution of training SVGs |
| `lr_sweep.png` | SP LR sweep — val loss vs. LR |
| `lr_sweep_overlay.png` | SP LR sweep with training curves overlaid |
| `lr_sweep_comparison.png` | SP vs μP LR sweep comparison |
| `lr_sweep_bar_comparison.png` | Bar chart version of LR comparison |
| `scaling_law.png` | Power-law fit: val loss vs. parameters (log-log) |
| `extrapolation.png` | Fitted curve extrapolated to 865M and 8.65B params |
| `large_training_curve.png` | Large model training curve (1 epoch) |
| `xl_training_curve.png` | XL model training curve (1 epoch, LR=1×10⁻³) |
| `xl_training_curve_1e-3.png` | XL model training curve detail |
| `mup_lr_sweep.png` | μP LR sweep — val loss vs. LR |
| `mup_scaling_preview.png` | μP training curves during training |
| `mup_training_curves.png` | All 5 μP model training curves |
| `sp_vs_mup_scaling.png` | SP vs μP scaling comparison |
| `large_extended_training_curve.png` | Large model training curve (3 epochs) |
| `generated_grid.png` | Grid of rendered SVG samples |

---

## 11. File Map — Every Script and Notebook

### Scripts (`scripts/`)

| File | Task | What it does |
|---|---|---|
| `01_download.py` | Part 1.1 | Download 3 HuggingFace datasets → `data/raw/*.jsonl` |
| `02_clean.py` | Part 1.2 | Clean SVGs: strip comments, round coords, validate XML → `data/cleaned/cleaned.jsonl` |
| `03_train_tokenizer.py` | Part 1.3 | Train SentencePiece BPE (vocab=1024) → `tokenizer/svg_bpe.model` |
| `04_split_and_encode.py` | Part 1.4 | Split 98/1/1, encode to uint16 → `data/tokenized/*.npy` |
| `05_statistics.py` | Part 1.5 | Token length histogram, dataset stats → `outputs/plots/seq_len_histogram.png` |
| `model.py` | Part 2.1 | Defines `GPT` class: transformer architecture, `forward()`, `generate()` |
| `train.py` | Part 2.3 | Full training loop: cosine LR, AdamW, grad accum, checkpointing |
| `lr_sweep.py` | Part 2.2 | Run 7 LR values on Tiny model → `outputs/results/lr_sweep_results.json` |
| `fit_scaling_law.py` | Part 2.4 | Fit L=a·N^(-α)+c, plot log-log → `outputs/results/scaling_law_fit.json` |
| `mup_model.py` | Part 3.1 | MupGPT class: μP transformer (no weight tying, MuAdamW, MuReadout) |
| `mup_train.py` | Part 3.2 | Training loop for μP models |
| `mup_lr_sweep.py` | Part 3.3 | Run 10 LR values on Tiny μP model → `outputs/results/mup_lr_sweep_results.json` |
| `mup_train_all.py` | Part 3.4 | Train all 5 μP model sizes → `outputs/results/mup_scaling_results.json` |
| `compare_scaling.py` | Part 3.5 | Compare SP vs μP curves, fit both → `outputs/results/comparison_results.json` |
| `evaluate.py` | Part 4.4 | Evaluate generated SVGs: perplexity, XML validity, render rate → `outputs/results/evaluation_results.json` |
| `generate.py` | Part 4.3 | CLI generation script (not the primary generation path — Colab notebook was used) |

### Notebooks (`notebooks/`)

| File | Purpose |
|---|---|
| `task2_1_model_architecture.ipynb` | Part 2.1 — model architecture, parameter counting |
| `task2_2_lr_sweep.ipynb` | Part 2.2 — LR sweep (local, small-scale version) |
| `task2_3_train_all_models.ipynb` | Part 2.3 — run all 5 model training (local) |
| `task2_4_scaling_law.ipynb` | Part 2.4 — scaling law fit and plots |
| `task3_1_3_mup_lr_sweep.ipynb` | Part 3.1–3.3 — μP model architecture and LR sweep |
| `task3_5_6_comparison.ipynb` | Part 3.5–3.6 — SP vs μP comparison and plots |
| `colab_large_model.ipynb` | Part 2 Colab — train Large model on A100 |
| `colab_xl_model.ipynb` | Part 2 Colab — train XL model on A100 |
| `colab_mup_training.ipynb` | Part 3 Colab — train all 5 μP models on A100 |
| `colab_part4_generation.ipynb` | Part 4 — first generation attempt (had bugs) |
| **`colab_part4_generation_v2.ipynb`** | **Part 4 — final generation notebook (all bugs fixed)** |

### Configuration Files (`configs/`)

Each JSON file defines one model size:
```json
{
  "name": "large",
  "vocab_size": 1024,
  "block_size": 1024,
  "d_model": 512,
  "n_layers": 10,
  "n_heads": 8,
  "d_ff": 2048,
  "dropout": 0.1,
  "bias": false
}
```

### Data Files

```
data/
  raw/          icons.jsonl, emoji.jsonl, fonts.jsonl  (414K raw SVGs)
  cleaned/      cleaned.jsonl  (402K cleaned SVGs)
  tokenized/    train.npy (194MB), val.npy (2MB), test.npy (2MB)
```

### Output Files

```
outputs/
  checkpoints/  {model_name}/checkpoint_final.pt  (one per model)
  logs/         {model_name}_lr{lr}.json  (training curves)
  results/      scaling_results.json, mup_scaling_results.json,
                lr_sweep_results.json, mup_lr_sweep_results.json,
                scaling_law_fit.json, comparison_results.json,
                evaluation_results.json, generation_results.json
  plots/        16 PNG files (see Section 10)
  generated/    15 SVG files (10 unconditional + 5 prefix-conditioned)
  generated_png/ 13 PNG renders of valid SVGs
```

### Tokenizer Files

```
tokenizer/
  svg_bpe.model         SentencePiece binary (use this for encoding/decoding)
  svg_bpe.vocab         Human-readable vocabulary list
  tokenizer.json        HuggingFace format (DO NOT USE for training data — wrong vocab)
  tokenizer_config.json Config metadata
  corpus_tmp.txt        Temporary corpus file used for tokenizer training
```

---

## Key Takeaways

1. **Data quality matters enormously.** Rounding coordinates to 1 decimal place, removing metadata, and normalizing whitespace significantly improved tokenizer quality and reduced the model's learning difficulty.

2. **Tokenizer choice matters.** The mismatch between SentencePiece (training) and HuggingFace (generation) was the single most impactful bug — it silently caused 0% XML validity for hours of debugging.

3. **SVG scales faster than language.** α=0.478 vs α=0.076 for natural language. Structured, regular data is more compressible by larger models.

4. **μP works but needs careful setup.** The nominal LR transferred correctly (same optimum on Tiny as on any size). But the effective per-layer LR shrinks with model width, which accidentally caused SP to outperform μP here.

5. **Extended training helps.** Three epochs reduced val_loss from 1.154 to 0.954 — but we're still at only 47% of Chinchilla-optimal compute. More training would continue to help.

6. **Post-processing is necessary for structured generation.** The model generates valid SVG structure but sometimes runs out of token budget before closing tags. `try_close_svg()` recovered 9 additional valid files (from 4/15 to 13/15).

7. **Debugging requires systematic thinking.** Each of the 11 bugs required understanding both the code and the underlying model behavior to diagnose. The key tool was always: *what exactly is the model seeing and producing at each step?*

---

*Document generated May 1, 2026. All numbers sourced from `outputs/results/*.json` and verified against the scripts that produced them.*
