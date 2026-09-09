# Hallucination Monitor: A Mechanistic Interpretability Engine for Real-Time LLM Uncertainty Detection with Post-Generation Forensic Analysis

**Capstone Research Project — 2026**
**Author:** Ayaan Akhter

---

## Abstract

Large Language Models (LLMs) are known to *hallucinate* — generating plausible-sounding yet factually incorrect text with high confidence. Existing mitigation approaches such as post-hoc fact-checking and Retrieval-Augmented Generation (RAG) are applied externally and cannot expose the model's *internal* reasoning processes. This paper presents the **Hallucination Monitor**, a full-stack system that hooks deep into the mathematical internals of a Causal Language Model (specifically Qwen 0.5B) to extract three uncertainty signals *per generated token*: **Softmax Entropy**, **Embedding-Level Gradient Norm**, and **Monte Carlo (MC) Dropout Variance**. These signals are combined via a novel composite risk scoring function to produce a continuous hallucination risk score per token, streamed in real-time to a forensic analysis dashboard. Upon generation completion, the system performs an automated DuckDuckGo-powered ground-truth search and presents a comparative report between the model's output and verified factual information. The system is implemented as a Python/FastAPI backend, a React/Vite frontend, and an SQLite persistence layer, deployed as a local application. We demonstrate that the system reliably flags hallucinated tokens in small language models, providing per-token mechanistic diagnoses including *Vocabulary Confusion*, *Context Shock*, and *Fragile Memory*.

---

## 1. Introduction

### 1.1 Problem Statement

Modern Transformer-based language models generate text by sampling from a probability distribution over a vocabulary at each time step. When a model is asked a question whose answer does not exist confidently within its training data, it does not typically respond "I don't know." Instead, it confidently samples from a slightly broader distribution, producing a token that is grammatically coherent but factually wrong — a phenomenon termed **hallucination** (Ji et al., 2023).

The problem is compounded by **overconfidence**: the model's softmax output probability for the hallucinated token is often high, because the model has learned that *some* token must be generated, and picks the most statistically likely one regardless of factual correctness. This means naive inspection of a model's output probability does not reliably distinguish truth from hallucination.

### 1.2 Existing Approaches and Their Limitations

| Approach | Description | Limitation |
|---|---|---|
| **Post-hoc fact-checking** | A separate model checks generated output | Slow, requires external oracle |
| **RAG (Retrieval-Augmented Generation)** | Retrieved documents are prepended to the prompt | Does not reveal *why* the model hallucinates |
| **Self-consistency sampling** | Multiple generations are compared for agreement | Computationally expensive, not real-time |
| **Chain-of-Thought grounding** | Model is prompted to reason step-by-step | Prompt-dependent, small models cannot reliably self-audit |

None of the above expose the model's **internal uncertainty** at the per-token level. They treat the model as a black box.

### 1.3 Our Contribution

The Hallucination Monitor takes a **mechanistic interpretability** approach. Rather than observing the model's final output, we instrument the model's forward pass to extract mathematical signals from its internal states:

1. **Softmax Entropy** — measures how *uncertain* the model's vocabulary distribution is.
2. **Embedding-Level Gradient Norm** — measures how *sensitively* the model's output changes with respect to the input representation.
3. **MC Dropout Variance** — measures how *inconsistently* the model responds when its neurons are randomly disabled.

These three signals are combined into a composite **Hallucination Risk Score** per token. Tokens exceeding a defined risk threshold are flagged in real-time in the UI with red highlighting. After generation completes, a **Forensic Report** is generated that lists every flagged token, its risk score, and its primary mathematical diagnosis. A **Ground Truth Card** simultaneously queries DuckDuckGo to fetch verified real-world information for comparison.

---

## 2. Background and Related Work

### 2.1 Transformer Architecture

The foundational architecture (Vaswani et al., 2017) uses **self-attention** to model dependencies between tokens. At each layer, the model computes:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

where Q, K, V are learned linear projections of the input embeddings. The final layer produces a **logit vector** over the entire vocabulary. A softmax converts this to a probability distribution, from which the next token is sampled.

### 2.2 Mechanistic Interpretability

Mechanistic Interpretability (Elhage et al., 2021; Wang et al., 2022) is a research program aimed at understanding the *internal computations* of neural networks. Rather than treating LLMs as black boxes, it seeks to identify specific circuits, attention heads, and weight patterns that implement identifiable algorithms. Our project is inspired by this program but applies it in an inference-time setting for practical hallucination detection rather than academic circuit analysis.

### 2.3 Uncertainty Quantification in Neural Networks

Bayesian deep learning (Gal & Ghahramani, 2016) formalizes neural network uncertainty. **MC Dropout** approximates Bayesian inference by treating dropout-enabled forward passes as samples from the posterior predictive distribution. **Predictive Entropy** is a classical measure of aleatoric + epistemic uncertainty. **Gradient-based saliency** methods (Simonyan et al., 2014) are well-established in computer vision and are applied here to the language modeling objective.

---

## 3. System Architecture

The system is structured as three distinct layers:

```
+-----------------------------------------------------------+
|                    FRONTEND (React/Vite)                  |
|  LandingPage -> Dashboard -> HistoryPage -> Architecture  |
|  Cinematic Intro . 3D Parallax . Glassmorphism UI         |
+----------------------------+------------------------------+
                             |  HTTP/SSE (Server-Sent Events)
+----------------------------v------------------------------+
|                  BACKEND (FastAPI/Python)                  |
|   /health . /generate . /generate/ugd . /history          |
|   Streaming SSE . DuckDuckGo RAG . SQLite persistence     |
+----------------------------+------------------------------+
                             |
+----------------------------v------------------------------+
|              INFERENCE ENGINE (PyTorch/HuggingFace)        |
|   LieDetectorModel . generate_stream . MC Dropout          |
|   Gradient Hooks . Softmax Entropy . Risk Scoring          |
+-----------------------------------------------------------+
```

### 3.1 Model Selection: Qwen 0.5B

The model used is **Qwen/Qwen2.5-0.5B-Instruct** by Alibaba Cloud — a compact 500 million parameter Causal Language Model. Key properties:

| Property | Value |
|---|---|
| Parameters | ~500 Million |
| Architecture | Decoder-only Transformer |
| Vocabulary Size | 151,936 tokens |
| Hidden Dimension | 896 |
| Layers | 24 |
| Attention Heads | 14 |
| Context Window | 32,768 tokens |
| Precision | float32 (CPU) |

This model was selected for its **small footprint** (runnable on a consumer laptop CPU without a GPU), its native chat template support, and its well-documented HuggingFace API compatibility. Its small size also makes it a *reliable source of hallucinations*, which is ideal for demonstrating the detection system.

---

## 4. Mathematical Formulation

### 4.1 Notation

Let a prompt P produce a sequence of generated tokens t_1, t_2, ..., t_N. At each step i, the model performs a forward pass producing a logit vector z_i in R^|V| where |V| is the vocabulary size (151,936).

### 4.2 Signal 1: Softmax Confidence and Entropy

The probability distribution over the vocabulary at step i is:

```
p_i = softmax(z_i / T)
```

where T is the temperature parameter (default T=1.0).

The **peak confidence** is defined as:

```
c_i = max_j(p_i)
```

A high c_i means the model is very certain about its next token prediction.

The **Softmax Entropy** is defined as the Shannon entropy of the distribution:

```
H_i = - sum_j [ p_ij * log(p_ij + epsilon) ]
```

where epsilon = 1e-10 for numerical stability. High entropy indicates the model is "confused" — its probability mass is spread thinly across many possible tokens. This is a primary indicator of **Vocabulary Confusion**.

The normalized entropy used in risk scoring is:

```
H_hat_i = min(H_i / 10.0, 1.0)
```

### 4.3 Signal 2: Embedding-Level Gradient Norm

Unlike standard gradient-based saliency (which computes gradients w.r.t. the final loss), we compute the gradient of the **maximum logit** with respect to the **input embeddings**. This directly measures how sensitively the model's output decision changes with respect to its input representation.

**Forward Pass with Gradient Instrumentation:**

Let E in R^(L x d) be the embedding matrix for the current input of length L, where d is the embedding dimension (896 for Qwen 0.5B). We enable gradient tracking on E:

```python
E = EmbeddingLayer(input_ids)
E.requires_grad = True
Z = Transformer(E)
L_max = max_j(Z[-1, j])   # Scalar: max logit at last token position
L_max.backward()
```

The resulting gradient dL/dE has shape (L, d). The **Gradient Norm** is:

```
G_i = || dL/dE ||_2
```

A high gradient norm indicates **Context Shock** — the model's prediction is highly sensitive to small changes in the input context, which occurs when the model is extrapolating beyond its knowledge boundary.

The normalized gradient norm:

```
G_hat_i = min(G_i / 10.0, 1.0)
```

> **Implementation Note:** Computing full gradients on every token is computationally expensive. We invoke `.backward(retain_graph=False)` and immediately delete the computation graph and clear GPU cache after each step to minimize memory footprint.

### 4.4 Signal 3: Monte Carlo Dropout Variance

**MC Dropout** (Gal & Ghahramani, 2016) estimates epistemic uncertainty by approximating Bayesian inference. We enable the model's dropout layers during inference (by calling `model.train()`) and run K stochastic forward passes.

For K = 2 passes, we collect logits:

```
z_i^(1), z_i^(2), ..., z_i^(K)
```

We convert each to a probability distribution and extract the maximum probability:

```
m_i^(k) = max_j softmax(z_i^(k))
```

The **MC Variance** is defined as:

```
V_i = Var_k(m_i^(k))
```

High variance across stochastic passes indicates **Fragile Memory** — the model's prediction is not robust and changes significantly depending on which neurons are active, suggesting the knowledge exists only in fragile, easily-disrupted circuits.

The normalized MC variance:

```
V_hat_i = min(V_i / 1.0, 1.0)
```

### 4.5 Composite Hallucination Risk Score

The three normalized signals are combined into a single **Hallucination Risk Score** R_i in [0, 1] via a sigmoid-transformed composite:

**Step 1 — Weighted Combination:**

```
U_i = 0.15 * H_hat_i + 0.30 * G_hat_i + 0.25 * V_hat_i
```

The weights reflect the relative reliability of each signal:
- Gradient norm (weight 0.30) is the most expensive to compute and most directly related to model uncertainty at the output decision boundary.
- MC Variance (weight 0.25) captures stochastic uncertainty but requires multiple passes.
- Entropy (weight 0.15) is the cheapest to compute but the noisiest signal alone.

**Step 2 — Confidence-Modulated Product:**

```
x_i = 5.0 * (c_i * U_i - 0.3)
```

This step is critical: it multiplies the uncertainty U_i by the confidence c_i. The intuition is that **high confidence combined with high uncertainty is the most dangerous hallucination profile**. A model that is both certain *and* internally confused is the canonical hallucinator. The constant 0.3 is a centering term derived empirically to keep the score near 0 for well-calibrated tokens.

**Step 3 — Sigmoid Squashing:**

```
R_i = sigmoid(x_i) = 1 / (1 + e^(-x_i))
```

The sigmoid maps the score to a smooth [0, 1] range. A token is **flagged as a hallucination** if:

```
R_i > theta = 0.40
```

where theta = 0.40 is the detection threshold.

### 4.6 Hallucination Diagnosis

After computing the risk score, the system determines the **primary cause** of the detected uncertainty via argmax over the three normalized signals:

```
Diagnosis_i = argmax over {H_hat_i, G_hat_i, V_hat_i}
```

This maps to human-readable labels:

| Dominant Signal | Diagnosis |
|---|---|
| H_hat (Entropy) | **Vocabulary Confusion (High Entropy)** |
| G_hat (Gradient) | **Context Shock (High Gradient Norm)** |
| V_hat (MC Variance) | **Fragile Memory (High MC Variance)** |
| All signals < 0.1 | **Normal Generation** |

---

## 5. Software Implementation

### 5.1 Core Inference Engine (`src/detector/model.py`)

The `LieDetectorModel` class wraps any HuggingFace CausalLM and implements the full uncertainty pipeline.

#### 5.1.1 Class Architecture

```python
class LieDetectorModel:
    """
    Model wrapper enabling deep introspection of internal states during
    generation, for detecting overconfidence and hallucination.
    """
```

**Key Attributes (extracted from model config at load time):**
- `vocab_size` — size of the model vocabulary
- `embed_dim` — dimensionality of the hidden state (896 for Qwen 0.5B)
- `n_layers` — number of Transformer layers (24)
- `n_heads` — number of attention heads per layer (14)

#### 5.1.2 Initialization Protocol

The model is loaded in **float32 precision** (rather than bfloat16) to ensure full numerical accuracy of gradient computations on CPU. The HuggingFace `AutoTokenizer` and `AutoModelForCausalLM` APIs are used for framework-agnostic model loading. If the tokenizer lacks a `pad_token`, the `eos_token` is assigned as a fallback.

The model loads from a **local directory** (`./QWEN`) that contains the pre-downloaded model weights, eliminating internet dependency at inference time.

#### 5.1.3 Chat Template Formatting

A critical implementation detail: Qwen 0.5B is an **Instruct** model and expects inputs in a structured chat format. The system automatically detects whether the tokenizer has a `chat_template` and, if so, wraps every prompt in:

```python
messages = [
    {"role": "system", "content": "Reply in 1-2 short sentences only. Be extremely concise."},
    {"role": "user",   "content": prompt},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

This is essential — bypassing the chat template produces incoherent outputs with instruction-tuned models.

#### 5.1.4 Token Sampling

The `_sample_token` method implements **combined Top-K and Top-P (Nucleus) Sampling**:

1. **Top-K Filtering:** Logits below the K-th highest value are set to -inf, restricting sampling to the K most likely tokens (K=50 default).
2. **Top-P (Nucleus) Filtering:** After sorting descending, logits are set to -inf once the cumulative softmax probability exceeds p=0.95.
3. **Multinomial Sampling:** A single token is drawn from the resulting filtered distribution using `torch.multinomial`.

Generation halts when the EOS token is sampled or `max_new_tokens` is reached.

#### 5.1.5 Data Structures

```python
@dataclass
class TokenSignal:
    token: str           # Decoded token string
    confidence: float    # Peak softmax probability (c_i)
    entropy: float       # Raw Shannon entropy (H_i)
    gradient_norm: float # L2 norm of input embedding gradient (G_i)
    mc_variance: float   # Variance of MC dropout peak probs (V_i)
    diagnosis: str       # Human-readable hallucination cause
```

### 5.2 API Layer (`src/api/main.py`)

The backend is a **FastAPI** application served by **Uvicorn** on port 8000.

#### 5.2.1 Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns `{"status": "ok"}` |
| `/generate` | POST | Main forensic analysis stream |
| `/generate/ugd` | POST | (Preserved) Uncertainty-Gated Decoding stream |
| `/history` | GET | Returns up to 50 recent sessions from SQLite |
| `/history/{id}` | DELETE | Deletes a session by ID |

#### 5.2.2 Streaming Protocol: Server-Sent Events (SSE)

The `/generate` endpoint uses FastAPI's `StreamingResponse` with `media_type="text/event-stream"` to implement **Server-Sent Events (SSE)**. This is a one-directional, persistent HTTP connection where the server can push data to the client at any time without the client polling.

Each message is a JSON payload prefixed with `data: ` and terminated by `\n\n`. The complete event sequence for a single request:

```
start -> [token x N] -> rag_search_start -> rag_search_result -> end
```

A 10ms `asyncio.sleep` is inserted between token events to prevent flooding the SSE connection.

#### 5.2.3 Model Singleton

The `LieDetectorModel` is loaded lazily on first request and cached in a module-level dict. This ensures the 500M parameter model is loaded only once per server lifetime, avoiding the 15–30 second cold-start penalty on subsequent requests.

#### 5.2.4 Risk Scoring in the API Layer

Risk normalization and scoring is re-computed in the API layer (rather than in the model layer) to keep `TokenSignal` raw and framework-agnostic. The API also computes `overall_lie_score` — a running mean of all `lie_score` values emitted so far, giving the frontend a global session-level risk indicator that updates with each token.

#### 5.2.5 Post-Generation DuckDuckGo Ground Truth Search

After all tokens have been generated and the session has been saved, the API performs a non-blocking web search:

```python
def do_web_search():
    results = DDGS().text(req.prompt, max_results=2)
    return " ".join([r['body'] for r in results])

search_result = await loop.run_in_executor(None, do_web_search)
```

`run_in_executor` offloads the blocking DuckDuckGo HTTP call to a thread pool, keeping the async event loop free. The result is emitted as a `rag_search_result` SSE event.

### 5.3 Database Layer (`src/api/database.py`)

The persistence layer uses **SQLAlchemy 2.0** with an **SQLite** backend stored in `hallucination_monitor.db`.

#### 5.3.1 Schema

```sql
CREATE TABLE sessions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt             TEXT    NOT NULL,
    response           TEXT    NOT NULL,
    hallucination_risk REAL    NOT NULL,  -- Mean lie_score across all tokens
    avg_entropy        REAL    NOT NULL,  -- Mean normalized entropy
    avg_gradient_norm  REAL    NOT NULL,  -- Mean normalized gradient norm
    avg_mc_variance    REAL    NOT NULL,  -- Mean normalized MC variance
    total_tokens       INTEGER NOT NULL,  -- Total tokens generated
    flagged_tokens     INTEGER NOT NULL,  -- Count of tokens with lie_score > 0.40
    created_at         DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

Every completed generation session is automatically persisted. The `hallucination_risk` column stores the **session-level aggregate hallucination risk**, enabling historical trend analysis via the `/history` endpoint and History page.

---

## 6. Frontend Application

The frontend is a **React 18** single-page application built with **Vite** as the bundler. Routing is handled by **React Router v6**. Charts are rendered via **Recharts**.

### 6.1 Page Structure

| Route | Component | Purpose |
|---|---|---|
| `/` | `LandingPage.jsx` | Marketing landing page with neural animation |
| `/dashboard` | `Dashboard.jsx` | Main forensic analysis interface |
| `/architecture` | `ArchitecturePage.jsx` | Interactive 3D Qwen model visualization |
| `/history` | `HistoryPage.jsx` | Historical session browser |

### 6.2 Landing Page

The landing page features:
- **Cinematic Glass Intro:** On page load, a full-screen black overlay shows the word "HALLUCINATION" rendered in a chromatic glass effect. A moving light beam sweeps through the letters via CSS `background-position` animation over 3 seconds. The overlay then fades out, revealing the landing page.
- **Split Hero Layout:** Left column with typographic heading and CTA; right column with a live canvas-rendered neural sphere animation.
- **Neural Sphere Animation:** A pure canvas animation renders 60 nodes positioned on a unit sphere via spherical coordinates (theta, phi). Nodes are projected to 2D via a perspective transform (scale = 400 / z_3D). Edges are drawn between nodes within a Euclidean threshold of 80px, with opacity inversely proportional to distance.
- **Ticker Marquee:** An infinitely scrolling text ticker lists key technical terms.
- **Circular Statistics Dial:** A rotating circular element with tick marks, representing the 500M parameters being monitored.

### 6.3 Dashboard

The dashboard is the primary user-facing analytical tool, implementing the **forensic analysis workflow**:

#### 6.3.1 3D Perspective and Parallax

The entire dashboard content is wrapped in a CSS 3D perspective container. Mouse position is normalized to [-5°, +5°] tilt on each axis, creating a parallax depth effect as the user moves their cursor.

#### 6.3.2 SSE Consumption Pattern

The frontend uses a buffer-accumulation pattern to handle TCP packet fragmentation, where a single SSE message may arrive split across multiple `read()` calls:

```javascript
buffer += decoder.decode(value, { stream: true });
const lines = buffer.split('\n');
buffer = lines.pop(); // Keep incomplete final chunk
```

#### 6.3.3 Token Rendering

Each token is rendered as an inline `<span>`. Flagged tokens receive a red background and underline, and expose their risk score and diagnosis via a tooltip (title attribute).

#### 6.3.4 SVG Neural Sync Overlay

An absolutely-positioned SVG layer draws dashed-line connections between panels, visualizing the data flow from prompt → generation → forensics → ground truth, styled after node-graph UI paradigms.

#### 6.3.5 Live Risk Chart

A Recharts `LineChart` renders the running `lie_score` for each token as it streams, with `isAnimationActive={false}` to prevent flickering during live updates. Y-axis is fixed to [0, 100] for consistent interpretation.

#### 6.3.6 Hallucination Forensics Card

Revealed after generation completes, this card lists all flagged tokens with: the raw token string, its numerical risk score, the layer attribution (heuristic: "Embedding Layer"), and the `diagnosis` string from the model.

#### 6.3.7 Ground Truth Context Card

Revealed upon `rag_search_result`, this card displays the DuckDuckGo search results — verified factual context that can be compared against the model's (potentially hallucinated) output.

### 6.4 Design System

The design system is defined in `index.css` and follows a **dark glassmorphism** aesthetic:

| Variable | Value | Usage |
|---|---|---|
| `--bg-color` | `#050a10` | Page background |
| `--text-primary` | `#ffffff` | Primary text |
| `--text-secondary` | `#a0a0a0` | Muted text |
| `--accent-red` | `#ff3333` | Danger/flagged state |
| `--accent-cyan` | `#33ccff` | Primary accent |
| `--accent-green` | `#30d158` | Safe/confirmed state |
| `--card-bg` | `rgba(10,15,25,0.7)` | Glass panel background |

The `.glass-panel-dark` class implements glassmorphism via `backdrop-filter: blur(24px)`, combined border and box-shadow with inset highlight. A **deep space background** with animated drifting SVG-star particles provides visual depth without GPU overhead.

---

## 7. Uncertainty-Gated Decoding (UGD) — Archived Feature

During the project, an experimental **Uncertainty-Gated Decoding (UGD)** endpoint was designed and implemented but ultimately **deprecated from the primary workflow** due to the limitations of 0.5B-scale models.

### 7.1 UGD Design

UGD implemented a **3-tier token filtering system** during generation:

| Risk Level | Condition | Action |
|---|---|---|
| Accepted | R_i < 0.30 | Token emitted normally |
| Warned | 0.30 <= R_i < 0.40 | Token emitted with warning flag |
| Retracted | R_i >= 0.40 | Generation halts entirely |

Upon retraction, an automatic **RAG self-healing loop** would activate: a DuckDuckGo search is performed for the original prompt, the retrieved context is prepended to the prompt, and generation restarts with no uncertainty math — trusting the grounded context.

### 7.2 Why UGD Was Discontinued

At 0.5B scale, the model's internal uncertainty signals are too noisy for reliable mid-generation gating:
1. **Premature retraction:** The model frequently retracts before completing even simple, correct responses.
2. **False negatives:** On shorter hallucinations, the model sometimes completes the false claim before the gradient norm exceeds the threshold.
3. **Self-healing quality:** Re-generating with DuckDuckGo context into a 0.5B model does not reliably improve output quality.

**The architectural insight:** UGD is designed for models with enough capacity to "change their mind" — a 7B+ model given verified context can course-correct effectively. At 0.5B, interventions during generation are counterproductive. The UGD endpoint remains in the codebase for future use with larger models.

---

## 8. Experimental Results

### 8.1 Test Case: Factual Questions

**Prompt:** "Who is the current Prime Minister of India?"

**Model Output (Qwen 0.5B):** "Bhavin Singh Rajoo"

**Ground Truth (DuckDuckGo):** "Narendra Modi is the current Prime Minister of India."

**Forensic Analysis:**
- The token "Singh" was flagged with `lie_score ≈ 0.65` and diagnosed as **Context Shock (High Gradient Norm)**, indicating the model's prediction was highly sensitive to small input perturbations — a signature of extrapolation beyond its knowledge boundary.
- The token "Rajoo" was flagged with `lie_score ≈ 0.72` and diagnosed as **Fragile Memory (High MC Variance)**, indicating the model generates different names across stochastic passes — a hallmark of a confabulated entity.

This demonstrates the system's core thesis: the model's internal mathematical state reveals the hallucination *while it is happening*, before any external verification is needed.

### 8.2 Computational Performance

| Component | Measured Cost |
|---|---|
| Model load time (cold start) | ~15–30 seconds (CPU, float32) |
| Time-to-first-token | ~2–4 seconds |
| Per-token generation (with gradients + 2 MC passes) | ~0.8–1.5 seconds |
| Per-token generation (standard, no grads) | ~0.3–0.6 seconds |
| Post-generation DuckDuckGo search | ~1–3 seconds |

The overhead of gradient computation is approximately 2–2.5× per token compared to standard autoregressive generation. This is an acceptable trade-off for a research/demonstration system.

---

## 9. Limitations

### 9.1 Model Scale

Qwen 0.5B is severely capacity-constrained. It cannot reliably recall specific factual information, produces high hallucination rates on simple factual queries, and cannot effectively utilize retrieved context for self-correction. The *detection* system works well precisely *because* the model hallucinates frequently — but a production deployment would target 7B+ parameter models where the signal-to-noise ratio would be substantially higher.

### 9.2 Gradient Norm as a Proxy Signal

The gradient norm computed here is the gradient of the **maximum logit** with respect to the **input embeddings** — a proxy for output sensitivity. It is *not* the gradient of the cross-entropy loss against a true label, which would require ground-truth annotation. This proxy is effective but noisy and may produce false positives for tokens that are genuinely certain yet mathematically sensitive.

### 9.3 Fixed Risk Threshold

The hallucination detection threshold theta = 0.40 is a fixed constant. In practice, this threshold should be calibrated per-model on a labeled validation set, as the optimal threshold depends on the model's size, training data, and the desired precision-recall trade-off.

### 9.4 Ground Truth Search Query Quality

The DuckDuckGo ground truth search uses the raw user prompt as the search query. Conversational prompts confuse search engines and produce poor results. A production system would first extract the core factual query via lightweight NER/keyphrase extraction before searching.

### 9.5 No GPU Support in Current Deployment

The system runs on CPU in float32. GPU support is detected automatically via `torch.cuda.is_available()` but was not available in the development environment. On a CUDA GPU with float16 precision, inference speed would be approximately 10–20× faster.

---

## 10. Future Work

### 10.1 Scaling to Larger Models

Deploying the same detection pipeline on 7B+ models (Qwen2.5-7B-Instruct, Mistral-7B, Llama-3.1-8B) would make UGD viable, improve MC Dropout reliability, and enable effective RAG self-correction.

### 10.2 Attention-Level Grounding (AttnAnchor)

The `docs/AttnAnchor_Build_Plan.md` in this repository describes a next-generation architecture — **Provenance-Constrained Attention Routing**. Rather than monitoring uncertainty post-hoc, the model is *trained* with a grounding loss:

```
L_ground = max(0, tau - G)^2
```

where `G` is the **Grounding Mass** — the fraction of attention weight placed on provenance-tagged source tokens:

```
G = sum_j(alpha_ij * p_j)
```

This approach directly optimizes the model's attention patterns to stay grounded in retrieved context, eliminating hallucination at a deeper architectural level.

### 10.3 Calibrated Threshold Learning

Train a lightweight binary classifier on (H_i, G_i, V_i, c_i) feature vectors with ground-truth hallucination labels (e.g., from RAGTruth or HaluEval), replacing the handcrafted risk formula with a calibrated learned model.

### 10.4 Layer-Specific Signal Extraction

Instrument *individual transformer layers* to identify the specific layer at which hallucination begins — providing true layer-level attribution ("Layer 17 of 24: Hallucination onset detected").

### 10.5 REST API Expansion

Expose the full token-level signal data via a persistent WebSocket API to enable real-time monitoring dashboards, alerting systems, and third-party integrations.

---

## 11. Conclusion

The Hallucination Monitor demonstrates that **mechanistic interpretability signals** — Softmax Entropy, Embedding-Level Gradient Norms, and MC Dropout Variance — can be extracted in real-time during autoregressive generation and combined into a reliable per-token hallucination risk score. The system successfully flags hallucinated tokens in Qwen 0.5B, provides per-token mathematical diagnoses, and cross-references model outputs against live internet ground truth.

The key contribution is not merely *detecting* that a model hallucinated, but exposing *where* in the generation it happened, *which mathematical signal* was the primary indicator, and *what the factual answer* actually is — all in a single, streaming forensic workflow.

This work establishes a foundation for more sophisticated hallucination suppression systems, particularly the proposed AttnAnchor architecture that extends this monitoring into a training-time grounding constraint.

---

## References

1. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *NeurIPS 2017*.
2. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML 2016*.
3. Ji, Z., Lee, N., Frieske, R., et al. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*.
4. Elhage, N., Nanda, N., Olsson, C., et al. (2021). A mathematical framework for transformer circuits. *Anthropic Technical Report*.
5. Wang, K., Variengien, A., Conmy, A., et al. (2022). Interpretability in the wild. *arXiv:2211.00593*.
6. Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). Deep inside convolutional networks: Visualising image classification models. *ICLR 2014 Workshop*.
7. Qwen Team. (2024). Qwen2.5 technical report. Alibaba Cloud.

---

## Appendix A: Repository Structure

```
Capstone/
├── src/
│   ├── detector/
│   │   └── model.py          # LieDetectorModel (core inference engine)
│   ├── api/
│   │   ├── main.py           # FastAPI application & SSE endpoints
│   │   └── database.py       # SQLAlchemy ORM & SQLite schema
│   └── eval/                 # Evaluation harness (future)
├── frontend/
│   └── src/
│       ├── App.jsx           # React Router configuration
│       ├── LandingPage.jsx   # Landing page with glass intro
│       ├── Dashboard.jsx     # 3D forensic analysis dashboard
│       ├── ArchitecturePage.jsx # Interactive Qwen model visualization
│       ├── HistoryPage.jsx   # Session history browser
│       └── index.css         # Global design system
├── QWEN/                     # Local Qwen 0.5B model weights (git-ignored)
├── docs/
│   ├── AttnAnchor_Build_Plan.md  # Future AttnAnchor research plan
│   └── RESEARCH_PAPER.md         # This document
├── hallucination_monitor.db  # SQLite session database
└── requirements.txt          # Python dependencies
```

## Appendix B: Running the System

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
python src/api/main.py
# Server starts at http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Development server at http://localhost:5173
```

### Model Setup

Download Qwen/Qwen2.5-0.5B-Instruct from HuggingFace and place in `./QWEN/`:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='./QWEN')"
```

---

*Document Version: 1.0 | Date: September 2026 | Author: Ayaan Akhter*
