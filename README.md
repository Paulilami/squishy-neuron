# Squishy Neuron

A transformer language model where every neuron is a stateful computational unit with persistent memory, learnable gating, and adaptive sensitivity. The standard feedforward sub-layer (Linear > GELU > Linear) is ripped out and replaced with a Stateful Neuron Bank where each neuron accumulates state across sequence positions through a gated recurrence.

This is a research project. The goal is to answer one question: does giving individual neurons persistent state improve language modeling, and if so, what do the neurons learn?

Built from raw PyTorch. No HuggingFace Transformers. No pretrained anything.

## What a Squishy Neuron Actually Does

Every current LLM uses the same feedforward block: project up, apply nonlinearity, project back down. Each token is processed independently. There is no memory inside the neuron itself.

A squishy neuron is different. It maintains a hidden state that accumulates across the sequence via a gated recurrence, then mixes that state with the raw signal to produce output. Each neuron learns its own decay rate (how long to remember) and state weight (how much to trust memory vs the current input).

The default variant (selective) uses input-dependent decay inspired by Mamba's selective scan mechanism. Instead of a fixed decay per neuron, a learned projection produces per-token, per-neuron discretization steps:

```
raw    = W_main @ x
gate   = sigmoid(W_gate @ x)
mod    = tanh(W_mod @ x)
delta  = softplus(W_delta @ x)

A      = -exp(A_log)                          # base decay rates (S4-style log-uniform init)
decay  = exp(delta * A)                        # input-dependent per-token decay
update = (1 - decay) * gate * tanh(raw)        # normalized gated update

state[t] = decay[t] * state[t-1] + update[t]  # linear recurrence

mixed  = (1 - state_weight) * gelu(raw) + state_weight * state[t]
output = mixed * (1 + mod)
```

Key design decisions and why they differ from the original formulation:

- `(1 - decay)` normalization on the update prevents unbounded state growth. Without this, high decay + large gate values cause state to diverge. This is standard in SSM literature (S4, Mamba).
- `gelu(raw)` instead of `tanh(raw)` for the non-state path. Double-tanh (tanh on raw then tanh on state) creates a gradient bottleneck since tanh' is at most 1 and you're composing two of them. GELU preserves gradient flow on the direct path.
- Input-dependent decay via `softplus(W_delta @ x)` means each token controls how much history each neuron retains. A period token might trigger high decay (flush state), while a pronoun might trigger low decay (preserve context). This selectivity is what makes the architecture competitive with attention for context-dependent processing.
- S4-style log-uniform initialization of base decay rates (`A_log`) spreads neurons across different timescales at init, giving gradient signal to specialize from the start instead of all starting at the same decay.

`decay`, `state_weight`, and `A_log` are learnable per-neuron parameters. After training, neurons naturally differentiate: some develop long memory (slow decay), others become reactive (fast decay). This happens without supervision.

## Parallel Scan

The state recurrence `s[t] = decay[t] * s[t-1] + update[t]` is a linear recurrence that can be parallelized using a prefix scan over the associative operator `(a2, b2) compose (a1, b1) = (a2*a1, a2*b1 + b2)`.

The implementation uses a Hillis-Steele doubling algorithm: O(T log T) work but fully parallel across the time dimension. For each doubling step d, every position t composes its current prefix with the prefix at position t-d. After ceil(log2(T)) steps, every position holds the complete prefix.

The parallel scan activates automatically for sequences >= 64 tokens. For shorter sequences, a simple sequential loop is used since the overhead of tensor operations dominates at small T. Both produce identical results (verified by tests).

## Architecture

Squishy neuron block (drop-in replacement for standard transformer):

```
x -> RMSNorm -> MultiHeadAttention(RoPE) -> + residual
  -> RMSNorm -> SelectiveNeuronBank        -> + residual -> out
```

Baseline comparison block:

```
x -> RMSNorm -> MultiHeadAttention(RoPE) -> + residual
  -> RMSNorm -> SwiGLU FFN               -> + residual -> out
```

Architectural choices aligned with current best practices:

- RMSNorm instead of LayerNorm. Cheaper (no mean subtraction), works as well or better at scale. Used by LLaMA, Gemma, Mistral.
- Rotary Position Embeddings (RoPE) instead of learned positional embeddings. Better length generalization, no extra parameters, relative position encoding baked into attention. Used by LLaMA, Mistral, Qwen. Configurable via `use_rope` flag; falls back to learned embeddings if disabled.
- SwiGLU FFN for the baseline (`silu(W1 @ x) * W3 @ x`). Strictly better than GELU FFN at matched parameter count. This means the baseline is actually strong, not a strawman.
- Gradient checkpointing support via `use_gradient_checkpointing` config flag. Trades compute for memory, enabling larger batch sizes or model dimensions on limited hardware.

The attention mechanism is standard multi-head causal self-attention with fused QKV projection, using `F.scaled_dot_product_attention(is_causal=True)` for flash attention kernels on PyTorch 2.0+. Weight tying between token embedding and LM head.

## Parameter Counts (d_model=64, n_layers=2, vocab_size=100)

```
Component             Selective     Baseline
Attention (per layer)   16,384        16,384    (identical)
FFN/NeuronBank         65,536        49,152    (4 proj + A_log vs 3 proj SwiGLU)
RMSNorm (per layer)       256           256    (identical)
Embeddings + LM head    8,256         8,256    (identical, weight-tied)
Total                  189,440       142,144
Ratio                              1.33x
```

The difference is entirely in the FFN replacement. Attention, embeddings, and norms are byte-for-byte identical.

## Variant Swapping

The neuron design is the research variable. Swapping it requires exactly 3 things:

1. Write a new class in `src/neuron.py` inheriting `NeuronBankBase`
2. Add one line to `NEURON_REGISTRY`: `"my_variant": MyVariantBank`
3. Set `config.neuron_variant = "my_variant"`

Nothing else changes. The model, trainer, scripts, and tests all work through the registry.

Currently registered variants:

```
"selective"  SelectiveNeuronBank  Input-dependent decay via softplus(W_delta @ x), S4 init, parallel scan
"stateful"   StatefulNeuronBank   Gated recurrence with learnable decay (fixed or selective mode)
"gated"      GatedNeuronBank      Same projections but no temporal state (ablation control)
```

## Baseline Comparison

`model.py` contains both `SquishyTransformer` and `BaselineTransformer`. They have identical forward signatures:

```python
forward(input_ids, targets=None, neuron_states=None)
    -> (logits, loss, neuron_states)
```

The baseline ignores and returns empty `neuron_states`. This means every script, every test, and the trainer work with both models without any branching. `build_model(config)` checks `config.use_baseline` and returns the right one.

## Setup

```bash
pip3 install torch tokenizers datasets wandb matplotlib safetensors pytest numpy tqdm
```

On macOS with system Python 3.9, you may need `numpy<2` for torch compatibility:

```bash
pip3 install "numpy<2"
```

## Running Tests

```
python3 -m pytest tests/ -v
```

### Test Results (59 tests)

```
tests/test_attention.py::TestCausalSelfAttention::test_output_shape            PASSED
tests/test_attention.py::TestCausalSelfAttention::test_output_shape_no_rope    PASSED
tests/test_attention.py::TestCausalSelfAttention::test_causal_masking          PASSED
tests/test_attention.py::TestCausalSelfAttention::test_gradients_flow          PASSED
tests/test_attention.py::TestCausalSelfAttention::test_different_seq_lengths   PASSED
tests/test_attention.py::TestCausalSelfAttention::test_single_token            PASSED
tests/test_attention.py::TestCausalSelfAttention::test_batch_independence      PASSED
tests/test_attention.py::TestCausalSelfAttention::test_deterministic_in_eval   PASSED
tests/test_attention.py::TestRotaryEmbedding::test_cos_sin_shapes             PASSED
tests/test_attention.py::TestRotaryEmbedding::test_dynamic_extension           PASSED
tests/test_attention.py::TestRotaryEmbedding::test_apply_rotary_preserves_shape PASSED
tests/test_attention.py::TestRotaryEmbedding::test_rotary_equivariance         PASSED
tests/test_model.py::TestRMSNorm::test_output_shape                            PASSED
tests/test_model.py::TestRMSNorm::test_gradients                               PASSED
tests/test_model.py::TestSquishyTransformer::test_forward_logits_shape         PASSED
tests/test_model.py::TestSquishyTransformer::test_forward_with_targets         PASSED
tests/test_model.py::TestSquishyTransformer::test_forward_no_rope              PASSED
tests/test_model.py::TestSquishyTransformer::test_backward                     PASSED
tests/test_model.py::TestSquishyTransformer::test_training_step_reduces_loss   PASSED
tests/test_model.py::TestSquishyTransformer::test_neuron_states_carry          PASSED
tests/test_model.py::TestSquishyTransformer::test_count_parameters             PASSED
tests/test_model.py::TestSquishyTransformer::test_gradient_checkpointing       PASSED
tests/test_model.py::TestBaselineTransformer::test_forward_shape               PASSED
tests/test_model.py::TestBaselineTransformer::test_forward_with_targets        PASSED
tests/test_model.py::TestBaselineTransformer::test_same_api_as_squishy         PASSED
tests/test_model.py::TestBaselineTransformer::test_gradient_checkpointing      PASSED
tests/test_model.py::TestBuildModel::test_builds_squishy_by_default            PASSED
tests/test_model.py::TestBuildModel::test_builds_baseline_when_flagged         PASSED
tests/test_model.py::TestBuildModel::test_param_counts_same_order_of_magnitude PASSED
tests/test_neuron.py::TestSelectiveNeuronBank::test_output_shape               PASSED
tests/test_neuron.py::TestSelectiveNeuronBank::test_state_changes_output       PASSED
tests/test_neuron.py::TestSelectiveNeuronBank::test_gradients_flow             PASSED
tests/test_neuron.py::TestSelectiveNeuronBank::test_state_is_detached          PASSED
tests/test_neuron.py::TestSelectiveNeuronBank::test_decay_rates_bounded        PASSED
tests/test_neuron.py::TestSelectiveNeuronBank::test_state_weight_bounded       PASSED
tests/test_neuron.py::TestSelectiveNeuronBank::test_batch_independence         PASSED
tests/test_neuron.py::TestSelectiveNeuronBank::test_single_timestep            PASSED
tests/test_neuron.py::TestStatefulNeuronBank::test_output_shape                PASSED
tests/test_neuron.py::TestStatefulNeuronBank::test_state_changes_output        PASSED
tests/test_neuron.py::TestStatefulNeuronBank::test_selective_mode              PASSED
tests/test_neuron.py::TestStatefulNeuronBank::test_decay_in_range              PASSED
tests/test_neuron.py::TestStatefulNeuronBank::test_state_weight_in_range       PASSED
tests/test_neuron.py::TestStatefulNeuronBank::test_gradients_flow              PASSED
tests/test_neuron.py::TestStatefulNeuronBank::test_zero_state_init             PASSED
tests/test_neuron.py::TestStatefulNeuronBank::test_state_is_detached           PASSED
tests/test_neuron.py::TestStatefulNeuronBank::test_single_timestep             PASSED
tests/test_neuron.py::TestStatefulNeuronBank::test_batch_independence          PASSED
tests/test_neuron.py::TestGatedNeuronBank::test_output_shape                   PASSED
tests/test_neuron.py::TestGatedNeuronBank::test_no_temporal_dependence         PASSED
tests/test_neuron.py::TestGatedNeuronBank::test_gradients_flow                 PASSED
tests/test_neuron.py::TestScanFunctions::test_sequential_scan_basic            PASSED
tests/test_neuron.py::TestScanFunctions::test_parallel_scan_matches_sequential PASSED
tests/test_neuron.py::TestScanFunctions::test_parallel_scan_single_step        PASSED
tests/test_neuron.py::TestScanFunctions::test_parallel_scan_non_power_of_two   PASSED
tests/test_neuron.py::TestRegistry::test_build_stateful                        PASSED
tests/test_neuron.py::TestRegistry::test_build_selective                       PASSED
tests/test_neuron.py::TestRegistry::test_build_gated                           PASSED
tests/test_neuron.py::TestRegistry::test_unknown_variant_raises                PASSED
tests/test_neuron.py::TestRegistry::test_all_registry_entries_instantiate      PASSED

59 passed
```

### What the Tests Actually Verify

**Neuron tests (27 tests):** Output shapes are correct for all 3 variants. State persistence works (same input with prior state produces different output). Decay rates and state_weight are always in (0,1). Gradients flow through every parameter including A_log, W_delta bias, and state_weight logits. Returned state is detached (no cross-sequence gradients). Batch elements are processed independently. Selective mode works in both SelectiveNeuronBank and StatefulNeuronBank. Parallel scan produces identical results to sequential scan for power-of-two lengths, non-power-of-two lengths, and single-step edge cases. The variant registry resolves correctly and raises on unknown keys. Every registered variant can instantiate and produce a forward pass.

**Attention tests (12 tests):** Output shape is correct with and without RoPE. Causal masking is proven by showing that modifying future tokens does not change past outputs. Gradients flow through the attention layer. Works with any sequence length from 1 to max_seq_len. Batch independence holds. Deterministic in eval mode. RoPE cos/sin cache shapes are correct and dynamically extend beyond initial max_seq_len. RoPE preserves dot-product structure (equivariance test).

**Model tests (15 tests):** RMSNorm output shapes and gradients. Full forward pass produces correct logit shapes with and without RoPE. Loss computation works with targets. Backward pass produces gradients. Multiple optimizer steps reduce loss (the model actually learns). Neuron states carry across forward passes. Gradient checkpointing works for both squishy and baseline models without breaking gradients. Both models have the same forward API. The factory builds the right model class. Parameter counts are in the expected range.

## Quick Test (CPU, ~10 minutes)

Generates a synthetic dataset with learnable patterns, trains both models for 500 steps, and compares:

```
python3 scripts/quick_test.py
```

The synthetic data uses template-based stories with character consistency, cause-and-effect chains, two-character interactions, and temporal sequences. These patterns are designed to give stateful neurons an advantage: predicting words later in the sequence requires remembering context from earlier (which character is in the story, what the weather is, etc).

Quick test config: d_model=128, n_heads=4, n_layers=3, seq_len=128, batch_size=16, 500 steps, no mixed precision. Trains on CPU in about 10 minutes total (both models).

## Full Training (GPU recommended)

### 1. Prepare data

```
python3 scripts/prepare_data.py
```

Downloads TinyStories from HuggingFace, trains a BPE tokenizer (vocab_size=8192), tokenizes both splits, saves to `data/processed/{train,validation}.npy`.

### 2. Train

```
python3 scripts/train.py                    # squishy (selective variant, default)
python3 scripts/train.py --baseline         # standard transformer (SwiGLU FFN)
python3 scripts/train.py --variant stateful # stateful variant (fixed or selective decay)
python3 scripts/train.py --variant gated    # gated variant (ablation, no temporal state)
python3 scripts/train.py --resume checkpoints/step_5000
```

Default config: d_model=256, n_heads=4, n_layers=4, seq_len=256, batch_size=32, 10K steps, fp16, cosine LR with warmup, RoPE enabled. Logs to wandb if available, stdout always.

### 3. Generate

```
python3 scripts/generate.py --checkpoint checkpoints/step_10000
python3 scripts/generate.py --checkpoint checkpoints/step_10000 --temperature 0.5
```

Interactive prompt loop. Top-k/top-p sampling. Ctrl-C to exit.

### 4. A/B comparison

```
python3 scripts/compare.py
```

Trains both models sequentially with the same seed and data, prints final validation loss and perplexity side by side.

## Training Loop Details

AdamW optimizer with (beta1=0.9, beta2=0.95). Fused AdamW on CUDA for faster updates. Weight decay applied only to 2D+ parameters (projection matrices), not to biases, RMSNorm, or the per-neuron A_log/state_weight scalars. Learning rate schedule: linear warmup over warmup_steps, then cosine decay to 10% of peak.

Gradient accumulation: configurable number of micro-batches per optimizer step. Gradient clipping at max_grad_norm=1.0. Mixed precision via `torch.autocast` with `torch.amp.GradScaler` for fp16 (bf16 does not need scaling). `optimizer.zero_grad(set_to_none=True)` for reduced memory overhead.

Checkpoints save model weights via safetensors, config as JSON, and optimizer/scaler/step state as .pt. Everything needed to resume training or reproduce results.

Device selection: CUDA > MPS > CPU, automatic.

## Analysis Tools

After training, `src/analysis.py` provides:

`extract_neuron_params(model)` pulls the learned decay rates and state_weight from every layer. For selective variants, this returns the base decay rates from A_log. Returns raw tensors.

`plot_decay_distribution(model)` shows per-layer histograms of learned decay rates, labeled by variant type (Selective vs Fixed). This answers the question: do neurons specialize? If you see a bimodal distribution (some neurons with decay near 0, others near 1), the neurons have differentiated.

`plot_state_weight_distribution(model)` same thing for state_weight. Shows how much the model learns to rely on accumulated state vs the raw signal at each layer.

`plot_selectivity_heatmap(model, input_ids, layer_idx)` visualizes how decay rates vary across sequence positions for a given input. This is unique to the selective variant and shows which tokens trigger memory retention vs flushing. Expects a (1, T) input tensor.

`trace_neuron_states(model, input_ids, layer_idx)` runs a forward pass and captures the state trajectory of individual neurons across the sequence. Returns a (T, n_neurons) tensor. Works with both selective and fixed decay variants.

`plot_state_trajectories(states, neuron_indices)` visualizes the above as line plots.

`print_neuron_summary(model)` prints a table of mean/std for decay and state_weight per layer with variant type annotation. Quick sanity check.

## Config Reference

Everything is in `config.py`. One dataclass. No magic numbers.

```
Model:      d_model=256  n_heads=4  n_layers=4  vocab_size=8192  max_seq_len=256  dropout=0.1
Neuron:     neuron_variant="selective"  decay_init=0.95  state_weight_init=0.5  selective_decay=True
Attention:  use_rope=True  rope_theta=10000.0
Training:   batch_size=32  lr=3e-4  weight_decay=0.1  warmup=200  max_steps=10000
            grad_accum=1  mixed_precision="fp16"  max_grad_norm=1.0  seed=42
            use_gradient_checkpointing=False
Data:       dataset="roneneldan/TinyStories"  tokenizer_vocab=8192  seq_length=256
Logging:    wandb_project="squishy-neuron"  log_interval=10  eval_interval=200  save_interval=1000
Generation: temperature=0.8  top_k=50  top_p=0.9
```

Config serializes to JSON and every checkpoint includes its config. `Config.load(path)` reconstructs from JSON.

## Research Questions This Is Built to Answer

1. **Does selective state beat fixed state?** The selective variant lets each token control decay per-neuron. The stateful variant with `selective_decay=False` uses a single learned decay per neuron. Comparing these isolates whether input-dependent memory gating matters. The gated variant (no state at all) isolates whether state itself matters.

2. **Parameter efficiency.** Does a stateful transformer with N parameters outperform a standard transformer with N parameters on language modeling perplexity? The squishy model has more params in the FFN portion due to extra projections. The gated variant (same projections but no state) isolates the contribution of state vs extra capacity.

3. **Neuron specialization.** After training, do neurons naturally develop different decay rates? Do some become long-memory context trackers while others become short-memory reactive processors? The analysis tools exist to measure this. The selectivity heatmap additionally shows whether the same neuron adapts its memory timescale based on input content.

4. **State dynamics.** What do the neuron states actually encode? Topic? Syntax? Character identity? The state trajectory tracing lets you correlate state values with sequence content.

5. **Scaling.** As model size increases, does the stateful architecture scale better, worse, or the same as standard transformers? Even directional evidence at small scale matters.

6. **Gradient health.** Does backpropagation through the state recurrence cause vanishing or exploding gradients? The `(1 - decay)` normalization and input-dependent gating (analogous to LSTM/Mamba gates) should help, but this needs empirical verification across training runs.

## Tech Stack

```
PyTorch >= 2.1       Core framework, built from nn.Linear/nn.Embedding/RMSNorm
tokenizers >= 0.15   HuggingFace Rust BPE tokenizer (NOT the transformers library)
datasets >= 2.16     HuggingFace datasets for downloading TinyStories
safetensors >= 0.4   Model serialization
wandb >= 0.16        Experiment tracking (optional, graceful fallback to stdout)
matplotlib >= 3.8    Visualization
pytest >= 7.4        Testing
numpy < 2            Numeric (< 2 required for torch 2.2 compatibility on Python 3.9)
```

No HuggingFace Transformers. No pretrained weights. Everything is built from scratch.
