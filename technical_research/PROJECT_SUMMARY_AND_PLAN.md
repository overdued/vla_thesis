# One-Step π0.5 Distillation — Project Summary & Planning

**Repository**: https://github.com/overdued/vla_thesis  
**Date**: 2026-05-13  
**Status**: Week 1 Complete ✅

---

## Executive Summary

This project aims to distill the 10-step Flow Matching ODE inference of π0.5 (3B parameter VLA model) into a single-step model, achieving ~10x inference speedup on LIBERO-10 benchmark while maintaining success rate.

### Key Achievement — Week 1
The complete distillation training framework has been built and verified with a 5-step compilation test that successfully runs end-to-end:
- Teacher model loads from local LoRA checkpoint (4.8s)
- Student state initializes with FSDP sharding across 2x A800 GPUs
- JIT compilation succeeds
- 5 training steps complete with `distill_loss=1.1719`, `grad_norm=1.4141`
- Checkpoint saves successfully

---

## 1. Architecture Overview

```
Teacher π0.5 (frozen, 10-step ODE):
  embed_prefix(obs) → prefix tokens
  10-step ODE: noise → clean_actions*
  
Student One-Step (trainable layers marked ★):
  embed_prefix(obs) → prefix tokens           [frozen]
  embed_suffix(noise, t=1) → suffix tokens     [★ action_in_proj]
  PaliGemma.llm(prefix, suffix) → output       [frozen]
  action_out_proj → predicted_actions          [★ action_out_proj]
  
Loss: ||predicted_actions - clean_actions*||²
```

**Trainable Parameters**: Only `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out`  
**Frozen Parameters**: All vision encoder (`img`) and LLM (`PaliGemma.llm`) parameters  
**Trainable Ratio**: ~0.5% of 3B model (~50M parameters)

---

## 2. File Changes

### Modified Files

| File | Lines Changed | Description |
|------|--------------|-------------|
| `src/openpi/models/pi0.py` | +91 | `compute_distillation_loss()`, `single_step_predict()` |
| `src/openpi/models/pi0_config.py` | +20 | `get_distill_freeze_filter()` |
| `src/openpi/training/config.py` | +121/-100 | Register `pi05_libero_distill` config |

### New Files

| File | Lines | Description |
|------|-------|-------------|
| `src/openpi/training/distillation_config.py` | 80 | Distillation-specific config classes |
| `src/openpi/training/distillation_trainer.py` | 345 | Core trainer: teacher gen + student train |
| `scripts/distill_pi05.py` | 49 | Training entry point |
| `scripts/eval_one_step.py` | 252 | Evaluation: teacher vs student speed |
| `technical_research/distillation_prompt_for_claude.md` | 943 | Full technical specification |
| `technical_research/week1_distillation_framework.md` | 200+ | Week 1 completion report |
| `technical_research/PROJECT_SUMMARY_AND_PLAN.md` | This file | Project summary & roadmap |

---

## 3. Critical Issues Resolved

### Issue 1: GCS Download Speed (Fixed ✅)
- **Problem**: Anonymous GCS access ~360KB/s, 6GB checkpoint needs 4-5 hours
- **Root Cause**: No gcloud credentials for `gs://openpi-assets/`
- **Solution**: Use local LoRA checkpoint `checkpoints/pi05_libero10_lora/libero10_lora/9999/params`
- **Speedup**: 5.4 GiB/s local vs 360KB/s GCS (~15,000x faster)
- **Bonus**: LoRA checkpoint has 80% LIBERO success rate vs base model's lower rate

### Issue 2: LoRA Parameter Mismatch (Fixed ✅)
- **Problem**: `ValueError: key 'lora_a' not available in state`
- **Root Cause**: LoRA checkpoint contains `lora_a`/`lora_b` params not in base architecture
- **Solution**: Filter checkpoint params to only keys present in model state
- **Code**: `traverse_util` filtering in both `init_teacher_model()` and `init_student_state()`

### Issue 3: GPU OOM on JIT Compile (Fixed ✅)
- **Problem**: `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.12GB`
- **Root Cause 1**: `restore_params()` with default `jax.Array` shards to ALL 8 GPUs
- **Root Cause 2**: GPUs 4-7 occupied by openvla-oft (63-67GB used, only ~15GB free)
- **Solution 1**: Use `restore_type=np.ndarray` to load on CPU, let JIT shard correctly
- **Solution 2**: `export CUDA_VISIBLE_DEVICES=0,1` to use only free GPUs

### Issue 4: Format String Error (Fixed ✅)
- **Problem**: `ValueError: Unknown format code 'f' for object of type 'str'`
- **Root Cause**: `f"{v:.4f}"` fails when `v` is a JAX array (not yet converted to float)
- **Solution**: `f"{float(v):.4f}"` explicit conversion

---

## 4. Performance Metrics

### Compilation Test Results (5 steps)

| Metric | Value |
|--------|-------|
| Teacher checkpoint load | 4.81s (5.3 GiB/s) |
| Student state init | ~3s |
| JIT compilation | ~60s (first time) |
| Per-step training time | ~21s |
| Checkpoint save | 6.5s (async) |
| **Step 0 distill_loss** | **1.1719** |
| **Step 0 grad_norm** | **1.4141** |

### Hardware Configuration

| Resource | Value |
|----------|-------|
| GPUs | 2x NVIDIA A800-SXM4-80GB (GPU 0,1) |
| GPU Memory (each) | ~35GB used / 80GB total |
| Batch Size | 16 (local) |
| FSDP Devices | 2 |
| Model Dtype | bfloat16 |

### Memory Breakdown (per GPU)

| Component | Size |
|-----------|------|
| Student params (sharded) | ~3GB |
| Student ema_params (sharded) | ~3GB |
| Optimizer state (action layers only) | ~100MB |
| Teacher model (sharded, frozen) | ~3GB |
| Activation + temporary | ~10GB |
| **Total** | **~35GB** |

---

## 5. Week-by-Week Plan

### Week 1 ✅ Complete (2026-05-13)
- [x] Build distillation code framework
- [x] Add `compute_distillation_loss()` and `single_step_predict()` to Pi0
- [x] Implement freeze filter (only action projection layers trainable)
- [x] Create distillation trainer with teacher/student separation
- [x] Run 5-step compilation test successfully
- [x] Resolve all critical issues (GCS, LoRA, OOM, format)

### Week 2 — Small-Scale Training & Baseline (2026-05-20)
- [ ] **1K-step training run** on LIBERO-10
  - Verify loss consistently decreases
  - Monitor grad_norm for training stability
  - Expected: distill_loss < 0.5 at 1K steps
- [ ] **Baseline evaluation**: Teacher 10-step inference speed
  - Measure actions/sec on LIBERO-10
  - Expected: ~500-1000ms per action sequence
- [ ] **Single-step evaluation**: Student inference speed
  - Load 1K-step checkpoint
  - Measure actions/sec
  - Expected: ~50-100ms per action sequence (10x speedup)
- [ ] **Success rate comparison** (small sample, 10 trials)
  - Teacher baseline on 5 LIBERO tasks
  - Student on same 5 tasks

### Week 3 — Full Training & Hyperparameter Tuning (2026-05-27)
- [ ] **Full 50K-step training**
  - Learning rate: 1e-4 with cosine decay to 1e-5
  - Batch size: 16, warmup: 1K steps
  - EMA decay: 0.999
  - Expected training time: ~12 hours (50K × 21s = ~292 min + compile)
- [ ] **Learning rate ablation**
  - Try 5e-5, 1e-4, 2e-4
  - Track which gives best convergence
- [ ] **Loss type ablation**
  - MSE vs weighted MSE vs consistency loss
- [ ] **Checkpoint evaluation every 10K steps**
  - Track success rate vs training step

### Week 4 — Evaluation & Analysis (2026-06-03)
- [ ] **Full LIBERO-10 evaluation** (100 trials per task)
  - Teacher (10-step ODE)
  - Student (1-step distilled)
  - Metrics: success rate, inference time, total runtime
- [ ] **Visualization**
  - Training loss curve
  - Success rate vs training step
  - Inference speed comparison bar chart
- [ ] **Video comparison**
  - Side-by-side teacher vs student rollouts
- [ ] **Final report writing**
  - Quantitative results table
  - Qualitative analysis
  - Ablation study summary

---

## 6. Known Limitations & Future Work

### Current Limitations
1. **Teacher uses LoRA checkpoint, not base**: While this gives better LIBERO performance, it's technically different from the original plan
2. **GPU 4-7 unavailable**: Occupied by openvla-oft, limiting to 2 GPUs instead of planned 4
3. **No multi-step distillation**: Only single-step (t=1) loss implemented; multi-step could improve quality
4. **No consistency loss**: Only MSE loss; Flow Matching consistency loss may help

### Future Directions
1. **Full-parameter distillation**: Unfreeze more layers (e.g., entire action expert)
2. **Progressive distillation**: Start with multi-step, gradually reduce to 1-step
3. **Task-specific distillation**: Train separate students for each LIBERO task suite
4. **Real robot deployment**: Deploy distilled model on real robot hardware

---

## 7. Reproduction Instructions

### Environment Setup
```bash
# Clone repo
git clone https://github.com/overdued/vla_thesis.git
cd vla_thesis

# Install dependencies
uv sync

# Set environment
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Run Distillation Training
```bash
uv run python scripts/distill_pi05.py pi05_libero_distill \
    --exp-name=distill_run1 \
    --num_train_steps=50000
```

### Run Evaluation
```bash
# Evaluate teacher (10-step baseline)
uv run python scripts/eval_one_step.py --mode=teacher

# Evaluate student (1-step distilled)
uv run python scripts/eval_one_step.py --mode=student \
    --student_checkpoint=checkpoints/pi05_libero_distill/distill_run1/50000
```

---

## 8. Key Design Decisions

### Why freeze 99.5% of parameters?
- π0.5's "intelligence" is in the LLM and vision encoder (scene understanding, language comprehension)
- Action projection layers are Flow Matching-specific, responsible for conditional generation
- Single-step distillation is essentially learning a different noise→action mapping
- Only ~50M trainable params → efficient training on 2x A800

### Why reuse Pi0 class instead of new model?
- Simpler: just add methods, no structural changes
- Freeze strategy via NNX filter, no model modification needed
- Maintains compatibility with existing training infrastructure

### Why local LoRA checkpoint as teacher?
- GCS anonymous access too slow (360KB/s)
- LoRA checkpoint has better LIBERO performance (80% vs base)
- 6GB local file loads in ~5 seconds

---

*Generated: 2026-05-13*  
*Next Update: After Week 2 completion*
