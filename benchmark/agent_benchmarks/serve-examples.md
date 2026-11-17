# vLLM serve examples

These examples adapt the deployed `serve_*.sh` scripts to the generic
[`serve_vllm_model.sh`](serve_vllm_model.sh) wrapper. They preserve the model
paths, ports, GPU assignments, and vLLM flags used in the actual deployment.

The first accuracy pass covers the Qwen3.8 Flash Next family. The remaining
Qwen3.8-27B, Qwen3.6, DeepSeek, and GLM examples are included for deployment
completeness; their accuracy can be added in a follow-up.

Environment-specific setup is a prerequisite and is **not** performed by the
generic wrapper:

- Put the intended vLLM executable on `PATH`, pass `--vllm`, or set `VLLM_BIN`.
- For HC-MXFP8, the patched vLLM overlay must already be installed; export
  `VLLM_OVERLAY`, `PYTHONPATH`, and `CUDA_HOME` before calling the wrapper.
- This host needs `NCCL_NVLS_ENABLE=0` and usually
  `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- `--language-model-only` is used for text benchmarks to skip the vision tower.

## Measured accuracy (7 completed runs)

Source: TaskWatch Summary, 7 completed runs per benchmark. The value shown is
the mean over the 7 runs; the per-example tables also include the trimmed mean
(drop the two highest and two lowest scores) and standard deviation. `—` means
no completed 7-run suite was available for that benchmark.

| Serve example | Served name | Port | GPU | SWE Verified Mini | SWE-bench Pro (642) | HumanEval+ | AIME-2026 | MATH-500 | GPQA Diamond |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HC-MXFP8 | `Qwen3.8-Flash-Next-HC-MXFP8` | 8008 | 6 | — | — | 89.90% | 98.57% | 98.91% | 92.42% |
| HC-MXFP8-MTP-BF16 | `Qwen3.8-Flash-Next-HC-MXFP8-MTP-BF16` | 8009 | 0 | — | — | — | — | — | — |
| MXFP4-Mixed (`qwen38-mxfp4`) | `qwen38-mxfp4` | 8008 | 6 | 72.29% | 81.98% | 89.20% | 99.52% | 99.03% | 92.13% |
| MXFP4-Mixed + BF16 SSM | `qwen38-mxfp4-bf16ssm` | 8005 | 6 | 72.57% | 81.49% | 89.20% | 96.67% | 98.83% | 91.70% |
| NVFP4 | `qwen38-nvfp4` | 8009 | 5 | 70.86% | 77.27% | 89.90% | 98.57% | 98.86% | 91.48% |
| NVFP4-QAD | `qwen38-nvfp4-qad` | 8011 | 6 | 71.71% | 76.03% | 89.98% | 99.52% | 99.00% | 91.27% |

Notes:

- `qwen38-mxfp4` and `qwen38-mxfp4-bf16ssm` use the same checkpoint. The
  bf16ssm launch adds `--mamba-ssm-cache-dtype bfloat16`; the historical
  `qwen38-mxfp4` run did not.
- `qwen38-nvfp4` and `qwen38-nvfp4-qad` both use the
  `serve_Qwen3.8-Flash-Next-MXFP4-Mixed-1005.sh` runner with different
  `CKPT`, `PORT`, `CUDA_VISIBLE_DEVICES`, and `SERVED_MODEL_NAME` values.
- SWE-bench Pro here is the full 642-instance ScaleAI dataset.
- The 27B series (`serve_qwen3.8_27B*.sh`) and the non-Qwen3.8 models are not
  included in this first table.

## 1. HC-MXFP8

Source script: `serve_Qwen3.8-Flash-Next-HC-MXFP8.sh`

Prerequisites: the vLLM HC-MXFP8 overlay/patch must be installed, and the
FlashInfer CUDA toolkit must point at the venv's CUDA 13 tree.

```bash
export VLLM_OVERLAY=/home/kaokaolv/vllm_overlay_hc_mxfp8
export PYTHONPATH="$VLLM_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=/home/kaokaolv/test-swe/venv_qwen38fn/lib/python3.12/site-packages/nvidia/cu13

bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/aqa_results/qwen38_flash_next_hc_mxfp8/quantized_hc_fp8 \
  --served-name Qwen3.8-Flash-Next-HC-MXFP8 \
  --port 8008 --gpus 6 --tp 1 \
  --max-model-len 262144 --gpu-mem-util 0.90 \
  --max-num-seqs 16 --max-num-batched-tokens 65536 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --safetensors-load-strategy prefetch \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --env NCCL_NVLS_ENABLE=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/test-swe/venv_qwen38fn/bin/vllm
```

Accuracy:

| Benchmark | Mean | Trimmed mean | Std | n |
| --- | ---: | ---: | ---: | ---: |
| SWE Verified Mini | — | — | — | — |
| SWE-bench Pro (642) | — | — | — | — |
| HumanEval+ | 89.90% | 90.04% | 0.99% | 7 |
| AIME-2026 | 98.57% | 98.89% | 1.78% | 7 |
| MATH-500 | 98.91% | 99.00% | 0.40% | 7 |
| GPQA Diamond | 92.42% | 92.25% | 0.83% | 7 |

## 2. HC-MXFP8-MTP-BF16

Source script: `serve_Qwen3.8-Flash-Next-HC-MXFP8-MTP-BF16.sh`

This variant uses a BF16 MTP draft and the same HC-MXFP8 overlay. MTP is
optional; enable it only when the draft patch is installed.

```bash
export VLLM_OVERLAY=/home/kaokaolv/vllm_overlay_hc_mxfp8
export PYTHONPATH="$VLLM_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=/home/kaokaolv/test-swe/venv_qwen38fn/lib/python3.12/site-packages/nvidia/cu13

bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/aqa_results/qwen38_flash_next_hc_mxfp8_mtp_bf16/quantized_hc_fp8_mtp_bf16 \
  --served-name Qwen3.8-Flash-Next-HC-MXFP8-MTP-BF16 \
  --port 8009 --gpus 0 --tp 1 \
  --max-model-len 262144 --gpu-mem-util 0.90 \
  --max-num-seqs 16 --max-num-batched-tokens 65536 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --safetensors-load-strategy prefetch \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --env NCCL_NVLS_ENABLE=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/test-swe/venv_qwen38fn/bin/vllm
```

Accuracy: no completed 7-run suite was available at the time of writing.

## 3. qwen38-mxfp4 (MXFP4-Mixed, no BF16 SSM flag)

Source configuration: same checkpoint as the bf16ssm script, but without
`--mamba-ssm-cache-dtype bfloat16` (the historical `qwen38-mxfp4` launch).

```bash
bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/models/Qwen3.8-Flash-Next-MXFP4-Mixed-BF16Attn-MTP8-1008-PLEMXFP8 \
  --served-name qwen38-mxfp4 \
  --port 8008 --gpus 6 --tp 1 \
  --max-model-len 262144 --gpu-mem-util 0.92 \
  --max-num-seqs 16 --max-num-batched-tokens 65536 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --safetensors-load-strategy prefetch \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --env NCCL_NVLS_ENABLE=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/test-swe/venv_qwen38fn/bin/vllm
```

Accuracy:

| Benchmark | Mean | Trimmed mean | Std | n |
| --- | ---: | ---: | ---: | ---: |
| SWE Verified Mini | 72.29% | 72.00% | 2.43% | 7 |
| SWE-bench Pro (642) | 81.98% | 82.03% | 0.39% | 7 |
| HumanEval+ | 89.20% | 89.23% | 0.91% | 7 |
| AIME-2026 | 99.52% | 100.00% | 1.26% | 7 |
| MATH-500 | 99.03% | 99.00% | 0.35% | 7 |
| GPQA Diamond | 92.13% | 91.92% | 1.05% | 7 |

## 4. qwen38-mxfp4-bf16ssm

Source script: `serve_Qwen3.8-Flash-Next-MXFP4-Mixed-bf16ssm.sh`

Same checkpoint as the example above, with the BF16 SSM cache dtype. The
generic wrapper passes `--mamba-ssm-cache-dtype` through the `--` separator.

```bash
bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/models/Qwen3.8-Flash-Next-MXFP4-Mixed-BF16Attn-MTP8-1008-PLEMXFP8 \
  --served-name qwen38-mxfp4-bf16ssm \
  --port 8005 --gpus 6 --tp 1 \
  --max-model-len 262144 --gpu-mem-util 0.92 \
  --max-num-seqs 16 --max-num-batched-tokens 65536 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --safetensors-load-strategy prefetch \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --env NCCL_NVLS_ENABLE=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/test-swe/venv_qwen38fn/bin/vllm \
  -- --mamba-ssm-cache-dtype bfloat16
```

Accuracy:

| Benchmark | Mean | Trimmed mean | Std | n |
| --- | ---: | ---: | ---: | ---: |
| SWE Verified Mini | 72.57% | 72.67% | 3.21% | 7 |
| SWE-bench Pro (642) | 81.49% | 81.46% | 0.44% | 7 |
| HumanEval+ | 89.20% | 89.43% | 0.84% | 7 |
| AIME-2026 | 96.67% | 96.67% | 1.93% | 7 |
| MATH-500 | 98.83% | 98.87% | 0.37% | 7 |
| GPQA Diamond | 91.70% | 91.75% | 1.12% | 7 |

## 5. qwen38-nvfp4

Source script: `serve_Qwen3.8-Flash-Next-MXFP4-Mixed-1005.sh` with the default
NVFP4 checkpoint. The deployed service used port 8009 and GPU 5.

```bash
bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/models/Qwen3.8-Flash-Next-NVFP4 \
  --served-name qwen38-nvfp4 \
  --port 8009 --gpus 5 --tp 1 \
  --max-model-len 262144 --gpu-mem-util 0.92 \
  --max-num-seqs 16 --max-num-batched-tokens 65536 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --safetensors-load-strategy prefetch \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --env NCCL_NVLS_ENABLE=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/test-swe/venv_qwen38fn/bin/vllm
```

Accuracy:

| Benchmark | Mean | Trimmed mean | Std | n |
| --- | ---: | ---: | ---: | ---: |
| SWE Verified Mini | 70.86% | 71.33% | 3.24% | 7 |
| SWE-bench Pro (642) | 77.27% | 77.45% | 1.26% | 7 |
| HumanEval+ | 89.90% | 90.04% | 0.69% | 7 |
| AIME-2026 | 98.57% | 100.00% | 2.62% | 7 |
| MATH-500 | 98.86% | 98.80% | 0.19% | 7 |
| GPQA Diamond | 91.48% | 91.24% | 1.14% | 7 |

## 6. qwen38-nvfp4-qad

Source script: `serve_Qwen3.8-Flash-Next-MXFP4-Mixed-1005.sh` with the QAD
checkpoint. The original `...-FP4-Local` launch failed engine initialization;
the benchmarked service used the `...-FP4-Local-F8Scale` checkpoint under the
same served name. The deployed service used port 8011 and GPU 6.

```bash
bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/models/Qwen3.8-Flash-Next-FP4-Local-F8Scale \
  --served-name qwen38-nvfp4-qad \
  --port 8011 --gpus 6 --tp 1 \
  --max-model-len 262144 --gpu-mem-util 0.92 \
  --max-num-seqs 16 --max-num-batched-tokens 65536 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --safetensors-load-strategy prefetch \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --env NCCL_NVLS_ENABLE=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/test-swe/venv_qwen38fn/bin/vllm
```

Accuracy:

| Benchmark | Mean | Trimmed mean | Std | n |
| --- | ---: | ---: | ---: | ---: |
| SWE Verified Mini | 71.71% | 71.33% | 1.80% | 7 |
| SWE-bench Pro (642) | 76.03% | 75.96% | 0.67% | 7 |
| HumanEval+ | 89.98% | 89.84% | 0.69% | 7 |
| AIME-2026 | 99.52% | 100.00% | 1.26% | 7 |
| MATH-500 | 99.00% | 99.00% | 0.12% | 7 |
| GPQA Diamond | 91.27% | 90.91% | 1.23% | 7 |

## 7. Qwen3.8-27B-MXFP4

Source script: `serve_qwen3.8_27B.sh`

```bash
bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/my_commit/Qwen3.8-27B-MXFP8Attn-MXFP4MLP-lmheadMXFP4/Qwen3.8-27B-mxfp-w8g32 \
  --served-name Qwen3.8-27B-mxfp4 \
  --port 8227 --gpus 7 --tp 1 \
  --max-model-len 262144 --gpu-mem-util 0.90 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/test-swe/venv_qwen38fn/bin/vllm
```

## 8. Qwen3.8-27B-MXFP4 with MTP

Source script: `serve_qwen3.8_27B_mtp.sh`

Uses the same checkpoint and served name as example 7; do not run both at the
same time.

```bash
bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/my_commit/Qwen3.8-27B-MXFP8Attn-MXFP4MLP-lmheadMXFP4/Qwen3.8-27B-mxfp-w8g32 \
  --served-name Qwen3.8-27B-mxfp4 \
  --port 8200 --gpus 5 --tp 1 \
  --max-model-len 262144 --gpu-mem-util 0.90 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/test-swe/venv_qwen38fn/bin/vllm
```

## 9. Qwen3.8-27B-MXFP8-MTP

Source script: `serve_qwen3.8_27B_mxfp8_mtp.sh`

```bash
bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/models/Qwen3.8-27B-MXFP8-CT-AutoRound \
  --served-name Qwen3.8-27B-mxfp8-mtp \
  --port 8205 --gpus 6 --tp 1 \
  --max-model-len 262144 --gpu-mem-util 0.90 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/test-swe/venv_qwen38fn/bin/vllm
```

## 10. Qwen3.8-27B-NVFP4

Source script: `serve_qwen3.8_27B_nvfp4.sh`

```bash
bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/my_commit/Qwen3.8-27B-NVFP4 \
  --served-name Qwen3.8-27B-nvfp4 \
  --port 8228 --gpus 4 --tp 1 \
  --max-model-len 262144 --gpu-mem-util 0.90 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/.venv-vllm-nightly/bin/vllm
```

## 11. Qwen3.8-27B-Coder

Source script: `serve_qwen3.8_27B_coder.sh`

The source script points `CUDA_HOME` at the venv CUDA 13 toolkit for
FlashInfer JIT; export it before calling the generic wrapper if needed.

```bash
export CUDA_HOME=/home/kaokaolv/test-swe/venv_qwen38fn/lib/python3.12/site-packages/nvidia/cu13

bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/models/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2/BF16 \
  --served-name Qwen3.8-27B-coder \
  --port 8229 --gpus 7 --tp 1 \
  --max-model-len 262144 --gpu-mem-util 0.90 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/test-swe/venv_qwen38fn/bin/vllm
```

## 12. Qwen3.6-35B-A3B

Source script: `serve_Qwen3.6-35B-A3B.sh`

```bash
bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/models/Qwen3.6-35B-A3B \
  --served-name qwen3.6-35b-a3b \
  --port 8232 --gpus 7 --tp 1 \
  --max-model-len 262144 --max-num-seqs 256 --gpu-mem-util 0.90 \
  --dtype bfloat16 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --env NCCL_NVLS_ENABLE=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/.venv-vllm-main/bin/vllm
```

## 13. Qwen3.6-35B-A3B-pt

Source script: `serve_Qwen3.6-35B-A3B_pt.sh`

```bash
bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/models/Qwen3.6-35B-A3B-sft-8k-hf-merged-iter1000 \
  --served-name qwen3.6-35b-a3b-pt \
  --port 8233 --gpus 7 --tp 1 \
  --max-model-len 262144 --max-num-seqs 256 --gpu-mem-util 0.90 \
  --dtype bfloat16 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --env NCCL_NVLS_ENABLE=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/.venv-vllm-main/bin/vllm
```

## 14. DeepSeek-V4-Flash-0731-MXFP4

Source script: `serve_DeepSeek-V4-Flash-0731-MXFP4.sh`

The source script checks a `wo_a` patch before starting; that prerequisite is
not handled by the generic wrapper.

```bash
bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/dsv4_0731/runs/quant_main/quantized \
  --served-name dsv4flash0731-mxfp4 \
  --port 8230 --gpus 0,1 --tp 2 \
  --max-model-len 8192 --max-num-seqs 256 --gpu-mem-util 0.90 \
  --dtype bfloat16 --block-size 256 --kv-cache-dtype fp8 \
  --enforce-eager --no-enable-prefix-caching \
  --env NCCL_NVLS_ENABLE=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/.venv-vllm-main/bin/vllm
```

## 15. GLM-5.3-Flash-MXFP4

Source script: `serve_GLM-5.3-Flash-MXFP4.sh`

The source script checks a checkpoint namespace patch before starting; that
prerequisite is not handled by the generic wrapper.

```bash
bash benchmark/agent_benchmarks/serve_vllm_model.sh \
  --model /home/kaokaolv/glm53/runs/quant_main/quantized \
  --served-name glm53-mxfp4 \
  --port 8231 --gpus 0,1 --tp 2 \
  --max-model-len 8192 --max-num-seqs 256 --gpu-mem-util 0.90 \
  --dtype bfloat16 --kv-cache-dtype fp8 \
  --enforce-eager --no-enable-prefix-caching \
  --env NCCL_NVLS_ENABLE=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --vllm /home/kaokaolv/.venv-vllm-main/bin/vllm
```

## Data provenance

The accuracy values were read from TaskWatch's Summary view on 2026-11-17.
Each value covers 7 completed runs. TaskWatch computes the trimmed mean by
dropping the two highest and two lowest scores when at least seven runs are
available; the standard deviation is the sample standard deviation across the
seven runs.

The serve commands were reconstructed from the deployed `serve_*.sh` scripts
and their launch logs (`serve_hc_mxfp8_gpu6.log`, `serve_mxfp4_gpu6.log`,
`serve_mxfp4_bf16ssm_gpu6.log`, `serve_nvfp4_gpu5.log`,
`serve_nvfp4_qad_gpu6.log`, and `serve_nvfp4_qad_gpu6_f8scale.log`). The
Qwen3.8-27B, Qwen3.6, DeepSeek, and GLM commands were taken from their
corresponding `serve_*.sh` scripts.
