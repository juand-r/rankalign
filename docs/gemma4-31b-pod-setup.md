# Setting up a RunPod pod for `google/gemma-4-31B-it`

Hard-won from a multi-hour debugging saga (2026-05-18). Read this *before*
provisioning a pod for any gemma-4-31b work. The single most important fact:

> **gemma-4-31b needs PyTorch ≥ 2.5.** The common
> `runpod/pytorch:2.4.0-...-cuda12.4.1` image ships **torch 2.4.1 — too old**.
> Using it causes three *different* fatal failures (below). Either start from
> a pod that already has torch ≥2.5, or provision on a torch ≥2.5 base image.
> Do **not** try to pip-upgrade torch onto a 2.4 image (cuDNN hell).

## 1. Why torch ≥ 2.5 is non-negotiable

Current `transformers` gemma-4 support fails on torch 2.4.1 in three ways,
each looking unrelated until you see the pattern:

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'torch.distributed.tensor.device_mesh'` (on `from transformers import AutoModelForCausalLM`) | transformers ≥5.x imports a torch ≥2.5 API |
| `ValueError: infer_schema(func): Parameter input has unsupported type torch.Tensor ... (input, weight, offs)` | gemma-4's grouped-GEMM custom op needs torch ≥2.5's `torch.library.infer_schema` |
| `ImportError: libcudnn.so.9: cannot open shared object file` (after pip-upgrading torch to 2.6 on the 2.4 image) | the torch-2.4 image ships cuDNN 8; do not chase this — use a correct base image instead |

The pod that actually works (`h100-train-4` / `d4gum0mttfqkre`, used for the
gemma-4 training+eval) has **torch 2.5.1+cu124**.

## 2. Base image

- **Reuse an existing working pod if possible** (its volume keeps torch 2.5.1
  + the model cache). Note: a *stopped* RunPod pod may refuse to restart —
  `"not enough free GPUs on the host"` — because RunPod reclaims GPUs while
  stopped. The volume survives but the pod can be unstartable; then you must
  reprovision.
- **Fresh pod:** pick a base image that already has torch ≥2.5, e.g.
  `runpod/pytorch:1.0.3-...-cu1290-torch280-ubuntu2404` (torch 2.8, CUDA 12.9)
  or any `...-torch26x/27x/28x/29x-...` tag. Check current tags first:
  `https://hub.docker.com/v2/repositories/runpod/pytorch/tags` — do **not**
  assume a tag exists.
- GPU: gemma-4-31b is ~62 GB in bf16. One 80 GB GPU loads it (tight with
  activations); 2× (A100-SXM-80GB or H100-NVL/SXM) gives headroom for the
  scorer (which can load a second model).
- Volume 100 GB at `/workspace` (model cache ~59 GB lives here). Container
  disk is small (~50 GB) — never let HF cache default there.

## 3. Step-by-step

```bash
# --- env (do BEFORE any HF download) ---
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1            # xet path errors mid-download on big models
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=<gated-access token>  # gemma-4 is GATED; without it
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN # snapshot_download silently 0-exits

# --- code ---
git clone -b longform https://github.com/juand-r/rankalign.git /workspace/rankalign

# --- venv: --system-site-packages so the image's torch/CUDA is used ---
python3 -m venv /workspace/.venv --system-site-packages
source /workspace/.venv/bin/activate
python -c "import torch; assert torch.__version__ >= '2.5', torch.__version__; print(torch.__version__)"

# --- PINNED deps (NEVER git+main — see gotchas) ---
pip install -r /workspace/rankalign/requirements-gemma4.txt
# committed authoritative pins ^ (transformers==5.8.1, tokenizers==0.22.2,
# huggingface_hub==1.15.0, accelerate==1.1.0, peft==0.14.0, etc.)
# Full 312-pkg lockfile (older, still useful as reference):
#   scripts-more/correct_multi/canary_env.freeze.txt

# --- verify the ACTUAL modules import (not just libs) ---
python - <<'PY'
import torch, transformers, libcst, pandas, accelerate          # libs
from transformers import AutoModelForCausalLM, AutoTokenizer     # the failing path
import sys; sys.path[:0]=["/workspace/rankalign/scripts-more/correct_multi",
                           "/workspace/rankalign/scripts"]
import transforms, build_canary, build_v2_1_correct_multi        # our code
from dataset_builder.build_humaneval_v2_1_correct_upper import validate
assert torch.cuda.is_available()
print("ALL OK", torch.__version__, transformers.__version__)
PY

# --- pre-download the model (retry; verify size, not exit code) ---
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('google/gemma-4-31B-it', cache_dir='$HF_HUB_CACHE')"
du -sh $HF_HUB_CACHE/models--google--gemma-4-31B-it   # expect ~59 GB
```

Or just run the committed one-shot (idempotent, fail-loud, installs pinned
reqs, pre-downloads the model):

```bash
bash /workspace/rankalign/setup-runpod-gemma4.sh
```

## 4. Versioned package list (the ones that matter)

Source of truth (committed, prefer these if this doc drifts):
`requirements-gemma4.txt` in the repo root (canonical pins for gemma-4 pods).
Legacy reference: `scripts-more/correct_multi/requirements-canary.txt` and
`canary_env.freeze.txt` (full 312-pkg lockfile, older but still useful).

| package | version | note |
|---|---|---|
| **torch** | **2.5.1+cu124** | from the pod image, NOT pip-pinned. **≥2.5 mandatory.** |
| **transformers** | `git+...@24f73bc27f066e01036b81c3dff84d1a3ddb4f71` | **pin the COMMIT**, never `main` (moving target — main advanced overnight and broke). `transformers==5.8.1` release also works on torch ≥2.5. |
| **accelerate** | 1.1.0 | required by `device_map="auto"` |
| **libcst** | 1.8.6 | needed by the correct-multi transforms; a gemma-4 *training* venv will NOT have it — install explicitly |
| tokenizers | 0.22.2 | |
| huggingface_hub | 1.14.0 | |
| hf_transfer | 0.1.9 | fast gated download |
| safetensors | 0.8.0rc0 | (an RC — pinned because it works) |
| pandas | 3.0.3 | |
| numpy | 2.1.3 | |
| scipy | 1.14.1 | |
| scikit-learn | 1.5.2 | |
| datasets | 3.1.0 | |
| sentencepiece | 0.2.1 | gemma tokenizer |
| protobuf | 7.34.1 | |
| pyarrow | 24.0.0 | |

(torchvision 0.20.1 / torchaudio 2.4.1+cu124 are present but unused for
inference — the torchaudio cu124 tag mismatch with torch 2.5.1 is harmless.)

## 5. Gotchas checklist

- [ ] **torch ≥2.5** — the whole saga. Verify `torch.__version__` first.
- [ ] **Never `pip install git+https://github.com/huggingface/transformers.git`** — `main` is a moving target; it worked 2026-05-09, broke by 2026-05-18. Pin the commit (or a tested release).
- [ ] **Don't pip-upgrade torch onto a torch-2.4 image** — cuDNN 9 / nvidia-lib hell with `--system-site-packages`. Use a correct base image.
- [ ] **`libcst` is missing from training venvs** — `pip install libcst==1.8.6`.
- [ ] **`accelerate` required** for `device_map="auto"` — easy to omit.
- [ ] **HF cache → `/workspace`** (set `HF_HUB_CACHE` *before* downloading) — container disk is ~50 GB, the model is ~59 GB.
- [ ] **`HF_HUB_DISABLE_XET=1`** — xet path errors mid-download on large models.
- [ ] **`HF_TOKEN` required** — gemma-4 is gated; `snapshot_download` exits 0 with an incomplete download if unauthenticated. Verify size (~59 GB), not exit code.
- [ ] **Stopped pods may not restart** (`not enough free GPUs on the host`) — volume survives, pod may not. Be ready to reprovision on a torch≥2.5 image.
- [ ] **Verify by importing the ACTUAL scripts**, not just libraries — an incomplete install passes `import torch` but breaks `transforms`/`ranking_loss_ref`.
- [ ] **`venv --system-site-packages`** so the image's torch/CUDA is used (don't reinstall torch).
- [ ] **`ssh pod 'pgrep -f PATTERN'` self-matches** — use file markers / GPU util / `ps` (not the ssh-arg-matching pgrep) to check job liveness.
- [ ] **pip on some pods is very slow** — `requirements-gemma4.txt` is lean (no torch, no trl); install should take a few minutes, not 50.

## 6. Verification it actually works

```bash
python -c "
import torch; from transformers import AutoModelForCausalLM, AutoTokenizer
m='google/gemma-4-31B-it'
AutoTokenizer.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16, device_map='auto')
print('gemma-4-31b LOADS OK on torch', torch.__version__)"
```

If that prints OK, the pod is correctly set up.
