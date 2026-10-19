# PCMN — Parallel Causal Multimodal Network

Experimental architecture exploring parallel recurrent sequence processing, addressable and external retrieval memory, local CPU/CUDA backends and eventual multimodal models. **Research prototype only**: not a production LLM; measured reasoning/coding/tool-use quality is not comparable to a modern trained LLM.

The repository currently contains the **v0.6 text-core sources, tests, scripts, native C++/CUDA prototypes and public experiment reports**. Large trained weights and generated binary datasets are intentionally excluded from the Git history.

## Install (Windows / Linux)

Use Python **3.12** for the most conservative compatibility with TorchScript tracing; Python 3.14 emits warnings about `torch.jit.trace`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.hardware_preflight_v06 --out benchmarks/hardware_local.json
```

`requirements.txt` specifies baseline CPU dependencies. For CUDA, **install a CUDA-enabled PyTorch build appropriate to your NVIDIA GPU separately** and check `torch.cuda.is_available()` before any training. The Tesla P100 uses compute capability **6.0 (`sm_60`)** and does not support BF16. CUDA toolkit `nvcc` being present does not mean the installed PyTorch build supports CUDA.

## Reproducing the user's Windows tests

See [Windows log and interpretation](docs/USER_WINDOWS_VALIDATION_RU.md). Detailed commands: [README_RU.md](README_RU.md).

User-provided Windows 11 / PyTorch CPU results (October 2026): 53 tests run, **52 passed and 1 skipped** (native library not built); trained mini model 2,273,480 parameters; exhaustive validation NLL **3.1340 → 2.5613**; exact-match cognitive tasks **0/8**; CPU decode eager **135.0 tokens/s** vs traced **220.47 tokens/s** with 2 threads in that run. Do not extrapolate these numbers to larger models or CUDA.

## Project layout

- `pcmn/`: model architecture, datasets, probabilistic/causal modules, memory, retrieval, serving and tool interfaces
- `scripts/`: training, evaluation, local serving and microbenchmarks
- `native/`: experimental C++/CUDA kernels
- `tests/`: unit tests and numerical equivalence checks
- `data/`: tiny demonstration texts and tasks (not a pretraining corpus)
- `benchmarks/`, `reports/`: historical evaluation artifacts and audit trails

## Roadmap

1. Restore native-library test on Windows with CMake/MSVC and run it explicitly.
2. Replace the CPU-only Torch installation with a verified CUDA build for Tesla P100, then build and test native CUDA kernels.
3. Move away from deprecated `torch.jit.trace` for Python 3.14 compatibility after baseline equivalence tests.
4. Construct licensed, deduplicated large BPE corpus with held-out sources and longer-term performance tests.
5. Conduct meaningful pretraining before making claims about coding, reasoning or autonomous agent abilities.

## Versioning and checkpoints

The text-core architecture retains checkpoint compatibility with v0.5. Weight files are not part of this repository to avoid bloating Git. Keep them locally under `runs/`; distribute future model weights through GitHub Releases (below GitHub limits) or a dedicated model host with checksums.
