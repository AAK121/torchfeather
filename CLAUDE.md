# TorchFeather — Claude Instructions

## Project overview
DeepSeek V3-style MoE language model pretraining framework, built on top of torchtitan (BSD-3).
GitHub: https://github.com/AAK121/torchfeather

## Repository layout
```
torchfeather/           # main Python package
  trainer.py
  deepseek_v3/          # model: MLA attention, MoE, RoPE
  training_components/  # optimizer, LR scheduler, dataloader, metrics, checkpoint, tokenizer
  job_configs/          # presets (deepseek_v3_presets.py, kaggle_t4_presets.py)
  parallelism/          # DDP, FSDP2, TP, PP, CP, EP, device mesh
  streaming_datasets/
  runtime_utils/
kaggle_t4/              # Kaggle notebook + results
  torchfeather_kaggle_t4.ipynb
  kernel-metadata.json  # Kaggle notebook config (DO NOT edit dataset_sources manually)
  results_ddp_2000_steps/
  results_strategy_comparison_300_steps/
parallelism_examples/   # standalone examples
asset_scripts/          # tokenizer downloader
```

## Running locally (2-GPU, PowerShell)
```powershell
# Download tokenizer (once)
python asset_scripts/hf_asset_downloader.py --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer

# Single GPU
$env:TORCHFEATHER_CONFIG="kaggle_t4_single"; torchrun --standalone --nproc_per_node=1 -m torchfeather.trainer

# Two GPUs (DDP)
$env:TORCHFEATHER_CONFIG="kaggle_t4_ddp"; torchrun --standalone --nproc_per_node=2 -m torchfeather.trainer

# 2000-step run with validation
$env:TORCHFEATHER_CONFIG="kaggle_t4_ddp_2k"; torchrun --standalone --nproc_per_node=2 -m torchfeather.trainer
```
Outputs land in `outputs/<config>/`.

## Kaggle workflow
Everything is already set up on this machine. The API token is at
`C:\Users\Asus\.kaggle\kaggle.json`.

**Dataset** (the code): `alok121/torchfeather-code`
**Notebook**: `alok121/torchfeather-kaggle-t4`
Accelerator: GPU T4 x2. Internet must be ON (for tokenizer download).

### Push a new code version
```powershell
# Run from INSIDE the torchfeather folder, not from its parent
cd D:\torchfeather
kaggle datasets version -p . -m "describe what changed"
```

### Push and run the notebook
```powershell
kaggle kernels push -p D:\torchfeather\kaggle_t4
```
This uploads the notebook AND starts a new run automatically.

### Monitor the run
```powershell
kaggle kernels status alok121/torchfeather-kaggle-t4
# Poll every ~60s until status is "complete" or "error"
```

### Pull results when done
```powershell
kaggle kernels output alok121/torchfeather-kaggle-t4 -p D:\torchfeather\kaggle_t4\latest_output
```

### Typical update workflow (code change → new run)
1. Edit code in `D:\torchfeather`.
2. Push new dataset version (command above).
3. Edit `kernel-metadata.json` to bump `dataset_sources` version if needed (usually automatic).
4. Push notebook → run starts.
5. Poll status until complete (~15 min for 2k steps).
6. Pull output.

### Windows gotcha
Always pass `-p .` from inside the folder, not `-p D:\torchfeather` — the CLI has a path bug on Windows with absolute paths.

## Presets available
Defined in `torchfeather/job_configs/kaggle_t4_presets.py`:
- `kaggle_t4_single` — 1 GPU, 300 steps
- `kaggle_t4_ddp`   — 2 GPU DDP, 300 steps
- `kaggle_t4_fsdp`  — 2 GPU FSDP2, 300 steps
- `kaggle_t4_tp`    — 2 GPU tensor parallel, 300 steps
- `kaggle_t4_ddp_2k`— 2 GPU DDP, 2000 steps, validation every 250 steps

To add a new preset: add a function in `kaggle_t4_presets.py` and register it in `kaggle_t4_config_map`.

## Report
LaTeX source: `D:\torchfeather_report\latex\torchfeather_report.tex`
Compiled PDF: `D:\torchfeather_report\TorchFeather_Technical_Report.pdf`
Tectonic binary (may need re-downloading): `C:\Users\Asus\AppData\Local\Temp\claude\...\tectonic.exe`

To recompile:
```powershell
cd D:\torchfeather_report\latex
tectonic -X compile torchfeather_report.tex
```
Or upload the `latex/` folder to Overleaf.

## Git
One commit on `main`. To push changes:
```powershell
cd D:\torchfeather
git add -A
git commit -m "your message`n`nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin main
```

## Key design notes
- T4 (sm 7.5): no bf16 → `float32` only. No `torch._grouped_mm` → MoE uses per-expert for-loop fallback.
- `torch.compile` disabled for T4 (Triton not available on Windows either).
- Validation uses a held-out split of FineWeb `sample-10BT`; controlled by `config.validation.*`.
- Generation runs after the last step; output written to `outputs/<config>/samples.txt`.
- `os._exit(0)` at shutdown prevents SIGALRM false-failure on Kaggle.
- MFU ~27% on T4 fp32 (theoretical peak 8.1 TFLOPs).
