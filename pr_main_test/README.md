# Test-Only Image Hiding Evaluation

This repository is a lightweight evaluation release for image hiding, secret
recovery, and denoising experiments. It intentionally excludes training scripts,
old experiment variants, caches, local datasets, and generated outputs.

The evaluator runs three inference tasks:

- hiding: `cover + secret -> stego`
- recovery: `stego -> recovered secret`
- denoising: `noisy secret -> denoised secret`

## Repository Layout

```text
.
├── test.py                         # test-only evaluation entry point
├── config.py                       # default paths and runtime options
├── models/network.py               # model definition used by the checkpoint
├── utils/dataset.py                # image-folder test loader
├── utils/image.py                  # PSNR, SSIM, MAE, RMSE helpers
├── model_zoo/default/checkpoint.pt # optional bundled checkpoint location
├── scripts/run_div2k_eval.sh       # DIV2K-style evaluation helper
└── docs/open_source_checklist.md   # release checklist
```

## Installation

Python 3.8+ is recommended. Install PyTorch for your CUDA version first, then
install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If you use Conda, this workspace was verified with:

```bash
conda run -n gps_plus python test.py --help
```

## Data

The evaluator expects one flat directory of RGB images. Images are sorted by
filename, resized to `--resize`, and paired inside each batch:

- first half of a batch: cover images
- second half of a batch: secret images

Use an even `--batch-size`; the default is `2`, which evaluates one pair at a
time.

Datasets are not included in this repository. For a DIV2K-style test folder,
run:

```bash
python test.py \
  --data-dir /path/to/div2k/test \
  --output-dir results/div2k_test \
  --no-save-images \
  --cuda-devices 0 \
  --num-workers 0
```

## Checkpoint

By default, the script looks for:

```text
model_zoo/default/checkpoint.pt
```

You can override it with either a command-line argument or an environment
variable:

```bash
python test.py --checkpoint /path/to/checkpoint.pt --data-dir /path/to/images
```

```bash
HIDING_CHECKPOINT=/path/to/checkpoint.pt \
HIDING_TEST_DATA_DIR=/path/to/images \
python test.py
```

## Quick Smoke Test

Run one batch without saving images:

```bash
python test.py \
  --data-dir /path/to/test/images \
  --max-batches 1 \
  --num-workers 0 \
  --no-save-images
```

Or use the helper script:

```bash
HIDING_TEST_DATA_DIR=/path/to/test/images HIDING_CUDA_DEVICES=0 bash scripts/run_smoke.sh
```

## Full Evaluation

```bash
python test.py \
  --data-dir /path/to/test/images \
  --checkpoint model_zoo/default/checkpoint.pt \
  --output-dir results/eval \
  --cuda-devices 0 \
  --batch-size 2 \
  --num-workers 0
```

Outputs:

- `test.log`
- `metrics.json`
- `metrics.csv`
- optional processed images under `cover/`, `secret/`, `stego/`,
  `secret_rev/`, `cover_resi/`, `secret_resi/`, `noisy/`, and `denoised/`

Image saving is disabled by default for cleaner runs. Add `--save-images` when
you need visual outputs.

## Reference Result

The cleanup run on the local DIV2K test split contained 100 images, evaluated as
50 cover/secret pairs with `resize=512`, `sigma=20`, and `batch_size=2`.

| Metric | Value |
| --- | ---: |
| stego PSNR | 39.23 |
| secret PSNR | 27.41 |
| noise PSNR | 22.51 |
| denoise PSNR | 32.28 |
| stego SSIM | 0.9835 |
| secret SSIM | 0.8588 |
| noise SSIM | 0.4582 |
| denoise SSIM | 0.9103 |

These numbers are provided as a local sanity reference, not as a formal
benchmark claim.

## Release Notes

This is a preliminary test-only release. Add your final project name, paper
metadata, license, and any release-specific wording before making the repository
public.
