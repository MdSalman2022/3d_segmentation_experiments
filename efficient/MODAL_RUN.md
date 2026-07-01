# Running WaLM-Net on Modal

This repo now has a Modal runner in `modal_walmnet.py`. It creates a persistent
Modal Volume named `walmnet-kits23-data`, downloads KiTS23 into that Volume, and
runs `walmnet_kits23.py` against the remote dataset.

## 1. Install and authenticate Modal

```powershell
pip install modal
modal setup
```

If Windows cannot find the `modal` command, use:

```powershell
python -m modal setup
python -m modal run modal_walmnet.py --action selftest
```

## 2. Run a no-data self-test on Modal

```powershell
modal run modal_walmnet.py --action selftest
```

## 3. Download KiTS23 into Modal storage

```powershell
modal run --detach modal_walmnet.py --action download
```

The download function clones the official KiTS23 repository inside the Modal
Volume, then runs the official downloader module so the final layout is:

```text
/data/kits23/dataset/case_XXXXX/imaging.nii.gz
/data/kits23/dataset/case_XXXXX/segmentation.nii.gz
```

## 4. Start a quick smoke-training run

```powershell
modal run --detach modal_walmnet.py --action train --mode quick_test
```

The default training GPU is `A100`. To use another Modal GPU, edit `TRAIN_GPU`
near the top of `modal_walmnet.py`.

## 5. Start a longer run

```powershell
modal run --detach modal_walmnet.py --action train --mode medium --output-subdir walmnet_medium
```

For full training:

```powershell
modal run --detach modal_walmnet.py --action train --mode full --output-subdir walmnet_full
```

Useful overrides:

```powershell
modal run --detach modal_walmnet.py --action train --mode medium --batch-size 1 --patch-size 96,96,96 --num-epochs 20
```

## 6. Download results back to this machine

```powershell
modal volume get walmnet-kits23-data output/walmnet_quick/summary.json summary.json
modal volume get walmnet-kits23-data output/walmnet_quick/best.pth best.pth
modal volume get walmnet-kits23-data output/walmnet_quick/training.log training.log
```

## Sources

- Modal setup: https://modal.com/docs/guide
- Modal `run` CLI: https://modal.com/docs/cli/latest/run
- Modal Apps and local entrypoints: https://modal.com/docs/guide/apps
- Modal Images: https://modal.com/docs/guide/images
- Modal Volumes: https://modal.com/docs/guide/volumes
- Modal GPUs: https://modal.com/docs/guide/gpu
- KiTS23 official downloader: https://github.com/neheller/kits23#data-download
