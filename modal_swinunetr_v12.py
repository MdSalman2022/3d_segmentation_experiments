 """
Run the SwinUNETR V12 KiTS23 experiment on Modal.

Typical workflow:

1. Upload the local dataset folder into a Modal Volume:
   modal run modal_swinunetr_v12.py --action upload

2. Inspect what is available remotely:
   modal run modal_swinunetr_v12.py --action summary

3. If your dataset folders contain segmentations but some cases are missing
   imaging.nii.gz, fetch those missing imaging files inside Modal:
   modal run modal_swinunetr_v12.py --action download_imaging

4. Build the preprocessing cache:
   modal run modal_swinunetr_v12.py --action build_cache

5. Run training:
   modal run modal_swinunetr_v12.py --action train --mode quick_test
   modal run modal_swinunetr_v12.py --action train --mode full

Notes:
- This wrapper intentionally does not use add_local_dir(".") because this repo
  contains large datasets, outputs, and a local virtual environment.
- The training code itself still lives in swinunetr_v11.py and
  swinunetr_v12.py. This file only packages and launches it on Modal.
- The official KiTS helper in this repo downloads imaging files only. If you do
  not already have the case folders and segmentations, upload the dataset from
  your local machine or mount it from cloud storage.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import modal


APP_NAME = "kidney-swinunetr-v12"
PYTHON_VERSION = "3.11"
GPU_TYPE = "L40S"

DATA_VOLUME_NAME = "kidney-kits23-data"
OUTPUT_VOLUME_NAME = "kidney-kits23-output"
HF_CACHE_VOLUME_NAME = "kidney-hf-cache"

REMOTE_CODE_DIR = "/root/code"
REMOTE_DATA_ROOT = "/vol/data"
REMOTE_OUTPUT_ROOT = "/vol/output"
REMOTE_DATASET_SUBDIR = "kits23"
REMOTE_DATASET_DIR = f"{REMOTE_DATA_ROOT}/{REMOTE_DATASET_SUBDIR}"
REMOTE_HF_CACHE = "/root/.cache/huggingface"


image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        "torch",
        "torchvision",
        "monai",
        "nibabel",
        "numpy",
        "scipy",
        "tqdm",
        "transformers",
        "accelerate",
    )
    .env(
        {
            "PYTHONPATH": REMOTE_CODE_DIR,
            "HF_HOME": REMOTE_HF_CACHE,
            "TRANSFORMERS_CACHE": REMOTE_HF_CACHE,
        }
    )
    .add_local_file("swinunetr_v11.py", remote_path=f"{REMOTE_CODE_DIR}/swinunetr_v11.py")
    .add_local_file("swinunetr_v12.py", remote_path=f"{REMOTE_CODE_DIR}/swinunetr_v12.py")
    .add_local_dir("dinov3", remote_path=f"{REMOTE_CODE_DIR}/dinov3")
)

app = modal.App(APP_NAME, image=image)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


def _remote_dataset_dir(dataset_subdir: str) -> Path:
    clean = dataset_subdir.strip().strip("/")
    if not clean:
        clean = REMOTE_DATASET_SUBDIR
    return Path(REMOTE_DATA_ROOT) / clean


def _remote_output_dir(output_name: str, mode: str) -> Path:
    name = output_name.strip() if output_name else ""
    if not name:
        name = f"swin_v12_modal_{mode}"
    return Path(REMOTE_OUTPUT_ROOT) / name


def _normalize_extra_args(extra_args: str) -> list[str]:
    return shlex.split(extra_args) if extra_args.strip() else []


def _upload_local_dataset(local_dataset_dir: str, dataset_subdir: str) -> dict:
    local_dir = Path(local_dataset_dir).resolve()
    if not local_dir.exists():
        raise FileNotFoundError(f"Local dataset directory does not exist: {local_dir}")
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Local dataset path is not a directory: {local_dir}")

    remote_dir = "/" + dataset_subdir.strip().strip("/")
    if remote_dir == "/":
        remote_dir = f"/{REMOTE_DATASET_SUBDIR}"

    print(f"Uploading {local_dir} -> {DATA_VOLUME_NAME}:{remote_dir}")
    with data_volume.batch_upload() as batch:
        batch.put_directory(str(local_dir), remote_dir)

    case_count = sum(1 for p in local_dir.iterdir() if p.is_dir() and p.name.startswith("case_"))
    return {
        "local_dataset_dir": str(local_dir),
        "remote_dataset_dir": remote_dir,
        "case_count": case_count,
    }


@app.function(
    volumes={REMOTE_DATA_ROOT: data_volume},
    cpu=2.0,
    memory=4096,
    timeout=60 * 30,
)
def dataset_summary(dataset_subdir: str = REMOTE_DATASET_SUBDIR) -> dict:
    dataset_dir = _remote_dataset_dir(dataset_subdir)
    if not dataset_dir.exists():
        return {
            "dataset_dir": str(dataset_dir),
            "exists": False,
            "message": "Dataset volume path does not exist yet. Upload data first.",
        }

    case_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("case_"))
    with_imaging = sum((case_dir / "imaging.nii.gz").exists() for case_dir in case_dirs)
    with_segmentation = sum((case_dir / "segmentation.nii.gz").exists() for case_dir in case_dirs)
    missing_imaging = [case_dir.name for case_dir in case_dirs if not (case_dir / "imaging.nii.gz").exists()]
    missing_segmentation = [
        case_dir.name for case_dir in case_dirs if not (case_dir / "segmentation.nii.gz").exists()
    ]

    return {
        "dataset_dir": str(dataset_dir),
        "exists": True,
        "case_count": len(case_dirs),
        "cases_with_imaging": with_imaging,
        "cases_with_segmentation": with_segmentation,
        "missing_imaging_count": len(missing_imaging),
        "missing_segmentation_count": len(missing_segmentation),
        "missing_imaging_examples": missing_imaging[:10],
        "missing_segmentation_examples": missing_segmentation[:10],
    }


@app.function(
    volumes={REMOTE_DATA_ROOT: data_volume},
    cpu=2.0,
    memory=4096,
    timeout=60 * 60 * 6,
)
def download_missing_imaging(
    dataset_subdir: str = REMOTE_DATASET_SUBDIR,
    overwrite: bool = False,
    limit: int = 0,
) -> dict:
    dataset_dir = _remote_dataset_dir(dataset_subdir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Remote dataset path does not exist: {dataset_dir}. "
            "Upload the dataset case folders before downloading imaging."
        )

    case_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("case_"))
    targets = []
    for case_dir in case_dirs:
        segmentation_path = case_dir / "segmentation.nii.gz"
        imaging_path = case_dir / "imaging.nii.gz"
        if not segmentation_path.exists():
            continue
        if overwrite or not imaging_path.exists():
            targets.append(case_dir)

    if limit > 0:
        targets = targets[:limit]

    downloaded = 0
    skipped = 0
    failed: list[dict[str, str]] = []

    for case_dir in targets:
        case_name = case_dir.name
        imaging_path = case_dir / "imaging.nii.gz"
        if imaging_path.exists() and not overwrite:
            skipped += 1
            continue

        case_num = int(case_name.split("_")[1])
        url = f"https://kits19.sfo2.digitaloceanspaces.com/master_{case_num:05d}.nii.gz"
        tmp_path = case_dir / ".partial.imaging.nii.gz"
        print(f"Downloading {case_name} from {url}")
        try:
            urllib.request.urlretrieve(url, str(tmp_path))
            shutil.move(str(tmp_path), str(imaging_path))
            downloaded += 1
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            failed.append({"case": case_name, "error": str(exc)})

    data_volume.commit()
    return {
        "dataset_dir": str(dataset_dir),
        "target_count": len(targets),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed_count": len(failed),
        "failed_examples": failed[:10],
    }


def _run_v12_subprocess(
    mode: str,
    dataset_subdir: str,
    output_name: str,
    extra_args: list[str],
    use_local_dinov3: bool,
    checkpoint: str | None = None,
) -> dict:
    dataset_dir = _remote_dataset_dir(dataset_subdir)
    output_dir = _remote_output_dir(output_name, mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        f"{REMOTE_CODE_DIR}/swinunetr_v12.py",
        "--mode",
        mode,
        "--kits23-dir",
        str(dataset_dir),
        "--output-dir",
        str(output_dir),
    ]
    if use_local_dinov3:
        cmd.append("--local-dinov3")
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    cmd.extend(extra_args)

    print("Running command:")
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)

    return {
        "mode": mode,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "used_local_dinov3": use_local_dinov3,
        "extra_args": extra_args,
    }


@app.function(
    volumes={
        REMOTE_DATA_ROOT: data_volume,
        REMOTE_OUTPUT_ROOT: output_volume,
        REMOTE_HF_CACHE: hf_cache_volume,
    },
    cpu=8.0,
    memory=32768,
    timeout=60 * 60 * 8,
)
def build_cache_remote(
    dataset_subdir: str = REMOTE_DATASET_SUBDIR,
    output_name: str = "",
    extra_args: list[str] | None = None,
    use_local_dinov3: bool = False,
) -> dict:
    result = _run_v12_subprocess(
        mode="build_cache",
        dataset_subdir=dataset_subdir,
        output_name=output_name,
        extra_args=extra_args or [],
        use_local_dinov3=use_local_dinov3,
    )
    data_volume.commit()
    output_volume.commit()
    return result


@app.function(
    volumes={
        REMOTE_DATA_ROOT: data_volume,
        REMOTE_OUTPUT_ROOT: output_volume,
        REMOTE_HF_CACHE: hf_cache_volume,
    },
    gpu=GPU_TYPE,
    cpu=8.0,
    memory=65536,
    timeout=60 * 60 * 24,
)
def train_remote(
    mode: str = "quick_test",
    dataset_subdir: str = REMOTE_DATASET_SUBDIR,
    output_name: str = "",
    extra_args: list[str] | None = None,
    use_local_dinov3: bool = False,
) -> dict:
    result = _run_v12_subprocess(
        mode=mode,
        dataset_subdir=dataset_subdir,
        output_name=output_name,
        extra_args=extra_args or [],
        use_local_dinov3=use_local_dinov3,
    )
    data_volume.commit()
    output_volume.commit()
    hf_cache_volume.commit()
    return result


@app.function(
    volumes={
        REMOTE_DATA_ROOT: data_volume,
        REMOTE_OUTPUT_ROOT: output_volume,
        REMOTE_HF_CACHE: hf_cache_volume,
    },
    gpu=GPU_TYPE,
    cpu=8.0,
    memory=65536,
    timeout=60 * 60 * 8,
)
def evaluate_remote(
    checkpoint: str,
    dataset_subdir: str = REMOTE_DATASET_SUBDIR,
    output_name: str = "",
    extra_args: list[str] | None = None,
    use_local_dinov3: bool = False,
) -> dict:
    result = _run_v12_subprocess(
        mode="evaluate",
        dataset_subdir=dataset_subdir,
        output_name=output_name,
        extra_args=extra_args or [],
        use_local_dinov3=use_local_dinov3,
        checkpoint=checkpoint,
    )
    output_volume.commit()
    return result


@app.local_entrypoint()
def main(
    action: str = "summary",
    mode: str = "quick_test",
    local_dataset_dir: str = "./kits23/dataset",
    dataset_subdir: str = REMOTE_DATASET_SUBDIR,
    output_name: str = "",
    extra_args: str = "",
    overwrite_imaging: bool = False,
    limit: int = 0,
    checkpoint: str = "",
    use_local_dinov3: bool = False,
):
    parsed_extra_args = _normalize_extra_args(extra_args)

    if action == "upload":
        result = _upload_local_dataset(local_dataset_dir, dataset_subdir)
    elif action == "summary":
        result = dataset_summary.remote(dataset_subdir)
    elif action == "download_imaging":
        result = download_missing_imaging.remote(dataset_subdir, overwrite_imaging, limit)
    elif action == "build_cache":
        result = build_cache_remote.remote(dataset_subdir, output_name, parsed_extra_args, use_local_dinov3)
    elif action == "train":
        result = train_remote.remote(mode, dataset_subdir, output_name, parsed_extra_args, use_local_dinov3)
    elif action == "evaluate":
        if not checkpoint.strip():
            raise SystemExit("`--checkpoint` is required when action=evaluate")
        result = evaluate_remote.remote(
            checkpoint.strip(),
            dataset_subdir,
            output_name,
            parsed_extra_args,
            use_local_dinov3,
        )
    else:
        raise SystemExit(
            "Unknown action. Use one of: upload, summary, download_imaging, "
            "build_cache, train, evaluate"
        )

    print(json.dumps(result, indent=2))
