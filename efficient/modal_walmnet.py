from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal


APP_NAME = "walmnet-kits23"
VOLUME_NAME = "walmnet-kits23-data"
VOLUME_MOUNT = Path("/data")
KITS23_REPO_DIR = VOLUME_MOUNT / "kits23"
KITS23_DATASET_DIR = KITS23_REPO_DIR / "dataset"
OUTPUT_ROOT = VOLUME_MOUNT / "output"

# Edit this constant if you want a cheaper/smaller GPU for smoke tests.
# Modal GPU names: https://modal.com/docs/guide/gpu
TRAIN_GPU = "A100"
MAX_TIMEOUT = 24 * 60 * 60

data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .uv_pip_install(
        "torch",
        "monai",
        "nibabel",
        "numpy",
        "scipy",
        "tqdm",
    )
    # Modal 1.x requires local project files to be included explicitly when
    # they are not imported by the app module.
    # Source: https://modal.com/docs/sdk/py/latest/modal.Image#add_local_file
    .add_local_file(Path(__file__).with_name("walmnet_kits23.py"), "/root/walmnet_kits23.py")
)

app = modal.App(APP_NAME, image=image)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _dataset_stats() -> dict[str, Any]:
    if not KITS23_DATASET_DIR.exists():
        return {"dataset_dir": str(KITS23_DATASET_DIR), "cases": 0, "images": 0, "segmentations": 0}

    cases = sorted(p for p in KITS23_DATASET_DIR.iterdir() if p.is_dir() and p.name.startswith("case_"))
    images = sum((case / "imaging.nii.gz").exists() for case in cases)
    segmentations = sum((case / "segmentation.nii.gz").exists() for case in cases)
    return {
        "dataset_dir": str(KITS23_DATASET_DIR),
        "cases": len(cases),
        "images": images,
        "segmentations": segmentations,
    }


def _output_dir(output_subdir: str, mode: str) -> Path:
    if output_subdir:
        return OUTPUT_ROOT / output_subdir
    return OUTPUT_ROOT / ("walmnet_quick" if mode == "quick_test" else f"walmnet_{mode}")


def _apply_overrides(
    cfg: dict[str, Any],
    output_subdir: str,
    patch_size: str,
    batch_size: int,
    num_epochs: int,
    lr: float,
    mixer: str,
) -> dict[str, Any]:
    cfg["kits23_dir"] = str(KITS23_DATASET_DIR)
    cfg["output_dir"] = str(_output_dir(output_subdir, cfg.get("mode", "full")))

    if patch_size:
        cfg["patch_size"] = tuple(int(v.strip()) for v in patch_size.split(","))
    if batch_size > 0:
        cfg["batch_size"] = batch_size
    if num_epochs > 0:
        cfg["num_epochs"] = num_epochs
    if lr > 0:
        cfg["lr"] = lr
    if mixer:
        cfg["mixer"] = mixer
    return cfg


@app.function(
    volumes={VOLUME_MOUNT: data_volume},
    cpu=4,
    memory=16384,
    timeout=MAX_TIMEOUT,
)
def download_kits23(force: bool = False) -> dict[str, Any]:
    """Clone the official KiTS23 repo into the Volume and download imaging."""
    VOLUME_MOUNT.mkdir(parents=True, exist_ok=True)

    if not (KITS23_REPO_DIR / ".git").exists():
        if KITS23_REPO_DIR.exists():
            raise RuntimeError(
                f"{KITS23_REPO_DIR} exists but is not a git checkout. "
                "Remove or rename it from a Modal shell before retrying."
            )
        _run(["git", "clone", "--depth", "1", "https://github.com/neheller/kits23", str(KITS23_REPO_DIR)])
    else:
        _run(["git", "pull", "--ff-only"], cwd=KITS23_REPO_DIR)

    if force and KITS23_DATASET_DIR.exists():
        for image_path in KITS23_DATASET_DIR.glob("case_*/imaging.nii.gz"):
            image_path.unlink()

    before = _dataset_stats()
    print("Before download: " + json.dumps(before, indent=2), flush=True)

    # This runs the official downloader from the cloned KiTS23 repository.
    # The official CLI docs recommend `kits23_download_data`; the module entry
    # point uses the same source and writes into this Volume-backed checkout.
    # Source: https://github.com/neheller/kits23#data-download
    expected_images = max(before["cases"], before["segmentations"])
    if expected_images == 0 or before["images"] < expected_images:
        _run([sys.executable, "-m", "kits23.download"], cwd=KITS23_REPO_DIR)
    else:
        print("KiTS23 imaging already appears complete; skipping download.", flush=True)

    after = _dataset_stats()
    print("After download: " + json.dumps(after, indent=2), flush=True)
    data_volume.commit()
    return after


@app.function(timeout=30 * 60, cpu=2, memory=8192)
def selftest() -> None:
    sys.path.insert(0, "/root")
    import walmnet_kits23

    walmnet_kits23.selftest()


@app.function(
    volumes={VOLUME_MOUNT: data_volume},
    gpu=TRAIN_GPU,
    cpu=8,
    memory=65536,
    timeout=MAX_TIMEOUT,
)
def train_walmnet(
    mode: str = "quick_test",
    output_subdir: str = "",
    patch_size: str = "",
    batch_size: int = 0,
    num_epochs: int = 0,
    lr: float = 0.0,
    mixer: str = "lite",
) -> dict[str, str]:
    sys.path.insert(0, "/root")
    import walmnet_kits23

    if mode not in {"quick_test", "medium", "full"}:
        raise ValueError("mode must be one of: quick_test, medium, full")

    data_volume.reload()
    stats = _dataset_stats()
    if stats["images"] == 0 or stats["segmentations"] == 0:
        raise FileNotFoundError(
            f"KiTS23 is not ready at {KITS23_DATASET_DIR}. "
            "Run `modal run modal_walmnet.py --action download` first."
        )

    cfg = walmnet_kits23.get_config(mode)
    cfg["mode"] = mode
    cfg = _apply_overrides(cfg, output_subdir, patch_size, batch_size, num_epochs, lr, mixer)

    walmnet_kits23.train(cfg)
    data_volume.commit()

    out_dir = Path(cfg["output_dir"])
    return {
        "output_dir": str(out_dir),
        "best_checkpoint": str(out_dir / "best.pth"),
        "last_checkpoint": str(out_dir / "last.pth"),
        "summary": str(out_dir / "summary.json"),
    }


@app.function(
    volumes={VOLUME_MOUNT: data_volume},
    gpu=TRAIN_GPU,
    cpu=8,
    memory=65536,
    timeout=MAX_TIMEOUT,
)
def evaluate_walmnet(
    checkpoint: str,
    mode: str = "quick_test",
    output_subdir: str = "",
    patch_size: str = "",
    batch_size: int = 0,
    num_epochs: int = 0,
    lr: float = 0.0,
    mixer: str = "lite",
) -> dict[str, str]:
    sys.path.insert(0, "/root")
    import walmnet_kits23

    if not checkpoint:
        raise ValueError("checkpoint is required for evaluation")
    if mode not in {"quick_test", "medium", "full"}:
        raise ValueError("mode must be one of: quick_test, medium, full")

    data_volume.reload()
    cfg = walmnet_kits23.get_config(mode)
    cfg["mode"] = mode
    cfg = _apply_overrides(cfg, output_subdir, patch_size, batch_size, num_epochs, lr, mixer)

    walmnet_kits23.evaluate(cfg, checkpoint)
    data_volume.commit()

    out_dir = Path(cfg["output_dir"])
    return {"evaluation": str(out_dir / "evaluation.json")}


@app.local_entrypoint()
def main(
    action: str = "train",
    mode: str = "quick_test",
    output_subdir: str = "",
    checkpoint: str = "",
    patch_size: str = "",
    batch_size: int = 0,
    num_epochs: int = 0,
    lr: float = 0.0,
    mixer: str = "lite",
    skip_download: bool = False,
    force_download: bool = False,
) -> None:
    """Terminal entrypoint for Modal runs."""
    if action == "download":
        print(download_kits23.spawn(force=force_download).get())
        return

    if action == "selftest":
        selftest.remote()
        return

    if action == "train":
        if not skip_download:
            print(download_kits23.spawn(force=force_download).get())
        result = train_walmnet.spawn(
            mode=mode,
            output_subdir=output_subdir,
            patch_size=patch_size,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            mixer=mixer,
        ).get()
        print(result)
        return

    if action == "evaluate":
        result = evaluate_walmnet.spawn(
            checkpoint=checkpoint,
            mode=mode,
            output_subdir=output_subdir,
            patch_size=patch_size,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            mixer=mixer,
        ).get()
        print(result)
        return

    raise ValueError("action must be one of: download, selftest, train, evaluate")
