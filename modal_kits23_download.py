import shutil
from pathlib import Path

import modal
from modal_resources import OPS_CPU, OPS_TIMEOUT_SECONDS


app = modal.App("kits23-download")

volume = modal.Volume.from_name("kits23-dataset", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .add_local_file("modal_resources.py", remote_path="/root/modal_resources.py")
    .run_commands(
        "git clone https://github.com/neheller/kits23 /root/kits23",
        "python -m pip install -e /root/kits23",
    )
)


def _seed_repo_dataset(repo_dataset: Path, target_dataset: Path) -> None:
    target_dataset.mkdir(parents=True, exist_ok=True)

    for src_case in repo_dataset.iterdir():
        if not src_case.is_dir() or not src_case.name.startswith("case_"):
            continue

        dst_case = target_dataset / src_case.name
        dst_case.mkdir(parents=True, exist_ok=True)

        src_seg = src_case / "segmentation.nii.gz"
        dst_seg = dst_case / "segmentation.nii.gz"
        if src_seg.exists() and not dst_seg.exists():
            shutil.copy2(src_seg, dst_seg)

        src_instances = src_case / "instances"
        dst_instances = dst_case / "instances"
        if src_instances.exists() and not dst_instances.exists():
            shutil.copytree(src_instances, dst_instances)


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=OPS_CPU,
    timeout=OPS_TIMEOUT_SECONDS,
)
def download_kits23():
    import kits23.download as kd

    repo_dataset = Path("/root/kits23/dataset")
    target_dataset = Path("/data/dataset")

    _seed_repo_dataset(repo_dataset, target_dataset)
    kd.DST_PTH = target_dataset
    kd.download_dataset()

    case_count = sum(1 for p in target_dataset.iterdir() if p.is_dir() and p.name.startswith("case_"))
    imaging_count = sum(
        1 for p in target_dataset.iterdir()
        if p.is_dir() and p.name.startswith("case_") and (p / "imaging.nii.gz").exists()
    )
    segmentation_count = sum(
        1 for p in target_dataset.iterdir()
        if p.is_dir() and p.name.startswith("case_") and (p / "segmentation.nii.gz").exists()
    )

    volume.commit()
    return {
        "dataset_dir": str(target_dataset),
        "case_count": case_count,
        "imaging_count": imaging_count,
        "segmentation_count": segmentation_count,
    }


@app.local_entrypoint()
def main():
    print(download_kits23.remote())
