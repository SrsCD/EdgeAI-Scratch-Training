"""Offline, no-training CUDA and six-class scratch architecture verification."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
for key, relative in {
    "YOLO_CONFIG_DIR": ".cache/ultralytics",
    "TORCH_HOME": ".cache/torch",
    "MPLCONFIGDIR": ".cache/matplotlib",
}.items():
    local = ROOT / relative
    local.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(local)
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"
os.environ["WANDB_MODE"] = "disabled"


def forbidden(*args, **kwargs):
    raise RuntimeError("Stage 1 forbids network access and checkpoint loading")


def main() -> int:
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "virtual_environment": sys.prefix != sys.base_prefix,
        "base_python": sys.base_prefix,
        "nvcc_path": shutil.which("nvcc"),
        "network_blocked": True,
        "checkpoint_loading_blocked": True,
        "training_performed": False,
        "dataset_accessed": False,
        "evaluation_performed": False,
    }
    try:
        assert Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(), "Use project .venv"
        report["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
            text=True,
        ).strip()
        socket.socket.connect = forbidden
        socket.socket.connect_ex = forbidden
        socket.create_connection = forbidden

        import torch
        torch.load = forbidden
        torch.hub.load_state_dict_from_url = forbidden
        import torchvision
        import ultralytics
        import yaml
        import imagehash
        from pycocotools import mask as coco_mask
        from ultralytics import YOLO
        from ultralytics.nn.tasks import yaml_model_load
        from edgeai_scratch import CLASS_NAMES

        report["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "ultralytics", "numpy", "Pillow", "ImageHash", "pycocotools")
        }
        # Confirm the future evaluation extension imports; do not score anything.
        assert callable(coco_mask.encode) and callable(imagehash.phash)
        report["evaluation_dependency_imports_passed"] = True
        report["torch_cuda_runtime"] = torch.version.cuda
        report["cudnn_version"] = torch.backends.cudnn.version()
        report["cuda_available"] = torch.cuda.is_available()
        assert report["cuda_available"], "CUDA unavailable"
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        report["compute_capability"] = list(torch.cuda.get_device_capability(0))
        report["compiled_cuda_architectures"] = torch.cuda.get_arch_list()

        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        a = torch.randn(256, 256)
        expected = a @ a.T
        actual = (a.cuda() @ a.cuda().T).cpu()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        report["cuda_matmul_passed"] = True
        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], device="cuda", dtype=torch.float32)
        kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
        assert kept.cpu().tolist() == [0]
        report["torchvision_cuda_nms_passed"] = True

        # Resolve the installed YOLO11 architecture; never open any .pt file.
        architecture = yaml_model_load("yolo11n.yaml")
        assert architecture["scale"] == "n"
        source = Path(ultralytics.__file__).parent / "cfg/models/11/yolo11.yaml"
        report["upstream_architecture_path"] = str(source)
        report["upstream_architecture_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        architecture["nc"] = len(CLASS_NAMES)
        architecture.pop("yaml_file", None)
        config_path = ROOT / "configs/yolo11n.yaml"
        config_path.write_text(
            "# Ultralytics YOLO11 YAML; nc=6, scale=n. See models/architecture_provenance.json.\n"
            + yaml.safe_dump(architecture, sort_keys=False), encoding="utf-8"
        )
        model = YOLO(str(config_path), task="detect")
        model.overrides["pretrained"] = False
        model.model.args["pretrained"] = False
        model.model.names = dict(enumerate(CLASS_NAMES))
        assert model.model.yaml["nc"] == 6
        assert model.model.model[-1].nc == 6
        assert not model.ckpt and model.ckpt_path is None
        assert model.overrides["pretrained"] is False
        assert any(p.requires_grad for p in model.model.parameters())
        model.model.to("cuda").eval()
        with torch.inference_mode():
            output = model.model(torch.zeros(1, 3, 320, 320, device="cuda"))
        predictions = output[0] if isinstance(output, tuple) else output
        assert predictions.shape[1] == 4 + 6
        assert torch.isfinite(predictions).all().item()
        torch.cuda.synchronize()
        report.update({
            "architecture": "yolo11n.yaml",
            "architecture_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "nc": 6,
            "classes": dict(enumerate(CLASS_NAMES)),
            "pretrained": False,
            "checkpoint": None,
            "checkpoint_loaded": False,
            "parameter_count": sum(p.numel() for p in model.model.parameters()),
            "cuda_forward_input_shape": [1, 3, 320, 320],
            "cuda_forward_output_shape": list(predictions.shape),
            "cuda_forward_finite": True,
            "initialization_passed": True,
            "status": "passed",
        })
        provenance = {
            "source_url": "https://github.com/ultralytics/ultralytics/blob/v8.3.233/ultralytics/cfg/models/11/yolo11.yaml",
            "package": f"ultralytics=={ultralytics.__version__}",
            "license": "AGPL-3.0; review Ultralytics license obligations before distribution",
            "license_url": "https://github.com/ultralytics/ultralytics/blob/v8.3.233/LICENSE",
            "source_sha256": report["upstream_architecture_sha256"],
            "generated_sha256": report["architecture_sha256"],
            "changes": ["nc: 6", "scale: n", "YAML serialized from installed yaml_model_load output"],
            "weights_loaded": False,
        }
        (ROOT / "models/architecture_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    destination = ROOT / "reports/environment/stage1_verification.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
