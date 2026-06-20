import torch

checks = []
checks.append(("PyTorch", torch.__version__, True))
checks.append(("CUDA available", str(torch.cuda.is_available()), torch.cuda.is_available()))
if torch.cuda.is_available():
    checks.append(("GPU", torch.cuda.get_device_name(0), True))
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    checks.append(("VRAM", f"{vram:.1f} GB", vram >= 8))

try:
    from ultralytics import YOLO
    checks.append(("ultralytics/YOLOv8", "OK", True))
except ImportError as e:
    checks.append(("ultralytics/YOLOv8", str(e), False))

try:
    import kornia
    checks.append(("kornia", kornia.__version__, True))
except ImportError as e:
    checks.append(("kornia", str(e), False))

try:
    from pytorch_grad_cam import GradCAM
    checks.append(("pytorch-grad-cam", "OK", True))
except ImportError as e:
    checks.append(("pytorch-grad-cam", str(e), False))

try:
    import shap
    checks.append(("shap", shap.__version__, True))
except ImportError as e:
    checks.append(("shap", str(e), False))

try:
    import streamlit
    checks.append(("streamlit", streamlit.__version__, True))
except ImportError as e:
    checks.append(("streamlit", str(e), False))

try:
    import mlflow
    checks.append(("mlflow", mlflow.__version__, True))
except ImportError as e:
    checks.append(("mlflow", str(e), False))

print("\n" + "="*55)
print("  REATS Environment Check")
print("="*55)
all_pass = True
for name, value, status in checks:
    icon = "✓" if status else "✗"
    print(f"  {icon}  {name:<25} {value}")
    if not status:
        all_pass = False
print("="*55)
print("  ✓ Sẵn s\xe0ng!" if all_pass else "  ✗ C\xf3 lỗi — kiểm tra lại mục ✗")
print("="*55 + "\n")
