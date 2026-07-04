import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

mode = "test"

test_data_dir = os.environ.get("HIDING_TEST_DATA_DIR", str(ROOT / "data" / "test"))
checkpoint_path = os.environ.get(
    "HIDING_CHECKPOINT",
    str(ROOT / "model_zoo" / "default" / "checkpoint.pt"),
)

# Runtime defaults. These can all be overridden from test.py arguments.
device = os.environ.get("HIDING_DEVICE", "auto")
cuda_device_ids = os.environ.get("HIDING_CUDA_DEVICES", "")
batch_size_test = int(os.environ.get("HIDING_BATCH_SIZE", "2"))
noise_sigma = float(os.environ.get("HIDING_SIGMA", "20"))
resize_size_test = int(os.environ.get("HIDING_RESIZE", "512"))
num_workers_test = int(os.environ.get("HIDING_NUM_WORKERS", "2"))
seed = int(os.environ.get("HIDING_SEED", "1"))

save_processed_img = os.environ.get("HIDING_SAVE_IMAGES", "0") == "1"
resi_magnification = float(os.environ.get("HIDING_RESI_MAGNIFICATION", "5"))
output_dir = os.environ.get("HIDING_OUTPUT_DIR", str(ROOT / "results" / "eval"))
