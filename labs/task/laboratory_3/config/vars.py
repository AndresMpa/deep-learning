from dotenv import load_dotenv

from types import SimpleNamespace

import os

load_dotenv()

env_vars = SimpleNamespace(
    dataset=os.environ.get("DATASET"),
    net_arch=os.environ.get("NET_ARCH"),
    use_cuda=os.environ.get("USE_CUDA") == "1",
    img_size=int(os.environ.get("IMG_SIZE")),

    # Data sets directory
    data_path=os.environ.get("DATA_PATH"),
    batch_size=int(os.environ.get("BATCH_SIZE")),

    # Results directory
    results_data=os.environ.get("RESULTS_DATA"),

    # Hyper params
    iterations=int(os.environ.get("ITERATIONS")),
    learning_rate=float(os.environ.get("LEARNING_RATE")),
    momentum_value=float(os.environ.get("MOMENTUM_VALUE"))
)
