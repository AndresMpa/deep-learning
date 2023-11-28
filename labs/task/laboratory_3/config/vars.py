from dotenv import load_dotenv
import os

load_dotenv()

env_vars = {
    "net_arch": os.environ.get("NET_ARCH"),
    "use_cuda": os.environ.get("USE_CUDA") == "1",
    "img_size": int(os.environ.get("IMG_SIZE")),
    "data_path": os.environ.get("DATA_PATH"),
    "batch_size": int(os.environ.get("BATCH_SIZE")),
}
