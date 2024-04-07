import torch
import sys
import os
import toml
import subprocess
import logging
from logging.handlers import RotatingFileHandler

# Set logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("reg.log", maxBytes=10000000, backupCount=5),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    # make sure all 4090's
    for i in range(num_gpus):
        # raise error if not
        assert torch.cuda.get_device_properties(i).name == "NVIDIA GeForce RTX 4090"
    # sys.exit(num_gpus)

    eth_wallet = input("Enter your intended reward address:")

    logger.info(
        f"Creating .env file for {eth_wallet} with {num_gpus} 4090's...")
    with open(".env", "w") as f:
        for i in range(num_gpus):
            f.write(f"MINER_ID_{i}={eth_wallet}\n")

    logger.info(
        f"Modifying the .toml file with {num_gpus} 4090's..."
    )
    # estimated best amount of child processes
    child_processes = num_gpus*2
    data = toml.load("config.toml") 
    data['system']['num_cuda_devices']=num_gpus
    data['system']['num_child_process']=child_processes

    with open("config.toml", "w") as f:
        toml.dump(data, f)
        f.close()
    sys.exit(num_gpus)
    