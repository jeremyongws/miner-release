#!/bin/bash
# Chasm's take on maximizing 4090's on a device. We optimize based on number of gpus available, running a mix of
# non sdxl models and mistral to fully utilize the stack.
apt-get update
apt-get install -y htop nano sudo screen
sudo apt-get install -y python3 python3-pip python3-venv nvtop jq bc
sudo apt -y install npm
sudo npm -y install pm2 -g

apt-get -y install cuda-toolkit-12-1
pip install -r requirements.txt
pip install vllm python-dotenv toml openai triton==2.1.0 wheel packaging
pip install flash-attn

python 4090.py
gpu_num=$?

chmod +x llm-miner-starter.sh
pm2 start sd-miner-v1.1.0.py --name sd-miner --interpreter python3  -- --log-level DEBUG --auto-confirm yes

count=$((gpu_num-1))
for i in $(seq 0 $count);do
	pm2 start llm-miner-starter.sh --name llm-$i -- openhermes-2.5-mistral-7b-gptq --miner-id-index $i --port 805$i --gpu-ids $i
done