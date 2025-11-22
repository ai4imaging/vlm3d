#!/bin/bash


prompt='a cute dog'

safe_prompt_part=$(echo "$prompt" | tr ' ' '_' | tr -cd '[:alnum:]_-' | head -c 64)
unique_id=$(date +%Y%m%d_%H%M%S)_$$
logfile="./log/${unique_id}-${safe_prompt_part}.log"
echo "Log file: $logfile"

python launch.py \
--config custom/threestudio-mvdream/configs/mvdream-sd21-shading.yaml \
--train \
--gpu 0 \
system.prompt_processor.prompt="$prompt" >> "$logfile" 2>&1 &