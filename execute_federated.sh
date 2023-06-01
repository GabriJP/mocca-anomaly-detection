#!/usr/bin/env bash

GID=${1:-"fed_exp_noname"}
COMMON_OPTS="--load-lstm --bidirectional --clip-length=16 --code-length=512 --dropout=0.3 --idx-list-enc=3,4,5,6 --wandb_group ${GID}"

cd "${HOME}/mocca-anomaly-detection" || exit
git pull

exec_client() {
  echo "Sending commands to ${CLIENT_NAME}"
  # shellcheck disable=SC2087
  ssh "$CLIENT_NAME" bash <<EOC
cd "\${HOME}/mocca-anomaly-detection" || exit
rm -r output/* nohup.out 2>/dev/null
git pull
export PATH="\${HOME}/miniconda3/condabin:$PATH"
eval "\$(conda shell.bash hook)"
conda activate mocca || exit
export FLWR_TELEMETRY_ENABLED=0
nohup python fed.py client $COMMON_OPTS --data-path $DATA_PATH --wandb_name $WANDB_NAME --batch-size $BATCH_SIZE >${GID}.log 2>&1 </dev/null &
EOC
}

# shellcheck disable=SC2029
echo "Starting server"
eval "$(conda shell.bash hook)"
conda activate mocca
nohup python fed.py server --num_rounds 50 --epochs 2 --warm_up_n_epochs=0 --proximal_mu 1 >"${GID}.log" 2>&1 </dev/null &
#SERVER_PID=$!
echo "Delay"
sleep 10

DATA_PATH="data/UCSDped2"
WANDB_NAME="platano"
CLIENT_NAME="platano"
BATCH_SIZE=8
exec_client

DATA_PATH="data/UCSDped1"
WANDB_NAME="almogrote"
CLIENT_NAME="almogrote"
BATCH_SIZE=16
exec_client

#echo "Waiting for server to finish"
#wait $SERVER_PID

# ffmpeg -http_persistent 0 -protocol_whitelist file,http,https,tcp,tls,crypto -i url.m3u8 -c copy video.mp4
