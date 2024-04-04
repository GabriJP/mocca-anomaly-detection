#!/usr/bin/env bash

GID=${1:-"continuous3"}
COMMON_OPTS="--load-lstm --bidirectional --clip-length=16 --code-length=512 --dropout=0.3 --idx-list-enc=3,4,5,6 --wandb-group=${GID}"
CLIENT_OPTS="citic:8081 --n-workers=4 --parallel --continuous $COMMON_OPTS"

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
ls data/shanghaitech/continuous_3/$NODE_N/ | xargs -i \
nohup sh -c "python fed.py client $CLIENT_OPTS --data-path=data/shanghaitech/continuous_3/$NODE_N/{} --wandb-name={} --batch-size=$BATCH_SIZE >${GID}_{}.log 2>&1 </dev/null &"
EOC
}

echo "Starting server"
# shellcheck disable=SC2086
nohup python fed.py server data/shanghaitech/complete/ --port=8081 --num-rounds=660 --epochs=2 --warm-up-n-epochs=0 --proximal-mu=1 --min-evaluate-clients=13 --min-available-clients=13 --min-fit-clients=2 --initialization=xavier_uniform --test-checkpoint=30 $COMMON_OPTS >"${GID}_server.log" 2>&1 </dev/null &
echo "Delay"
sleep 5

CLIENT_NAME="almogrote"
BATCH_SIZE=8
NODE_N=0
exec_client

CLIENT_NAME="gofio"
BATCH_SIZE=8
NODE_N=1
exec_client

CLIENT_NAME="platano"
BATCH_SIZE=8
NODE_N=2
exec_client
