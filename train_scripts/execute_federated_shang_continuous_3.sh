#!/usr/bin/env bash

GID=${1:-"continuous3"}
COMMON_OPTS="citic:8081 --load-lstm --n-workers=4 --bidirectional --clip-length=16 --code-length=512 --dropout=0.3 --idx-list-enc=3,4,5,6 --parallel --continuous --wandb-group=${GID}"

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
nohup sh -c "python fed.py client $COMMON_OPTS --data-path=data/shanghaitech/continuous_3/$NODE_N/{} --wandb-name={} --batch-size=$BATCH_SIZE >${GID}_{}.log 2>&1 </dev/null &"
EOC
}

echo "Starting server"
nohup python fed.py server data/shanghaitech/complete/ --port=8081 --num-rounds=660 --epochs=2 --warm-up-n-epochs=0 --load-lstm --proximal-mu=1 --min-evaluate-clients=13 --min-available-clients=13 --initialization=xavier_uniform --wandb-group="${GID}" --min-fit-clients=2 --test-checkpoint=30 >"${GID}_server.log" 2>&1 </dev/null &
echo "Delay"
sleep 5

CLIENT_NAME="almogrote"
BATCH_SIZE=16
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
