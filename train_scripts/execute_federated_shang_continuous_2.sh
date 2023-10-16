#!/usr/bin/env bash

GID=${1:-"continuous2"}
COMMON_OPTS="almogrote:50011 --load-lstm --n_workers 2 --bidirectional --clip-length=16 --code-length=512 --dropout=0.3 --idx-list-enc=3,4,5,6 --parallel --continuous --wandb_group ${GID}"

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
ls data/shanghaitech/continuous_2/$NODE_N/ | xargs -i \
nohup sh -c "python fed.py client $COMMON_OPTS --data-path data/shanghaitech/continuous_2/$NODE_N/{} --wandb_name {} --batch-size $BATCH_SIZE >${GID}_{}.log 2>&1 </dev/null &"
EOC
}

echo "Starting server"
nohup python fed.py server --port 50011 --num_rounds 50 --epochs 2 --warm_up_n_epochs=0 --proximal_mu 1 --min_evaluate_clients 0 --min_available_clients 13 --wandb_group "${GID}" --test_checkpoint 5 >"${GID}_server.log" 2>&1 </dev/null &
echo "Delay"
sleep 5

CLIENT_NAME="almogrote"
BATCH_SIZE=16
NODE_N=0
exec_client

CLIENT_NAME="platano"
BATCH_SIZE=6
NODE_N=1
exec_client
