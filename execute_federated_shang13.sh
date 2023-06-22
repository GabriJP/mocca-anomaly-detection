#!/usr/bin/env bash

GID=${1:-"fed_shang13_autoname"}
COMMON_OPTS="--load-lstm --bidirectional --clip-length=16 --code-length=512 --dropout=0.3 --idx-list-enc=3,4,5,6 --seed 2 --parallel --wandb_group ${GID}"

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
export FLWR_TELEMETRY_ENABLED=0 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 UBLAS_WORKSPACE_CONFIG=:4096:8
nohup nice -n $NICE_N python fed.py client $COMMON_OPTS --data-path $DATA_PATH --wandb_name $WANDB_NAME --batch-size $BATCH_SIZE >${GID}_${WANDB_NAME}.log 2>&1 </dev/null &
EOC
}

# shellcheck disable=SC2029
echo "Starting server"
#eval "$(conda shell.bash hook)"
#conda activate mocca
export FLWR_TELEMETRY_ENABLED=0
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nohup python fed.py server --num_rounds 100 --epochs 2 --warm_up_n_epochs=0 --proximal_mu 1 --min_fit_clients 13 --min_evaluate_clients 13 --min_available_clients 13 >"${GID}_server.log" 2>&1 </dev/null &
#SERVER_PID=$!
echo "Delay"
sleep 5

CLIENT_NAME="almogrote"
BATCH_SIZE=16
NICE_N=0
for i in 01 05 04 08 03; do
  DATA_PATH="data/shang$i"
  WANDB_NAME="${CLIENT_NAME}_$i"
  exec_client
  NICE_N=$((NICE_N + 1))
done

CLIENT_NAME="platano"
BATCH_SIZE=8
NICE_N=0
for i in 06 02 09 11 13; do
  DATA_PATH="data/shang$i"
  WANDB_NAME="${CLIENT_NAME}_$i"
  exec_client
  NICE_N=$((NICE_N + 1))
done

CLIENT_NAME="citic"
BATCH_SIZE=4
NICE_N=0
for i in 12 07 10; do
  DATA_PATH="data/shang$i"
  WANDB_NAME="${CLIENT_NAME}_$i"
  exec_client
  NICE_N=$((NICE_N + 1))
done

#echo "Waiting for server to finish"
#wait $SERVER_PID
