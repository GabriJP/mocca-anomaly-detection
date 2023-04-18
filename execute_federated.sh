#!/usr/bin/env bash

cd "${HOME}/PycharmProjects/mocca-anomaly-detection/" || exit
git pull
export PATH="${HOME}/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate mocca
#gid=$(python -c "import wandb ; print(wandb.util.generate_id())")
gid="baseline_fed_ped2"
client_command="nohup ./execute.sh --data-path data/UCSDped2 --load-lstm --seed 3 --bidirectional --clip-length=16 --code-length=512 --dropout=0.3 --idx-list-enc=3,4,5,6 --wandb_group ${gid} >${gid}.log 2>&1 </dev/null &"

# shellcheck disable=SC2029
ssh xavier "nohup ./execute.sh --num_rounds 7 --epochs 2 --warm_up_n_epochs=1 --batch_size 4 --proximal_mu 1 >${gid}.log 2>&1 </dev/null &"
sleep 10
# shellcheck disable=SC2029
ssh citic "${client_command}"
# shellcheck disable=SC2029
ssh d11 "${client_command}"

# ffmpeg -http_persistent 0 -protocol_whitelist file,http,https,tcp,tls,crypto -i url.m3u8 -c copy video.mp4
