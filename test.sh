export WANDB_MODE=disabled
export FLWR_TELEMETRY_ENABLED=0

python fed.py server data/shanghaitech/complete/ --port=8081 --num-rounds=660 --epochs=2 --warm-up-n-epochs=0 --load-lstm --proximal-mu=1 --min-evaluate-clients=13 --min-available-clients=13 --initialization=xavier_uniform --wandb-group="${GID}" --min-fit-clients=2 --test-checkpoint=30 >test_server.log

python fed.py client citic:8081 --load-lstm --n-workers=4 --bidirectional --clip-length=16 --code-length=512 --dropout=0.3 --idx-list-enc=3,4,5,6 --parallel --continuous --wandb-group=test --data-path=data/shanghaitech/continuous_3/0/shang01 --wandb-name=test_client --batch-size=4 >test_client.log
