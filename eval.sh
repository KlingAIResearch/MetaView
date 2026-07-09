CUDA_VISIBLE_DEVICES=0 python -u src/metric.py \
    --data_path "Path/to/evaluation/folder" \
    > ./eval_output.txt  2>&1 