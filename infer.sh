CUDA_VISIBLE_DEVICES=5 python src/inference.py \
    --image_path "./assets/data/0.png" \
    --ckpt_path "Path/to/MetaView/checkpoint" \
    --da3_giant_path "Path/to/DA3-GIANT/model" \
    --da3_depth_path "Path/to/DA3-NESTED/model" \
    --yaw -30 \
    --pitch 15 \
    --radius 3.5 \
    --output_path "./result/generated.png"