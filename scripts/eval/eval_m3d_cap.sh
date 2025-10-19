export PYTHONPATH=$PYTHONPATH:path/to/Med-2E3/src

MODEL_PATH="work_dirs/Med-2E3-M3D"

CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --cache_dir_hf path/to/cache_dir_hf \
    --llm_dtype bfloat16 \
    --data_path path/to/m3d_cap.json \
    --conv_version phi \
    --vision2d_data_path / \
    --vision3d_data_path /path/to/M3D/npys_256 \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_length 768 \
    --num_beams 1 \
    --temperature 0

CUDA_VISIBLE_DEVICES=0 python scripts/eval/utils/convert_output_to_m3d_cap.py \
    --input_path $MODEL_PATH/eval/m3d_cap.json \
    --answer_path path/to/m3d_cap_answers.json \
    --output_path $MODEL_PATH/eval/m3d_cap_std.json
