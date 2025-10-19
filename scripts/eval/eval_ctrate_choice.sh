export PYTHONPATH=$PYTHONPATH:path/to/Med-2E3/src

MODEL_PATH="work_dirs/Med-2E3-M3D"

CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --cache_dir_hf path/to/cache_dir_hf \
    --llm_dtype bfloat16 \
    --data_path path/to/ctrate/valid_choice.json \
    --conv_version phi \
    --vision2d_data_path / \
    --vision3d_data_path path/to/CT-RATE/dataset/preprocessed_256/valid \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_length 768 \
    --num_beams 1 \
    --temperature 0

python scripts/eval/utils/eval_ctrate_choice.py \
    --answer_path path/to/ctrate/valid_choice.json \
    --predict_path $MODEL_PATH/eval/valid_choice.json \
    --result_path $MODEL_PATH/eval/valid_choice_results.json

CUDA_VISIBLE_DEVICES=0 python scripts/eval/utils/eval_ctrate_text.py \
    --answer_path path/to/ctrate/valid_choice.json \
    --predict_path $MODEL_PATH/eval/valid_choice.json \
    --result_path $MODEL_PATH/eval/valid_choice_results_text.json
