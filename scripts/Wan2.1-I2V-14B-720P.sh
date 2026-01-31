model_dir='/scratch/prj0000000275/jcy/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-720P/snapshots/8823af45fcc58a8aa999a54b04be9abc7d2aac98'
dataset_dir='/scratch/prj0000000275/jcy/visual_reasoning/sudoku/sudoku'
output_dir='/scratch/prj0000000275/jcy/visual_reasoning/sudoku/sudoku/checkpoints/Wan2.1-I2V-14B-720P_full'

accelerate launch \
  --config_file scripts/configs/accelerate_config_14B.yaml \
  --num_processes 6 \
  scripts/train.py \
  --dataset_base_path $dataset_dir \
  --dataset_metadata_path ${dataset_dir}/train.csv \
  --height 720 \
  --width 720 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-720P:${model_dir}/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-720P:${model_dir}/models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-720P:${model_dir}/Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-720P:${model_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path $output_dir \
  --trainable_models "dit" \
  --extra_inputs "input_image"