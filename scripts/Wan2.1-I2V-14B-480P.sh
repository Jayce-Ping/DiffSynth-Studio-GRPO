model_path='/scratch/prj0000000275/jcy'
dataset_dir='/scratch/prj0000000275/jcy/visual_reasoning/sudoku/sudoku'
accelerate launch \
  --config_file scripts/configs/accelerate_config_14B.yaml \
  --num_processes 6 \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path $dataset_dir \
  --dataset_metadata_path ${dataset_dir}/train.csv \
  --height 480 \
  --width 480 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-I2V-14B-480P_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image" \