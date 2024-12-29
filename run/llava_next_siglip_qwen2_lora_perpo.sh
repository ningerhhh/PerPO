export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# sh ./run/llava_next_siglip_qwen2_lora_perpo.sh

deepspeed perpo/models/llava_next/train_next_perpo_lora.py \
    --deepspeed ./perpo/models/llava_next/scripts/zero3.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --mm_projector_lr 5e-6 \
    --model_name_or_path /models/ocr_models_trained/llavanext-qwen2-siglip-page_ocr-50k-bs1-accm8-lrx3-nodes4-0914 \
    --version qwen_1_5 \
    --ours_data_path ./data/llava_next_siglip_qwen2_ocr_data.json \
    --image_folder ./data/books_arxiv_pdf_png_page/ \
    --mm_vision_tower_lr 5e-7 \
    --vision_tower /models/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name llava_next_siglip_qwen2_lora_perpo \
    --output_dir "./checkpoints/llava_next_siglip_qwen2_lora_perpo" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --beta 0.1 \
    --perpo_gamma 0.5 \
    --best_of_n 5

