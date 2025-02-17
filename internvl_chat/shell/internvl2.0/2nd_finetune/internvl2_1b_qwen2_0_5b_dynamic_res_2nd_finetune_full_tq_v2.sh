set -x

N_NODES=${N_NODES:-4}
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/internvl_chat_v2_0/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full_v2'
LOGGING_DIR='work_dirs/internvl_chat_v2_0/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full_v2/tf_logs'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 30
torchrun \
  --nnodes=${N_NODES} \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "/mnt/zj-gpfs/home/zsb/InternVL/internvl_chat/shell/pretrained/InternVL2-1B/" \
  --conv_style "Hermes-2" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/internvl_1_2_finetune_table_tq_v2.json" \
  --overwrite_output_dir False \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 30 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_steps 200 \
  --save_total_limit 30 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  --logging_dir ${LOGGING_DIR} \
  2>&1 | tee -a "${LOGGING_DIR}/training_log.txt"
