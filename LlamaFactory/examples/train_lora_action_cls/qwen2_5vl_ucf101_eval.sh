export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export HF_HUB_OFFLINE=1

llamafactory-cli train examples/train_lora_action_cls/qwen2_5vl_ucf101_eval.yaml
