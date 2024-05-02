# AICUP

run diffuser controlnet training
```sh
accelerate launch ./contrlnet_script/train_controlnet.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --controlnet_model_name_or_path="lllyasviel/sd-controlnet-mlsd" \
 --output_dir="output_sd15" \
 --train_data_dir="training/river" \
 --resolution=512 \
 --checkpointing_steps=2400 \
 --resume_from_checkpoint="output_sd15/checkpoint-3000" \
 --num_train_epochs=20 \
 --train_batch_size=1 \
 --learning_rate=1e-5 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none
```

## pretrained
### base
- runwayml/stable-diffusion-v1-5
- stabilityai/stable-diffusion-2-1
### ControlNet
- lllyasviel/sd-controlnet-mlsd
- thibaud/controlnet-sd21-scribble-diffusers