# AICUP_Image_Generating

## Run training script
```sh
accelerate launch ./contrlnet_script/train_controlnet.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --controlnet_model_name_or_path="lllyasviel/sd-controlnet-mlsd" \
 --output_dir="output_sd15_2" \
 --train_data_dir="training/river" \
 --resolution=512 \
 --checkpointing_steps=1200 \
 --num_train_epochs=20 \
 --train_batch_size=1 \
 --learning_rate=1e-5 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --report_to="wandb" \
 --set_grads_to_none \
 --pretrained_model_file="../beautifulrealityv3_full.safetensors" \
 --resume_from_checkpoint="output_sd15/checkpoint-10800"
```
```sh
accelerate launch ./text_to_image_script/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --resolution=512 --random_flip \
  --train_data_dir="training/lora" \
  --train_batch_size=2 \
  --num_train_epochs=100 \
  --checkpointing_steps=4320 \
  --learning_rate=1e-04 --lr_warmup_steps=0 \
  --output_dir="sd15_lora" \
  --use_8bit_adam --enable_xformers_memory_efficient_attention \
  --report_to="wandb" \
  --snr_gamma=5.0 \
  --rank=32 \
  --resume_from_checkpoint="sd15-river-lora/checkpoint-4320" \
  --pretrained_model_file="../beautifulrealityv3_full.safetensors" \
```


## Data Directory Structure for training
```
|-- training
|   |-- river
|   |   |-- train
|   |   |   |-- conditioning_images
|   |   |   |   |-- (mask images)
|   |   |   |-- images
|   |   |   |   |-- (images)
|   |   |   |-- metadata.jsonl
|   |-- road
|   |   |-- train
|   |   |   |-- conditioning_images
|   |   |   |   |-- (mask images)
|   |   |   |-- images
|   |   |   |   |-- (images)
|   |   |   |-- metadata.jsonl
```

## Reference scripts and technique
- [diffusers controlnet training script](https://github.com/huggingface/diffusers/tree/main/examples/controlnet)
- [diffusers lora training script](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)
- [ControlNet](https://github.com/lllyasviel/ControlNet)

## Reference Pretrained Model
### Base
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- ~~[stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)~~

    Not use finally
- [BeautifulRealityV3 (base on sd15)](https://civitai.com/models/389456?modelVersionId=434546)
    
    Improve output quality
### ControlNet
- [lllyasviel/sd-controlnet-mlsd](https://huggingface.co/lllyasviel/sd-controlnet-mlsd)
- ~~[thibaud/controlnet-sd21-scribble-diffusers](https://huggingface.co/thibaud/controlnet-sd21-scribble-diffusers)~~
    
    Not use finally.
