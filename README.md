# AICUP_Image_Generating

## Required model
- [All required models]()
- Please decompresses the zip file and puts them under this project.

## Run main script
- Please ensure the enviroment has been set up.
- Please download the model above mentioned.
- Ensure project directory follow the format below.
```
|-- main.py
|-- beautifulrealityv3_full.safetensors
|-- road_controlnet
|   |-- [model files]
|-- river_controlnet
|   |-- [model files]
|-- training
|   |-- label_img
|   |   |-- [image files]
|   |-- img
|   |   |-- [image files]
|-- testing
|   |-- label_img
|   |   |-- [image files]
```

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
 --pretrained_model_file="beautifulrealityv3_full.safetensors" \
 --resume_from_checkpoint="checkpoint-19200"
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
