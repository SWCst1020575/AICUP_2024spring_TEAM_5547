# AICUP_2024spring_Image_Generating

## Requirement
- [All required models](https://drive.google.com/file/d/1zyBKI_85ka__xBK7qdURfYC3nkcM7kGm/view)
- 請下載後解壓縮到此專案底下
- 安裝套件
```sh
pip install pillow numpy tqdm torch==2.3.0 torchvision==0.18.0
pip install diffusers["torch"] transformers xformers
pip install bitsandbytes datasets
```
or
```sh
pip install -r requirements.txt
```

## Run inference script

```sh
python main.py
```
or
```sh
python main.py \
 --road_controlnet_path="road_controlnet" \
 --river_controlnet_path="river_controlnet" \
 --training_dataset="training" \
 --testing_dataset="testing" \
 --base_stable_diffusion_file="beautifulrealityv3_full.safetensors" \
 --output_dir="output"
```
- 如果有把dataset保持下述資料夾結構，則可直接運行不用傳參數
- 請確保上述套件安裝完成
- 請確保上述模型下載到此資料夾
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
- 上述提供的模型是river和road分開，運行以下腳本只會生成一個合併的版本
- 訓練過程若使用GPU，VRAM至少需12G。
- 首先是生成metadata，training dataset一樣要確保上述資料夾結構。因為有使用blip model生成prompt所以比較慢。
```sh
python metadata.ipynb
```
- 接著是訓練controlnet模型，我上方所提供的model是10個epoch
- 訓練完後將 `output_model/checkpoint-*/controlnet` 的資料夾路徑，放到main.py的 `--road_controlnet_path`, `--river_controlnet_path`參數即可運行生成
```sh
accelerate launch ./controlnet_script/train_controlnet.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --controlnet_model_name_or_path="lllyasviel/sd-controlnet-mlsd" \
 --output_dir="output_model" \
 --train_data_dir="training" \
 --resolution=512 \
 --checkpointing_steps=3240 \
 --num_train_epochs=10 \
 --train_batch_size=1 \
 --learning_rate=1e-5 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --pretrained_model_file="beautifulrealityv3_full.safetensors"
```

## Reference scripts and technique
- [diffusers controlnet training script](https://github.com/huggingface/diffusers/tree/main/examples/controlnet)
- [ControlNet](https://github.com/lllyasviel/ControlNet)

## Reference Pretrained Model
### Base
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [BeautifulRealityV3 (base on sd15)](https://civitai.com/models/389456?modelVersionId=434546)
    
    Improve output quality
### ControlNet
- [lllyasviel/sd-controlnet-mlsd](https://huggingface.co/lllyasviel/sd-controlnet-mlsd)

