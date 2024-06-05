
import os
import json
from PIL import Image
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration,BitsAndBytesConfig
import torch
from tqdm import tqdm
import argparse


def parse_args():
    # Parse argument.
    # All arguments have default value,
    # so we don't need to pass argument if we follow the directory rules in README.
    parser = argparse.ArgumentParser(
        description="The script to generate metadata for training."
    )
    parser.add_argument(
        "--training_dataset",
        type=str,
        default="training",
        help="The training dataset directory where the images save.",
    )
    args = parser.parse_args()
    return args


args = parse_args()

img_type = "river"
prefix_path = f"{os.getcwd()}/{args.training_dataset}/"
train_path = f"{args.training_dataset}/"
img_path = f"{args.training_dataset}/img/"
label_img_path = f"{args.training_dataset}/label_img/"
img_list = os.listdir(img_path)
img_list.sort()
label_img_list = os.listdir(label_img_path)
label_img_list.sort()


device = "cuda" if torch.cuda.is_available() else "cpu"
device


river_img_list = [file for file in img_list if "_RI_" in file]
river_label_img_list = [file for file in label_img_list if "_RI_" in file]
road_img_list = [file for file in img_list if "_RO_" in file]
road_label_img_list = [file for file in label_img_list if "_RO_" in file]

print("Model loading...")
base_model = "Salesforce/blip2-opt-6.7b-coco"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
processor = Blip2Processor.from_pretrained(base_model)
model = Blip2ForConditionalGeneration.from_pretrained(
    base_model,
    quantization_config=quantization_config,
)
print("Load model completely.")


with tqdm(total=len(river_img_list)) as pbar:
    for img, label_img in zip(river_img_list, river_label_img_list):
        pbar.update(1)
        img_pil = Image.open(img_path + img)
        new_img_name = img.split(".")
        img_pil.transpose(Image.FLIP_LEFT_RIGHT).save(
            f"{img_path}{new_img_name[0]}_flip.{new_img_name[1]}"
        )
        label_img_pil = Image.open(label_img_path + label_img)
        new_img_name = label_img.split(".")
        label_img_pil.transpose(Image.FLIP_LEFT_RIGHT).save(
            f"{label_img_path}{new_img_name[0]}_flip.{new_img_name[1]}"
        )



# 重新讀檔
img_list = os.listdir(img_path)
img_list.sort()
label_img_list = os.listdir(label_img_path)
label_img_list.sort()
river_img_list = [file for file in img_list if "_RI_" in file]
river_label_img_list = [file for file in label_img_list if "_RI_" in file]

print("生成river metadata")


with tqdm(total=len(river_img_list)) as pbar:
    with open(f"{train_path}/metadata.jsonl", "w") as f:
        for img, label_img in zip(river_img_list, river_label_img_list):
            pbar.update(1)
            img_mat_pil = Image.open(label_img_path + label_img)
            img_mat = np.array(img_mat_pil)
            prompt = "river, "
            for row in img_mat:
                row = row[:, 0]
                if row.max() > 128:
                    if row[0] > 128 or row[-1] > 128:
                        prompt += "lush shore"
                    else:
                        prompt += "rural ,aerial view"
                break
            line = {
                "file_name": f"img/{img}",
                "image": f"{prefix_path}img/{img}",
                "conditioning_image": f"{prefix_path}label_img/{label_img}",
                "text": prompt,
            }
            f.write(json.dumps(line) + "\n")
            line = {
                "file_name": f"label_img/{label_img}",
                "image": f"{prefix_path}img/{img}",
                "conditioning_image": f"{prefix_path}label_img/{label_img}",
                "text": prompt,
            }
            f.write(json.dumps(line) + "\n")



with tqdm(total=len(road_img_list)) as pbar:
        
    for img, label_img in zip(road_img_list, road_label_img_list):
        pbar.update(1)
        
        img_mat_pil = Image.open(label_img_path + label_img)
        img_mat = np.array(img_mat_pil)
        for i in range(img_mat.shape[0]):
            if (img_mat[i,:]>128).sum()>0 :
                top_of_road = i
                break
        for i in range(img_mat.shape[0]-1,-1,-1):
            if (img_mat[i,:]>128).sum()>0 :
                bottom_of_road = i
                break
        if (img_mat[top_of_road,:,0]>128).sum() > (img_mat[bottom_of_road,:,0]>128).sum():
            img_mat_pil.rotate(180).save(label_img_path + label_img)
            img_pil = Image.open(img_path +img)
            img_pil.rotate(180).save(img_path + img)


print("生成road metadata")

with tqdm(total=len(road_img_list)) as pbar:
    with open(f"{train_path}/metadata.jsonl", "a") as f:
        for img, label_img in zip(road_img_list, road_label_img_list):
            pbar.update(1)
            
            img_pil = Image.open(img_path + img)
            inputs = processor(
                            images=img_pil, text="river", return_tensors="pt"
                        ).to(device, torch.float16)
            generated_ids = model.generate(**inputs)
            generated_text = processor.batch_decode(
                            generated_ids, skip_special_tokens=True
                        )[0].strip()
            prompt = "road"
            if "motorcycle" in generated_text:
                prompt+=", motorcycle"
            if "car" in generated_text:
                prompt+=", car"
            if "person" in generated_text:
                prompt+=", person"
            if "people" in generated_text:
                prompt+=", people"
            
            line = {
                "file_name": f"img/{img}",
                "image": f"{prefix_path}img/{img}",
                "conditioning_image": f"{prefix_path}label_img/{label_img}",
                "text": prompt,
            }
            f.write(json.dumps(line) + "\n")
            line = {
                "file_name": f"label_img/{label_img}",
                "image": f"{prefix_path}img/{img}",
                "conditioning_image": f"{prefix_path}label_img/{label_img}",
                "text": prompt,
            }
            f.write(json.dumps(line) + "\n")


