import os
from tqdm import tqdm
from PIL import Image
import torch
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from parser_args import parse_args
from image_processing import image_preprocessing, get_image_construct_list
from typing import Literal
from diffusers import logging

logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_TYPE = Literal["road", "river"]


def get_controlnet_model(args, img_type: IMAGE_TYPE):
    controlnet_path = (
        args.road_controlnet_path if img_type == "road" else args.river_controlnet_path
    )
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, safety_checker=None, torch_dtype=torch.float16
    ).to(device)

    pipe_controlnet = StableDiffusionControlNetPipeline.from_single_file(
        args.base_stable_diffusion_file,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True,
    )
    pipe_controlnet.scheduler = UniPCMultistepScheduler.from_config(
        pipe_controlnet.scheduler.config
    )
    pipe_controlnet.safety_checker = None

    pipe_controlnet.enable_xformers_memory_efficient_attention()

    pipe_controlnet.set_progress_bar_config(disable=True)
    pipe_controlnet = pipe_controlnet.to(device)

    return pipe_controlnet


def get_img2img_model(args):
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_single_file(
        args.base_stable_diffusion_file,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe_img2img.scheduler = UniPCMultistepScheduler.from_config(
        pipe_img2img.scheduler.config
    )
    pipe_img2img.enable_xformers_memory_efficient_attention()

    pipe_img2img.set_progress_bar_config(disable=True)
    pipe_img2img = pipe_img2img.to(device)
    return pipe_img2img


def controlnet_generate(
    origin_img: Image.Image, img_type: IMAGE_TYPE, pipe
) -> Image.Image:
    # generator = torch.manual_seed(0)
    generator = torch.manual_seed(56) if img_type == "river" else torch.manual_seed(17)
    prompt = (
        "river with muddy and light earth color water, aerial view, field, lush shore, gress, masterpiece, best quality, high resolution"
        if img_type == "river"
        else "road with lush grove, masterpiece, best quality, high resolution"
    )
    negative_prompt = (
        "bridge, stone, sand, tall trees, sandbank, worst quality, jpeg artifacts, mutation, duplicate"
        if img_type == "river"
        else "bridge, sand, tall trees, sandbank, worst quality, jpeg artifacts, normal quality, low quality, mutation, duplicate, car, flower"
    )
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=origin_img,
        generator=generator,
        num_inference_steps=30,
        strength=0.5,
        guidance_scale=4.0,
        controlnet_conditioning_scale=1.0,
        height=240,
        width=428,
    ).images
    return images[0].resize((428, 240))


def img2img_generate(
    origin_img: Image.Image, img_type: IMAGE_TYPE, pipe
) -> Image.Image:
    prompt = img_type
    prompt = (
        "river with muddy and light earth color water, aerial view, field, lush shore, gress"
        if img_type == "river"
        else "road with lush grove"
    )
    generator = torch.manual_seed(0)
    negative_prompt = "worst quality, jpeg artifacts, mutation, duplicate"
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=origin_img,
        generator=generator,
        strength=0.1,
        guidance_scale=4.0,
        height=240,
        width=428,
    ).images
    return images[0].resize((428, 240))


def main(args):
    if args.debug:
        print(f"Use score, road: {args.road_score}, river: {args.river_score}")
    print("Image preprocessing...")
    (
        training_image_list_river,
        training_image_list_road,
        testing_image_list_river,
        testing_image_list_road,
    ) = image_preprocessing(args)
    print("-" * 50)
    print("Identifing river image score...")
    river_image_construct_list = get_image_construct_list(
        args=args,
        training_image_list=training_image_list_river,
        testing_image_list=testing_image_list_river,
    )

    print("Identifing road image score...")
    road_image_construct_list = get_image_construct_list(
        args=args,
        training_image_list=training_image_list_road,
        testing_image_list=testing_image_list_road,
    )
    # if args.debug:
    #     return

    print("-" * 50)
    print(f"Device: {device}")
    print("Model loading...")
    controlnet = get_controlnet_model(args=args, img_type="river")
    img2img = get_img2img_model(args=args)
    print("Loading completely.")
    print("-" * 50)
    print("River image generating...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for image_info in tqdm(river_image_construct_list):
        if image_info[0] == "img2img":
            img = img2img_generate(
                origin_img=image_info[2], img_type="river", pipe=img2img
            )
        elif image_info[0] == "controlnet":
            img = controlnet_generate(
                origin_img=image_info[2], img_type="river", pipe=controlnet
            )
        file_name = image_info[1]
        if "_rotate" in file_name:
            img = img.rotate(180)
            file_name = file_name[:-7]
        img.save(f"{args.output_dir}/{file_name}.jpg")
    print("Model loading...")
    controlnet = get_controlnet_model(args=args, img_type="road")
    print("Loading completely.")
    print("-" * 50)
    print("Road image generating...")
    for image_info in tqdm(road_image_construct_list):
        if image_info[0] == "img2img":
            img = img2img_generate(
                origin_img=image_info[2], img_type="road", pipe=img2img
            )
        elif image_info[0] == "controlnet":
            img = controlnet_generate(
                origin_img=image_info[2], img_type="road", pipe=controlnet
            )
        file_name = image_info[1]
        if "_rotate" in file_name:
            img = img.rotate(180)
            file_name = file_name[:-7]
        img.save(f"{args.output_dir}/{file_name}.jpg")
    print("Generate completely.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
