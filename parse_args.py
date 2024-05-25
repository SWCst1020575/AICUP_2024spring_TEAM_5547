def parse_args():
    # Parse argument.
    # All arguments have default value,
    # so we don't need to pass argument if we follow the directory rules in README.
    parser = argparse.ArgumentParser(
        description="The script to generate target image for AICUP."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the images save.",
    )
    parser.add_argument(
        "--training_dataset",
        type=str,
        default="training",
        help="The training dataset directory where the images save.",
    )
    parser.add_argument(
        "--testing_dataset",
        type=str,
        default="testing",
        help="The testing dataset directory where the images save.",
    )
    parser.add_argument(
        "--road_score",
        type=float,
        default=0.6,
        help="The score to identify which pipeline the label image should get in.",
    )
    parser.add_argument(
        "--river_score",
        type=float,
        default=0.57,
        help="The score to identify which pipeline the label image should get in.",
    )
    parser.add_argument(
        "--road_controlnet_path",
        type=str,
        default="road_controlnet",
        help="The controlnet directory for road where the model saves.",
    )
    parser.add_argument(
        "--river_controlnet_path",
        type=str,
        default="river_controlnet",
        help="The controlnet directory for river where the model saves.",
    )
    parser.add_argument(
        "--base_stable_diffusion_file",
        type=str,
        default="beautifulrealityv3_full.safetensors",
        help="Where the file of the base stable diffusion model saves.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug message.",
    )
    args = parser.parse_args()
    return args
