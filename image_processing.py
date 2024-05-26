import numpy as np
from tqdm import tqdm
from PIL import Image
import os


def check_label_image_rotate(img: Image.Image) -> bool:
    img_mat = np.array(img)
    for i in range(img_mat.shape[0]):
        if (img_mat[i, :] > 128).sum() > 0:
            top_of_road = i
            break
    for i in range(img_mat.shape[0] - 1, -1, -1):
        if (img_mat[i, :] > 128).sum() > 0:
            bottom_of_road = i
            break
    if (img_mat[top_of_road, :, 0] > 128).sum() > (
        img_mat[bottom_of_road, :, 0] > 128
    ).sum():
        return True
    return False


def get_image_pipeline_type(
    args, img_info: (str, Image.Image), training_image_list: list
):
    # According to the score measure formula,
    # define each image in testing data should belong to which pipeline.
    img = np.array(img_info[1])[:, :, 0]
    b = img > 128
    img_total = b.sum()
    SCORE = args.river_score if "_RI_" in img_info[0] else args.road_score
    score = 0
    pos = 0
    for i in range(len(training_image_list)):
        a = training_image_list[i][1] > 128
        score_current = (np.logical_and(a, b)).sum() / img_total
        if score_current > score:
            score = score_current
            pos = i
    if score > SCORE:
        img_target = training_image_list[pos][2]
        construct_type = "img2img"
    else:
        img_target = img_info[1]
        construct_type = "controlnet"
    return (construct_type, img_info[0], img_target)


def get_borden_line_label_image(img: Image.Image) -> np.ndarray:
    img = img.resize((107, 60)).resize((428, 240), Image.BILINEAR)
    img = (np.array(img)[:, :, 0] > 0).astype(np.uint8) * 255
    return img


def image_preprocessing(args):
    # Since to searching performance, we seperate the data to river and road.
    training_dataset_path = args.training_dataset
    testing_dataset_path = args.testing_dataset
    training_label_files = os.listdir(f"{training_dataset_path}/label_img")
    test_label_files = os.listdir(f"{testing_dataset_path}/label_img")
    training_image_list_river = []
    training_image_list_road = []
    testing_image_list_river = []
    testing_image_list_road = []
    for file in tqdm(training_label_files):
        # Make width of line larger

        label_img_pil = Image.open(f"{training_dataset_path}/label_img/{file}")
        label_img = get_borden_line_label_image(label_img_pil)
        label_img_rotate = get_borden_line_label_image(label_img_pil.rotate(180))

        name_of_file = file.split(".")[0]
        img = Image.open(f"{training_dataset_path}/img/{name_of_file}.jpg")
        if "_RI_" in name_of_file:
            training_image_list_river.append(
                [
                    f"{name_of_file}_rotate",
                    label_img_rotate,
                    img.rotate(180),
                ]
            )
            training_image_list_river.append((name_of_file, label_img, img))
        elif "_RO_" in name_of_file:
            training_image_list_road.append(
                [
                    f"{name_of_file}_rotate",
                    label_img_rotate,
                    img.rotate(180),
                ]
            )
            training_image_list_road.append((name_of_file, label_img, img))
    for file in tqdm(test_label_files):
        label_img = Image.open(f"{testing_dataset_path}/label_img/{file}")
        name_of_file = file.split(".")[0]
        if "_RO_" in file:
            if check_label_image_rotate(label_img):
                testing_image_list_road.append(
                    (f"{name_of_file}_rotate", label_img.rotate(180))
                )
            else:
                testing_image_list_road.append((name_of_file, label_img))
        elif "_RI_" in file:
            testing_image_list_river.append((name_of_file, label_img))
    return (
        training_image_list_river,
        training_image_list_road,
        testing_image_list_river,
        testing_image_list_road,
    )


def get_image_construct_list(args, training_image_list, testing_image_list):
    # Return a list of images and their info for testing data.
    image_construct_list = []

    for img_info in tqdm(testing_image_list):
        image_construct_list.append(
            get_image_pipeline_type(
                args=args, img_info=img_info, training_image_list=training_image_list
            )
        )
    if args.debug:
        img2img_count = 0
        for info in image_construct_list:
            if info[0] == "img2img":
                img2img_count += 1
        print("Pipeline result:")
        print(
            f"img2img: {img2img_count}, controlNet: {len(image_construct_list) - img2img_count}\n"
        )
    return image_construct_list
