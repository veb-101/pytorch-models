import os
import shutil
from sklearn.model_selection import train_test_split
from multiprocessing import Pool


def move_data(src_dir, dst_dir, *args):
    src_dir_images = src_dir["images"]
    src_dir_masks = src_dir["masks"]

    dst_dir_images = dst_dir["images"]
    dst_dir_masks = dst_dir["masks"]

    images, mask = args

    os.makedirs(dst_dir_images, exist_ok=True)
    os.makedirs(dst_dir_masks, exist_ok=True)

    for image_name, mask_name in zip(images, mask):
        image_path = os.path.join(src_dir_images, image_name)
        mask_path = os.path.join(src_dir_masks, mask_name)

        dst_image_path = os.path.join(dst_dir_images, image_name)
        dst_mask_path = os.path.join(dst_dir_masks, mask_name)

        shutil.copyfile(image_path, dst_image_path)
        shutil.copyfile(mask_path, dst_mask_path)

        new_mask_name = mask_name.replace(r"_mask.gif", ".jpg")

        os.rename(
            os.path.join(dst_dir_masks, mask_name),
            os.path.join(dst_dir_masks, new_mask_name),
        )

    print("Done")
    return


if __name__ == "__main__":
    MAIN_DIR = r"dataset"
    IMG_DIR = r"train"
    MASK_DIR = r"train_masks"

    all_images = sorted(os.listdir(os.path.join(MAIN_DIR, IMG_DIR)))
    all_masks = sorted(os.listdir(os.path.join(MAIN_DIR, MASK_DIR)))

    train_image, val_image, train_mask, val_mask = train_test_split(
        all_images, all_masks, shuffle=True, random_state=10, test_size=0.1
    )

    src = {
        "images": os.path.join(MAIN_DIR, IMG_DIR),
        "masks": os.path.join(MAIN_DIR, MASK_DIR),
    }
    dst_train = {
        "images": os.path.join(MAIN_DIR, "data", "train_set", "images"),
        "masks": os.path.join(MAIN_DIR, "data", "train_set", "masks"),
    }
    dst_val = {
        "images": os.path.join(MAIN_DIR, "data", "validation_set", "images"),
        "masks": os.path.join(MAIN_DIR, "data", "validation_set", "masks"),
    }

    half = len(train_mask) // 2
    arguments = [
        (src, dst_train, train_image[:half], train_mask[:half]),
        (src, dst_train, train_image[half:], train_mask[half:]),
        (src, dst_val, val_image, val_mask),
    ]

    pool = Pool(processes=3)
    print("[INFO] processes starting...")
    pool.starmap(move_data, arguments)
    print("[INFO] waiting for processes to finish...")
    pool.close()
    pool.join()
    print("[INFO] multiprocessing complete")
