import os
import cv2
import numpy as np

from pathlib import Path

def normalize_image(img):
    # Konversi ke float32 dan normalisasi nilai piksel ke 0-255 (berdasarkan min-max)
    img = img.astype(np.float32)
    img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img_norm.astype(np.uint8)

def process_and_save_images(input_folder, output_folder, img_size=(128, 128)):
    os.makedirs(output_folder, exist_ok=True)

    for label_name in os.listdir(input_folder):
        label_path = os.path.join(input_folder, label_name)
        if not os.path.isdir(label_path):
            continue

        save_label_path = os.path.join(output_folder, label_name)
        os.makedirs(save_label_path, exist_ok=True)

        for file_name in os.listdir(label_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(label_path, file_name)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Gagal membaca: {img_path}")
                    continue

                img = cv2.resize(img, img_size)
                img_norm = normalize_image(img)

                save_path = os.path.join(save_label_path, file_name)
                cv2.imwrite(save_path, img_norm)
                print(f"Disimpan: {save_path}")

if __name__ == "__main__":
    input_dataset_folder = "00000"
    output_normalized_folder = "normalisasi"
    process_and_save_images(input_dataset_folder, output_normalized_folder)
