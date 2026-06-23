import cv2
import os
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

print("--- START: POTRIVIRE IMAGINI CU MASK_ ---")

path_imagini_pos = Path("datasets/images/positive")
path_imagini_neg = Path("datasets/images/negative")
path_masti_pos = Path("datasets/masks/positive")
output_base = Path("yolo_dataset")

if output_base.exists():
    shutil.rmtree(output_base)

for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
    (output_base / folder).mkdir(parents=True, exist_ok=True)

toate_mastile = {}
for m_path in path_masti_pos.glob("*.png"):
 
    cifre = "".join(filter(str.isdigit, m_path.stem))
    if cifre:
        toate_mastile[cifre] = m_path

def genereaza_yolo_label(mask_path, img_w, img_h):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None: return []
    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 5: continue
        x, y, w, h = cv2.boundingRect(cnt)
        labels.append(f"0 {(x + w/2) / img_w:.6f} {(y + h/2) / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}")
    return labels


date_totale = []
masti_gasite = 0
pos_images = list(path_imagini_pos.glob("*.png"))

for img_p in pos_images:
    cifre_img = "".join(filter(str.isdigit, img_p.stem))
    m_p = toate_mastile.get(cifre_img)
    
    if m_p:
        date_totale.append((img_p, m_p, True))
        masti_gasite += 1

for img_p in path_imagini_neg.glob("*.png"):
    date_totale.append((img_p, None, False))

print(f"SUCCES: Am potrivit {masti_gasite} masti din {len(pos_images)} imagini pozitive!")

train_list, val_list = train_test_split(date_totale, test_size=0.2, random_state=42)

def proceseaza(lista, tip):
    count = 0
    for img_p, mask_p, is_pos in lista:
        shutil.copy(img_p, output_base / "images" / tip / img_p.name)
        label_file = output_base / "labels" / tip / (img_p.stem + ".txt")
        if is_pos and mask_p:
            img = cv2.imread(str(img_p))
            if img is not None:
                h, w = img.shape[:2]
                yolo_data = genereaza_yolo_label(mask_p, w, h)
                if yolo_data:
                    with open(label_file, "w") as f:
                        f.write("\n".join(yolo_data))
                    count += 1
        else:
            open(label_file, "w").close()
    print(f"Gata {tip}: {count} etichete.")

proceseaza(train_list, "train")
proceseaza(val_list, "val")