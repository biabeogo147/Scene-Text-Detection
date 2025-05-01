import os
import random
import subprocess
import cv2
import numpy as np
import yaml
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def extract_data(path):
    tree = ET.parse(path)
    root = tree.getroot()

    im_paths, im_sizes, im_labels, im_bboxes = [], [], [], []
    for im in root:
        bboxes = []
        labels = []

        for bbs in im.findall('taggedRectangles'):
            for bb in bbs:
                if not bb[0].text.isalnum():
                    continue
                if "é" in bb[0].text.lower() or "ñ" in bb[0].text.lower():
                    continue

                bboxes.append(
                    [float(bb.attrib['x']),
                     float(bb.attrib['y']),
                     float(bb.attrib['width']),
                     float(bb.attrib['height'])]
                )
                labels.append(bb[0].text.lower())

        im_paths.append(im[0].text)
        im_sizes.append((int(im[1].attrib['x']), int(im[1].attrib['y'])))
        im_labels.append(labels)
        im_bboxes.append(bboxes)

    return im_paths, im_sizes, im_labels, im_bboxes


def convert_to_yolo_format(im_paths, im_sizes, im_bboxes):
    yolo_data = []
    for im_path, im_size, im_bbox in zip(im_paths, im_sizes, im_bboxes):
        im_width, im_height = im_size
        yolo_labels = []

        for bbox in im_bbox:
            x, y, w, h = bbox # (topleft_x, topleft_y, width, height)
            x_center = (x + w / 2) / im_width
            y_center = (y + h / 2) / im_height
            w /= im_width
            h /= im_height
            class_id = 0

            yolo_labels.append(f'{class_id} {x_center} {y_center} {w} {h}')
        yolo_data.append((im_path, yolo_labels))
    return yolo_data


def save_data(data, src_im_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'labels'), exist_ok=True)

    for im_path, labels in data:
        shutil.copy(os.path.join(src_im_dir, im_path), os.path.join(save_dir, 'images'))
        im_name = os.path.basename(im_path).split('.')[0]
        with open(os.path.join(save_dir, 'labels', f'{im_name}.txt'), 'w') as f:
            for label in labels:
                f.write(f'{label}\n')


def plot_bboxes(im_path, im_labels, im_bboxes):
    im = cv2.imread(im_path)
    if im is None:
        print(f'Error: Could not load image at path {im_path}')
        return
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for label, bbox in zip(im_labels, im_bboxes):
        x, y, w, h = bbox
        cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(im, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    plt.imshow(im)
    plt.tight_layout()
    plt.axis('off')
    plt.show()


def split_bboxes(im_paths, im_labels, im_bboxes, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    labels = []
    for im_path, im_label, im_bbox in zip(im_paths, im_labels, im_bboxes):
        im = Image.open(im_path)
        for label, bbox in zip(im_label, im_bbox):
            cropped_im = im.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

            if np.mean(cropped_im) < 35 or np.mean(cropped_im) > 220:
                continue

            if cropped_im.size[0] < 10 or cropped_im.size[1] < 10:
                continue

            filename = f'{count:06d}.jpg'
            save_path = os.path.join(save_dir, filename)
            cropped_im.save(save_path)

            label = save_path + '\t' + label
            labels.append(label)
            count += 1

    print(f'Created {count} images')
    with open(os.path.join(save_dir, 'labels.txt'), 'w') as f:
        for label in labels:
            f.write(label + '\n')


if __name__ == "__main__":
    subprocess.run(['gdown', '1kUy2tuH-kKBlFCNA0a9sqD2TG4uyvBnV'], check=True)
    subprocess.run(['unzip', '-q', 'icdar2003.zip', '-d', 'datasets'], check=True)

    path = './datasets/SceneTrialTrain/words.xml'
    tree = ET.parse(path)
    root = tree.getroot()

    i = 0
    for im in root:
        print(im[0].text)
        print(im[1].attrib)
        i += 1
        if i == 5:
            break

    for im in root:
        im_name = im[0].text
        for bboxes in im.findall('taggedRectangles'):
            for bbox in bboxes:
                print(bbox[0].text)
                print(bbox.attrib)
        break

    im_paths, im_sizes, im_labels, im_bboxes = extract_data(path)
    print(im_paths[0])
    print(im_sizes[0])
    print(im_labels[0])
    print(im_bboxes[0])

    class_labels = ['text']
    yolo_data = convert_to_yolo_format(im_paths, im_sizes, im_bboxes)

    path = './datasets/SceneTrialTrain/words.xml'
    im_paths, im_sizes, im_labels, im_bboxes = extract_data(path)

    seed = 0
    val_size = 0.2
    test_size = 0.125
    train_data, val_data = train_test_split(yolo_data, test_size=val_size, random_state=seed, shuffle=True)
    train_data, test_data = train_test_split(train_data, test_size=test_size, random_state=seed, shuffle=True)

    data_dir = 'datasets/SceneTrialTrain'
    save_dir = './datasets/yolo_data'
    save_train_dir = os.path.join(save_dir, 'train')
    save_val_dir = os.path.join(save_dir, 'val')
    save_test_dir = os.path.join(save_dir, 'test')

    save_data(train_data, data_dir, save_train_dir)
    save_data(val_data, data_dir, save_val_dir)
    save_data(test_data, data_dir, save_test_dir)

    data_yml = {
        'path': './datasets/yolo_data',
        'train': './datasets/yolo_data/train/images',
        'val': './datasets/yolo_data/val/images',
        'test': './datasets/yolo_data/test/images',
        'nc': 1,
        'names': class_labels
    }
    yolo_yml_path = os.path.join(save_dir, 'data.yml')
    with open(yolo_yml_path, 'w') as f:
        yaml.dump(data_yml, f, default_flow_style=False)

    sample_im = random.randint(0, len(im_paths))
    plot_bboxes(im_paths[sample_im], im_labels[sample_im], im_bboxes[sample_im])

    save_dir = './datasets/ocr_dataset'
    split_bboxes(im_paths, im_labels, im_bboxes, save_dir)
