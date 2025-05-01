import os
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO('yolo11m.pt')
    save_dir = '../datasets/yolo_data'
    yolo_yml_path = os.path.join(save_dir, 'data.yml')

    results = model.train(
        data=yolo_yml_path,
        epochs=100,
        imgsz=640,
        cache=True,
        patience=20,
        plots=True,
        project='../models/yolo11m',
        name='yolo11m_640_100epochs',
    )
