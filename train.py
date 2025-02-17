from ultralytics import YOLO
import torch


def train(
    root_dir: str,
    n_epoch=10,
):
    torch.backends.cudnn.enabled = False

    model = YOLO("yolov8n-seg.pt")
    results = model.train(
        data=f"{root_dir}/data/YOLODataset/dataset.yaml",
        epochs=n_epoch,
        imgsz=640,
        degrees=0.5,
        hsv_s=0.7,
    )
    return results

    # validation_results = model.val(
    #     data="./data/YOLODataset/dataset.yaml",
    # imgsz=640, batch=8, conf=0.25, iou=0.6
    # )
