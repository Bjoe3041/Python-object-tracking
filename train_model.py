from ultralytics import YOLO

model = YOLO("Models//yolov8s-oiv7.pt") # input with pretrained weights

if __name__ == '__main__':
    results = model.train(
        data="combination_dataset/data.yaml",
        epochs=10,
        imgsz=640, # img input size
        batch=16,
        workers=4,
        augment=True, ##Augments below here
        mosaic=1.0,       # combines 4 images
        mixup=0.1,
        degrees=15,       # rotation
        flipud=0.3,
        fliplr=0.5,
        hsv_s=0.5,        # saturation shift
        hsv_v=0.4,        
        iou=0.4,
    )

    # best.pt is the best-performing checkpoint, last.pt is the final epoch
    model_path = "runs/detect/train/weights/best.pt"
    print(f"Training complete. Model saved to {model_path}")
