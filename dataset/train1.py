from ultralytics import YOLO

# Load a model
model = YOLO(r"/Users/meghrajdesai11/Desktop/SEM_5/SGP/Drones/yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data=r"/Users/meghrajdesai11/Desktop/SEM_5/SGP/Drones/data.yaml", epochs=1, imgsz=640, device="mps" , batch=8)