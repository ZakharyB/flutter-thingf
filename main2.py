import cv2
import numpy as np
import time
import os
import urllib.request
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
import threading
import io
from starlette.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="YOLO Object Detection API")

config = {
    "show_fps": False,
    "dark_mode": False,
    "fancy_boxes": False,
    "confidence_threshold": 0.5,
    "hide_labels": False
}

net = None
classes = []
colors = []
detection_running = False
current_fps = 0
detected_objects = []

camera_lock = threading.Lock()
camera = None

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")

def initialize_model():
    global net, classes, colors
    
    os.makedirs('models', exist_ok=True)

    config_path = 'models/yolov3.cfg'
    weights_path = 'models/yolov3.weights'
    classes_path = 'models/coco.names'

    download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", config_path)
    download_file("https://pjreddie.com/media/files/yolov3.weights", weights_path)
    download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", classes_path)

    print("Loading YOLO model...")
    net = cv2.dnn.readNet(weights_path, config_path)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 

    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    np.random.seed(42)  
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    print("Model initialized successfully")

def get_layer_outputs():
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def process_frame(frame):
    global current_fps, detected_objects
    
    start_time = time.time()
    
    height, width, channels = frame.shape
    
    if config["fancy_boxes"]:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (width//2, height//2), (width//2, height//2), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        frame_vignette = frame.copy()
        alpha = 0.85  
        for c in range(0, 3):
            frame[:, :, c] = frame_vignette[:, :, c] * (mask / 255.0 * (1 - alpha) + alpha)
    
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    outs = net.forward(get_layer_outputs())
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > config["confidence_threshold"]:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, config["confidence_threshold"], 0.4)
    
    if config["dark_mode"]:
        bg_color = (40, 40, 40)  
        text_color = (255, 255, 255)  
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
    else:
        bg_color = (255, 255, 255)  
        text_color = (0, 0, 0)  
    
    detected_objects = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            color = colors[class_id]
            
            detected_objects.append({
                "class": classes[class_id],
                "confidence": float(confidence),
                "box": [int(x), int(y), int(w), int(h)]
            })
            
            if config["fancy_boxes"]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y - 20), (x + w, y), color, -1)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            if not config["hide_labels"]:
                label = f"{classes[class_id]}: {confidence:.2f}"
                if config["fancy_boxes"]:
                    cv2.putText(frame, label, (x + 5, y - 5), font, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, label, (x, y - 10), font, 0.5, color, 2)
    
    if config["show_fps"]:
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), font, 0.7, text_color, 2)
    
    processing_time = time.time() - start_time
    current_fps = 1.0 / processing_time if processing_time > 0 else 30.0
    
    return frame

def generate_frames():
    global camera, detection_running
    
    if camera is None:
        with camera_lock:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("Error: Could not open camera.")
                return
    
    detection_running = True
    
    try:
        while detection_running:
            with camera_lock:
                success, frame = camera.read()
            
            if not success:
                break
            
            processed_frame = process_frame(frame)
            
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
    except Exception as e:
        print(f"Error in video stream: {e}")
    finally:
        detection_running = False

def cleanup_camera():
    global camera
    if camera is not None:
        with camera_lock:
            camera.release()
            camera = None

@app.on_event("startup")
async def startup_event():
    initialize_model()

@app.on_event("shutdown")
async def shutdown_event():
    cleanup_camera()

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the YOLO Object Detection API!"}

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    with open("index.html") as f:
        return HTMLResponse(content=f.read())


@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/stop")
def stop_video():
    global detection_running
    detection_running = False
    return {"message": "Video stream stopped"}

@app.get("/fps")
def get_fps():
    return {"fps": current_fps}

@app.get("/detections")
def get_detections():
    return {"detections": detected_objects}

@app.post("/settings")
def update_settings(settings: dict):
    for key, value in settings.items():
        if key in config:
            config[key] = value
    return {"message": "Settings updated", "config": config}

@app.get("/settings")
def get_settings():
    return {"config": config}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)