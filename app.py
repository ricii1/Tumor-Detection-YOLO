from fastapi import FastAPI, File, UploadFile, Response, status
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["content-type", "accept"],
)

model = YOLO("best.pt")

@app.post("/inference")
async def inference(file: UploadFile):
    if file.content_type != "image/jpeg" and file.content_type != "image/png" and file.content_type != "image/jpg":
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid file format"}
        )
    image_bytes = await file.read()
    img = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    results = model.predict(source=img, conf=0.5)
    for r in results: 
        boxes = r.boxes
    for box in boxes: 
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        label = box.cls[0].item()  
        label_name = model.names[label]  
        (text_width, text_height), baseline = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (255, 0, 255), -1)
        cv2.putText(img, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    resp_img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
    resp_filename = f"inferenced_{file.filename}" if file.filename else "inferenced_image.jpg"
    return StreamingResponse(BytesIO(resp_img_bytes), media_type="image/jpg", headers={"Content-Disposition": f"attachment; filename={resp_filename}"})