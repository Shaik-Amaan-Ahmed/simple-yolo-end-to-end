from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI()
model = YOLO("yolov8n.pt")

# Serve static files (like output image)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/movie", response_class=HTMLResponse)
async def upload_form():
    return """
    <html>
        <head>
            <title>Object Detection Upload</title>
        </head>
        <body>
            <h1>Upload Image for Detection</h1>
            <form action="/detect/" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """

@app.post("/detect/", response_class=HTMLResponse)
async def detect(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model(image)
    
    os.makedirs("static", exist_ok=True)
    results[0].save("static/output.jpg")
    
    return """
    <html>
        <head>
            <title>Detection Result</title>
        </head>
        <body>
            <h1>Detection Completed</h1>
            <img src="/static/output.jpg" alt="Detected Image" style="max-width: 600px;">
            <br><br>
            <a href="/movie">Try another image</a>
        </body>
    </html>
    """