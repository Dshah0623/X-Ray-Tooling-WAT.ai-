from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import shutil
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

phase1_model = torch.load("../models/phase1_model.pth")
file_location = ""
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image_map = {"image/jpeg": ".jpg", "image/png": ".png", "image/bmp": ".bmp", "image/gif": ".gif"}
    file_location = "../assets/" + "1" + image_map[file.content_type]
    with open(file_location, "wb+") as file_object:
        file_object.write(contents)
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}

# @app.post("/phase1")
# async def upload_file(file: UploadFile = File(...)):
#     # Do something with the uploaded file
#     contents = await file.read()
#     # Process the file data here
#     return {"filename": file.filename}
