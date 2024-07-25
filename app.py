import sys
from src.logger import logging
from src.exception import CustomException
from src.pipeline.predict import Predict

from fastapi import FastAPI, UploadFile


from io import BytesIO
from PIL import Image

app = FastAPI(title = "MINST")
predict = Predict()
@app.get("/")
async def root():
    return {"Status":"OK"}


@app.post("/recognise")
async def create_upload_file(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        logging.info("API SERVER : Loaded Image.")
        number = predict(image)
        return {"number": number}
    
    except Exception as e:
        logging.info(f"Caught Exception : {e}")
        raise CustomException(e,sys)