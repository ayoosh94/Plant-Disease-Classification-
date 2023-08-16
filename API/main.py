from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import pickle
import cv2
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


with open('C:\\Users\\jash\\support_vector.pkl', 'rb') as f:
    SVM_model = pickle.load(f)
with open('C:\\Users\\jash\\random_forest.pkl', 'rb') as f:
    Random_model = pickle.load(f)
with open('C:\\Users\\jash\\gradient_boosting.pkl', 'rb') as f:
    Gradient_model = pickle.load(f)
with open('C:\\Users\\jash\\stacked_model.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)

CLASS_NAMES = ["Healthy","Early Blight", "Late Blight" ]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.get("/")
async def ping():
    return "Hello world"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    
    image = read_file_as_image(await file.read())
    print(type(image))
    newimg=image[:,:,0]
    print(newimg.shape)

    potato_image = cv2.resize(newimg, (224, 224))  # Resize the original image
    # img = potato_image.flatten().reshape(1,-1) 
    # img = flattened_image.reshape(1, -1) 
    img_batch =potato_image.flatten().reshape(1,-1)
    print(img_batch.shape)
    svm = SVM_model.predict(img_batch)
    random = Random_model.predict(img_batch)
    gradient = Gradient_model.predict(img_batch)

    pred_features=np.column_stack((svm,random,gradient))
    # predictions=ensemble_model.predict(pred_features)
    # print(predictions)
    # print(predictions[0])
    # predicted_class = CLASS_NAMES[predictions[0]]
    # print(predicted_class)
    # confidence = np.max(ensemble_model_proba[0])
    ensemble_prediction_proba = ensemble_model.predict_proba(pred_features)  # Use predict_proba instead of predict
    predicted_class_index = np.argmax(ensemble_prediction_proba[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = np.max(ensemble_prediction_proba[0])
    return {
        'class': predicted_class,
        'confidence': confidence
    }
    
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)