from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()
try:
    model = load_model("cat_dog_model.h5")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image = image.resize((150, 150))
        image_array = np.array(image)
        tensor = np.expand_dims(image_array, axis=0)
        tensor = tensor.astype(np.float32) / 255.0
        prediction = model.predict(tensor)
        class_label = "Dog" if prediction[0][0] > 0.5 else "Cat"
        return {"class_label": class_label, "prediction_score": float(prediction[0][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
