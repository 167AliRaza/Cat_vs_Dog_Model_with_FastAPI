
```markdown
# 🐾 Cat vs Dog Image Classifier with TensorFlow & FastAPI

A complete deep learning project that trains a Convolutional Neural Network (CNN) to classify images as **cats or dogs**, and deploys it using a **FastAPI** endpoint for real-time prediction.

---

## 📌 Features

- ✅ Trained on 25,000 labeled images of cats and dogs
- ✅ TensorFlow/Keras-based CNN with multiple Conv2D and MaxPooling layers
- ✅ Preprocessing using `ImageDataGenerator`
- ✅ Binary classification using sigmoid activation
- ✅ 20% of the dataset used for validation
- ✅ FastAPI-based REST API for image classification
- ✅ Model saved as `.h5` for easy deployment and reuse

---

## 🛠️ Tech Stack

- Python 3.10
- TensorFlow / Keras
- FastAPI
- Uvicorn (ASGI server)
- Pillow (for image handling)

---

## 📁 Project Structure

```
cat-dog-classifier/
│
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── validation/
│       ├── cats/
│       └── dogs/
│
├── model/
│   └── cat_dog_model.h5          # Trained model
│
├── main.py                       # FastAPI app
├── cat_vs_dog.py                # Training script
├── pyproject.tolm             # Project dependencies
└── README.md                     # This file
```


## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/cat-dog-classifier.git
cd cat-dog-classifier
```


### 3. Train the model

Make sure your dataset is organized correctly under the `dataset/` folder, then run:

```bash
python train_model.py
```

This will train the model and save it as `cat_dog_model.h5` inside the `model/` directory.

### 4. Run the API

```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` to use the interactive Swagger UI for testing your API.

---

## 🖼️ API Usage

- **Endpoint**: `POST /predict`
- **Form Field**: `file` (image)
- **Returns**: Predicted class (`Cat` or `Dog`) with confidence score

Example using `curl`:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -F 'file=@your_image.jpg'
```

---

## ✨ Example Output

```json
{
  "prediction": "Dog",
  "confidence": 0.976
}
```

---

## 🧠 Model Summary

- **Input size**: 150x150 RGB images
- **Architecture**:
  - 3 Conv2D layers (32, 64, 128 filters)
  - MaxPooling after each Conv
  - Flatten + Dense (512)
  - Dropout (0.5)
  - Output layer with sigmoid activation

---

## 📦 Dependencies

```
tensorflow
fastapi
uvicorn
pillow
```

(Full list in `pyproject.toml`)

---

## 📜 License

This project is licensed under the MIT License. Feel free to use and modify it.

---

## 🙌 Acknowledgements

Inspired by classic cat vs dog classification datasets and TensorFlow tutorials.

---

## 📣 Let's Connect

If you liked this project or have ideas to improve it, feel free to connect or fork the repo! 🤝
```

---

Let me know if you'd like to include setup for Docker or instructions for deploying on a cloud platform like Heroku, Render, or AWS.
