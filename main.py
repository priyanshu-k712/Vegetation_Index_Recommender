import pandas as pd
from langchain_ollama import OllamaLLM
from vector import retriever
from langchain_core.prompts import ChatPromptTemplate
import time
import joblib
from PIL import Image
import numpy as np
from tensorflow import keras

df = pd.read_excel("Data/vegetation_indices_predictions_ALL_IMAGES.xlsx")

model = OllamaLLM(model="gemma3:1b")

chat_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert agricultural advisor. "
     "Use vegetation indices, Analyse them to give practical recommendations."
     "You should give recommendation on agricultural aspect but it should be scientific and technical."
     "Also provide a priority based ranking for your suggestions."),
    ("user",
     "Context from knowledge base:\n{context}\n\n"
     "Vegetation Index values:\nVNDVI:{vndvi}, MGRVI:{mgrvi}, VARI:{vari}\n\n"
     "Give a clear recommendation for farmers.")
])

chain = chat_template | model

def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0  # normalize
    arr = np.expand_dims(arr, axis=0)  # shape becomes (1, H, W, 3)
    return arr


def load_pickle_model(pkl_path):
    model_dict = joblib.load(pkl_path)
    model = keras.models.model_from_json(model_dict["model_json"])
    model.set_weights(model_dict["weights"])
    return model


def get_VIs(path):
    vndvi_model = load_pickle_model(
        "Code/pretrain_data/best_model_vndvi_pickle_20250913_145619.pkl"
    )
    mgrvi_model = load_pickle_model(
        "/home/priyanshu/PycharmProjects/Vegetation_Index_Recommender/mnt/data/best_model_mgrvi_pickle.pkl"
    )
    vari_model = load_pickle_model(
        "/home/priyanshu/PycharmProjects/Vegetation_Index_Recommender/mnt/data/best_model_vari_pickle.pkl"
    )

    print("Models loaded successfully!")



    X_vndvi = preprocess_image(path, (128, 128))  # vndvi expects 128x128
    X_mgrvi = preprocess_image(path, (224, 224))  # mgrvi expects 224x224
    X_vari = preprocess_image(path, (224, 224))  # vari expects 224x224

    vndvi_preds = vndvi_model.predict(X_vndvi)
    mgrvi_preds = mgrvi_model.predict(X_mgrvi)
    vari_preds = vari_model.predict(X_vari)

    return vndvi_preds, mgrvi_preds, vari_preds

image_path = "DATASET/Olive/Multispectral Images/leaf000d0_1.tif"
predicted_vndvi, predicted_mgrvi, predicted_vari = get_VIs(image_path)
query = f"Recommendations for VNDVI={predicted_vndvi}, MGRVI={predicted_mgrvi}, VARI={predicted_vari}"
start = time.time()
docs = retriever.invoke(query)
print("Retriever took:", time.time() - start, "seconds")
context = "\n".join([d.page_content for d in docs])

start = time.time()
result = chain.invoke({
    "context": context,
    "vndvi": predicted_vndvi,
    "mgrvi": predicted_mgrvi,
    "vari": predicted_vari
})
print("LLM took:", time.time() - start, "seconds")

recommendation = result.content if hasattr(result, "content") else str(result)
print(recommendation)
