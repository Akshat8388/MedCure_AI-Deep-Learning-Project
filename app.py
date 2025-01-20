# Importing Libraries for Backend Logic
from flask import Flask, render_template, request, jsonify #framework for building web applications.

import tensorflow as tf #deep learning library

import numpy as np #numerical operations and handling arrays.

import random #module generating random numbers and making random selections.
from tensorflow.keras.preprocessing import image #Used for loading and processing image data for deep learning models.
from tensorflow.keras.preprocessing.sequence import pad_sequences #Helps in handling sequences of text, like padding sequences for neural networks.

import io # Provides Python's core tools for reading and writing in-memory binary streams.

import matplotlib.pyplot as plt # data visualization library
import matplotlib 
matplotlib.use('Agg')# Matplotlib backend to Agg for non-GUI environments (like servers).
from markdown import markdown

# NLTK (Natural Language Toolkit): A library for working with human language data (text).
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Python module for serializing and deserializing objects (e.g., saving and loading models).
import pickle

import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")

# Initializing the Flask web application
app = Flask(__name__)

cnn_model = tf.keras.models.load_model("Model/chest_xray_model (1).h5")
lstm_model = tf.keras.models.load_model("Model/lstm_model.h5")

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 1000,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
)

with open('Model/label_encoder.pkl','rb') as f:
    le = pickle.load(f)

with open('Model/tokenizer.pkl','rb') as f:
    tokenizer = pickle.load(f)
    

class_labels = ["Covid", "Normal Chest", "Pneumonia", "Tuberculosis", "Glioma", "Meningioma", "Normal Brain", "Pituitary"]


@app.route('/predict_xray', methods=['POST'])
def predict_xray():
    try:
        img = request.files.get('file')
        if not img:
            return jsonify({'error': 'No image provided'}), 400
        
        img_bytes = io.BytesIO(img.read())
        img = image.load_img(img_bytes, target_size=(224, 224))  
        img_arr = image.img_to_array(img)
        img_arr = tf.convert_to_tensor(img_arr, dtype=tf.float32)  
        img_arr = img_arr / 255.0  
        img_arr = tf.expand_dims(img_arr, axis=0)  

        # Predictions
        predictions = cnn_model.predict(img_arr)
        if predictions is None or len(predictions) == 0:
            return jsonify({'error': 'No predictions made by the model'}), 500
        
        
        class_idx = tf.argmax(predictions[0], output_type=tf.int32)
        
        
        class_idx_val = class_idx.numpy()  # Convert the class_idx value to numpy as class_idx is a TensorFlow tensor

        # Model Results
        result = class_labels[class_idx_val]  
        confidence = predictions[0][class_idx_val].item()#.item() to get the scalar value from a NumPy object
    
        # saliency map 
        with tf.GradientTape() as tape:
            tape.watch(img_arr)  # Watch the input image tensor
            class_output = predictions[0][class_idx]  # Ensure class_output is a TensorFlow tensor
            
            if class_output is None:
                return jsonify({'error': 'class_output is None'}), 500  # Check if class_output is valid

            # Convert to tensor if it's not already
            class_output = tf.convert_to_tensor(class_output)

        # Enable gradient tracking for layers
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(img_arr)
            predictions = cnn_model(img_arr)
            class_output = predictions[0][class_idx]

        grads = tape.gradient(class_output, img_arr)

        if grads is None:
            return jsonify({'error': 'Gradients not computed properly'}), 500

        saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
        saliency = tf.maximum(saliency, 0)  
        saliency = saliency / (tf.reduce_max(saliency) + 1e-8)  
        saliency_resized = np.uint8(np.interp(saliency.numpy(), (saliency.numpy().min(), saliency.numpy().max()), (0, 255)))  

    
        plt.imshow(np.array(img).astype("uint8"))  
        plt.imshow(saliency_resized, alpha=0.6, cmap="hot")
        plt.axis('off')

        
        img_path = "static/saliency_map.png"  # Save to the static folder for serving
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        messages = [
               f"Based on the X-ray you provided, it looks like there are signs of '{result}' disease. CNN Model is {confidence*100:.2f}% confident about this diagnosis. However, it is recommended to consult a healthcare professional for a more accurate diagnosis.",
               f"AI Model has analyzed the X-ray and found signs of '{result}' disease. The confidence level for this diagnosis is {confidence*100:.2f}%. Please consult a doctor to confirm the diagnosis and take necessary actions.",
               f"Upon reviewing the X-ray, the AI model predicts '{result}' disease. CNN Model is {confidence*100:.2f}% confident in this diagnosis. It's important to consult a healthcare professional for further examination.",
               f"The analysis suggests signs of '{result}' disease. However, please consult a healthcare professional for a definitive diagnosis and further guidance. Confidence: {confidence*100:.2f}%.",
              ]
        
        chat_session = model.start_chat(history=[])
        
        disease_name_query = f'What is {result} disease and its Symptoms.(explain in short) and start answer without saying (okay).'
        disease_response = chat_session.send_message(disease_name_query)
        ai_disease_response = disease_response.text
        ai_disease_response_html = markdown(ai_disease_response)
        
        treat_name_query = f'What are the treatment for {result} disease and What are the doctor advice for it.(in 5 points (short)) and start answer without saying (okay)'
        treat_response = chat_session.send_message(treat_name_query)
        ai_treat_response = treat_response.text
        ai_treat_response_html = markdown(ai_treat_response)
        
        # Model Result Behaviour
        if result in["Normal Chest","Normal Brain"]:
            selected_message = f"Your {result} X-Ray appears to be normal, with no signs of abnormalities detected. The AI model (CNN_Model) is {confidence*100:.2f}% confident in this result. However, for peace of mind, it's always advisable to consult with a healthcare professional for a thorough check-up."
            return jsonify({'message': selected_message})
        elif result in ["Covid","Pneumonia", "Tuberculosis"]:
            selected_message = random.choice(messages)
            return jsonify({'message': selected_message, 'saliency_map': f"/{img_path}",'disease_ai':ai_disease_response_html,'treatment_ai':ai_treat_response_html})
        else:
            selected_message = random.choice(messages)
            return jsonify({'message': selected_message, 'saliency_map': f"/{img_path}",'disease_ai':ai_disease_response_html,'treatment_ai':ai_treat_response_html})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
   

def preprocessing_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = " ".join(tokens)
    return cleaned_text

@app.route('/disease predict',methods=['POST'])
def disease_predict():
    query = request.form.get('text')
    
    cleaned_query = preprocessing_text(query)
    query_sequence = tokenizer.texts_to_sequences([cleaned_query])
    query_sequence = pad_sequences(query_sequence, maxlen=20, padding='post')
    
    prediction = lstm_model.predict(query_sequence)
    
    predicted_label_index = prediction.argmax(axis=-1)[0]
    confidence_score = prediction[0][predicted_label_index]
    
    predicted_disease = le.inverse_transform([predicted_label_index])
    disease = predicted_disease[0]
    chat_session_lstm = model.start_chat(history=[])
    sym_query = f'A Person feeling {query} and my lstm model predicts {disease}.give general medical medicine names ,treatments and doctor advice.(in 5 points (short)) and start answer without saying (okay).'
    sym_response = chat_session_lstm.send_message(sym_query)
    ai_sym_response = sym_response.text
    ai_sym_response_html = markdown(ai_sym_response)
    
    
    lstm_result = f"The predicted disease is '{disease}' with a confidence of {float(confidence_score*100):.2f}%."
    return jsonify({"disease":lstm_result,"ai_advice":ai_sym_response_html})


@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    