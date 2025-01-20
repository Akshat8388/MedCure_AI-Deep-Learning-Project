# AIzaSyCAa0U_hQ-yJ81bskvOMCZUefu8rY0UG74
import google.generativeai as genai


# Configure the API key
genai.configure(api_key="AIzaSyCAa0U_hQ-yJ81bskvOMCZUefu8rY0UG74")  # Replace <YOUR_API_KEY> with your actual API key

# Create the generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1000,  # Set to 100 for short and on-point responses
    "response_mime_type": "text/plain",
}

# Load the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Start a chat session
chat_session = model.start_chat(history=[])

# Query for a short response
q = (
    "A Person is Diagnosed with Pneumonia disease. "
    "Suggest a treatment, necessary precautions, and follow-up steps in short. "
    "Answer on point and start with: 'Here is the treatment, precaution etc...'."
)

# Send the query
response = chat_session.send_message(q)

print(response.text)


  
  
  
  
  
  
  
  
  
  
  