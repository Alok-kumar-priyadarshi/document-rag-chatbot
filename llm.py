from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

model_id = "openai/gpt-oss-120b"

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

def generate_answer(context , question):
    
    prompt = f"""
    
    use the following context to answer the question
    
    context : {context}

    question : {question}

    Answer :
    
    """
    
    response = client.chat.completions.create(
        model = model_id,
        messages=[{"role":"user" , "content": prompt}]
    )
    
    return response.choices[0].message.content
    