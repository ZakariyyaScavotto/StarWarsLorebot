from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from swLlamaBot import swLlamaBot

app = FastAPI()

# Initialize the bot when the API starts
# Load the vector store only once at the start to save time/resources
bot = swLlamaBot(loadVecStore=True, vecStorePath="FAISSvectorstore")

# Define request and response models for the API
class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Use the bot's chat method to generate a reply
        reply = bot.chat(request.user_input)
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset-chat")
async def reset_chat_endpoint():
    try:
        # Reset the chat history
        message = bot.reset_chat()
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
