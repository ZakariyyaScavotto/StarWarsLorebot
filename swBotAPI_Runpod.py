from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from swLlamaBot import swLlamaBot
import time
from fastapi.responses import RedirectResponse
import runpod
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize the bot when the API starts
bot = swLlamaBot(loadVecStore=True, vecStorePath="FAISSvectorstore")

# Define request and response models for the API
class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/ready")
async def ready():
    # Check if the model, tokenizer, vector store, and chain are initialized
    if bot.gen_text and bot.tokenizer and bot.vecStore and bot.chain:
        return {"status": "ready"}
    else:
        return {"status": "not ready"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    try:
        # Use the bot's chat method to generate a reply
        reply = bot.chat(request.user_input)
        duration = time.time() - start_time
        logger.info(f"Response time: {duration:.2f} seconds")
        return ChatResponse(reply=reply)
    except Exception as e:
        # Log detailed traceback
        logger.error(f"Error occurred: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.post("/reset-chat")
async def reset_chat_endpoint():
    try:
        # Reset the chat history
        message = bot.reset_chat()
        return {"message": message}
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/docs-link")
async def docs_link():
    # Redirect to the FastAPI docs
    return RedirectResponse(url="/docs")

# Runpod handler functions
def runpod_ready_handler(event):
    return app.router.get("/ready")

def runpod_chat_handler(event):
    try:
        # Validate input
        request = ChatRequest(user_input=event['user_input'])
        return app.router.post("/chat")(request)
    except KeyError as e:
        logger.error(f"Missing key in event: {e}")
        return {"error": f"Missing key: {e}"}, 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": "Internal server error"}, 500

def runpod_reset_chat_handler(event):
    try:
        return app.router.post("/reset-chat")()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": "Internal server error"}, 500

def runpod_docs_link_handler(event):
    return app.router.get("/docs-link")

# Register Runpod handlers
runpod.serverless.register("/ready", runpod_ready_handler)
runpod.serverless.register("/chat", runpod_chat_handler)
runpod.serverless.register("/reset-chat", runpod_reset_chat_handler)
runpod.serverless.register("/docs-link", runpod_docs_link_handler)

# Start the Runpod serverless API
runpod.serverless.start()
