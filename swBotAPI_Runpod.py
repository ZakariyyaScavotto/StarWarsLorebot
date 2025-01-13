import runpod
from swLlamaBotRunpod import swLlamaBot
import logging
import traceback

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the bot
bot = swLlamaBot(loadVecStore=True, vecStorePath="FAISSvectorstoreIVFPQ")

def handler(event):
    logger.info(f"Received event: {event}")
    try:
        operation = event.get("operation") or event.get("input", {}).get("operation")
        if not operation:
            return {"error": "Missing 'operation' in the request"}
        
        # Map operations to their corresponding functions
        if operation == "ready":
            return ready_handler()
        elif operation == "chat":
            return chat_handler(event.get("input", {}))
        elif operation == "reset-chat":
            return reset_chat_handler()
        else:
            return {"error": f"Unknown operation: {operation}"}
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}\n{traceback.format_exc()}")
        return {"error": "Internal server error"}

# Define each operation's logic as a function
def ready_handler():
    # Check if the bot is initialized and ready
    if bot.gen_text and bot.tokenizer and bot.vecStore and bot.chain:
        return {"status": "ready"}
    else:
        return {"status": "not ready"}

def chat_handler(input_data):
    user_input = input_data.get("user_input")
    chat_history = input_data.get("chat_history", [])
    if not user_input:
        return {"error": "Missing 'user_input'"}
    
    # Convert chat history to the expected format
    formatted_chat_history = [(entry['sender'], entry['text']) for entry in chat_history]
    
    try:
        # Process chat request
        reply = bot.chat(user_input, formatted_chat_history)
        return {"reply": reply, "chat_history": chat_history}
    except Exception as e:
        logger.error(f"Error during chat operation: {str(e)}\n{traceback.format_exc()}")
        return {"error": "Chat operation failed"}

def reset_chat_handler():
    try:
        # Reset chat history
        message = bot.reset_chat()
        return {"message": message}
    except Exception as e:
        logger.error(f"Error during reset operation: {str(e)}\n{traceback.format_exc()}")
        return {"error": "Reset operation failed"}

# Start the RunPod worker with the single handler
runpod.serverless.start({"handler": handler})
