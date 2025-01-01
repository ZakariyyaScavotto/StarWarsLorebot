from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-13b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("models/Llama-2-13b-chat-hf")
tokenizer.save_pretrained("models/Llama-2-13b-chat-hf")
