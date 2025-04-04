from transformers import pipeline

model_id = "cjvt/GaMS-9B-Instruct"

pline = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto"  # Let transformers + accelerate handle GPU
)

# Parse input prompt
with open("vhod.txt","r") as f:
    input_string = f.read()

message = [{"role": "user", "content": f"Prosim, izdelaj prometno poročilo za radio. Uporabi podatke v obliki CSV:\n{input_string}"}]
response = pline(message, max_new_tokens=512)
print("Model's response:", response[0]["generated_text"][-1]["content"])