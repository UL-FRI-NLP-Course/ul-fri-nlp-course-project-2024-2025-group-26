from transformers import pipeline
import torch

model_id = "cjvt/GaMS-9B-Instruct"

pline = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16, # Use float16 to conserve memory
    device_map="auto"  # Let transformers + accelerate handle GPU
)

# Read CSV input
with open("vhod.txt", "r") as f:
    input_string = f.read()

# Read formatting instructions
with open("navodila.txt", "r") as f:
    formatting = f.read()
with open("navodila_osnove.txt", "r") as f:
    formatting_basic = f.read()

# Combine all parts into one prompt
full_prompt = (
    "Prosim, izdelaj prometno poročilo za radio. "
    "Uporabi podatke v obliki CSV, ki so podani spodaj. "
    "CSV podatki:\n"
    f"{input_string}\n\n"
    "Upoštevaj naslednja navodila za oblikovanje besedila:\n\n"
    f"{formatting_basic}\n\n"
    "Dodatna navodila:\n"
    f"{formatting}\n\n"
    "Prosim, ne povzemaj navodil. Potrebujem samo pravilno formatirano poročilo."
    "Bodi kratek. Poročila so tipično krajša od 250 besed."
)

# Wrap the full prompt as a user message
message = [{"role": "user", "content": full_prompt}]

# Generate response
# response = pline(message, max_new_tokens=512)
response = pline(message, max_new_tokens=1024)
print("Model's response:", response[0]["generated_text"][-1]["content"])
