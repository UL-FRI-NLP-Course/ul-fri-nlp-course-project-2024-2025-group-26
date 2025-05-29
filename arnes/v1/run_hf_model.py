import os
import json
import random

from transformers import pipeline
import torch

model_id = "cjvt/GaMS-9B-Instruct"

pline = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16, # Use float16 to conserve memory
    device_map="auto"  # Let transformers + accelerate handle GPU
)

# === Setup: Get Input and Output file names ===
input_path = os.getenv("INPUT_FILE")
output_path = os.getenv("OUTPUT_FILE")
if not input_path:
    raise ValueError("⚠️ INPUT_FILE environment variable not set.")
if not output_path:
    raise ValueError("⚠️ OUTPUT_FILE environment variable not set.")

# === Load input examples from file ===
with open(input_path, "r", encoding="utf-8") as f:
    examples = json.load(f)

few_shot_count = 8
num_trials = 32

if len(examples) < few_shot_count + 1:
    raise ValueError("⚠️ Potrebujemo vsaj 9 primerov v datoteki!")

results = []

# === Main loop: Run multiple trials ===
for trial_num in range(num_trials):
    print("=" * 100)
    print(f"🔁 Trial {trial_num + 1}/{num_trials} — Few-shot: {few_shot_count}")

    # Randomly sample few-shot examples; Choose one test example not in few-shot
    selected_examples = random.sample(examples, k=few_shot_count + 1)
    few_shot_examples = selected_examples[:-1]
    test_example = selected_examples[-1]

    # Build the prompt for the model
    prompt_parts = [
        f"Vhod:\n{ex['Input']}\nIzhod:\n{ex['GroundTruth']}\n"
        for ex in few_shot_examples
    ]
    prompt_parts.append(f"Vhod:\n{test_example['Input']}\nIzhod:\n")
    full_prompt = "\n".join(prompt_parts)

    # Format as a chat message for the model
    message = [{"role": "user", "content": full_prompt}]

    # Generate response from model
    with torch.no_grad():
        response = pline(message, max_new_tokens=1024)
    generated_text = response[0]["generated_text"]

    model_output = (
        generated_text[-1]["content"]
        if isinstance(generated_text, list)
        else generated_text
    )

    # Log to console
    print("\n✅ Pričakovani izhod:")
    print(test_example['GroundTruth'].strip())

    print("\n🤖 Modelov izhod:")
    print(model_output.strip())

    # Append result to list
    results.append({
        "ModelOutput": model_output.strip(),
        "GroundTruth": test_example['GroundTruth'].strip()
    })

    print("=" * 100 + "\n")

# === Save all results to a JSON file ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
