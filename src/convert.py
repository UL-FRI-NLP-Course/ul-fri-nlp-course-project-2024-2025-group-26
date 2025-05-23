import json

def convert_to_text_format(examples):
    """
    Convert a list of examples with 'Input' and 'GroundTruth'
    into a format suitable for fine-tuning.

    Format:
    {
        "text": "Vhod: <Input>\nIzhod: <GroundTruth>"
    }
    """
    formatted = []
    for ex in examples:
        text = f"Vhod: {ex['Input'].strip()}\nIzhod: {ex['GroundTruth'].strip()}"
        formatted.append({"text": text})
    return formatted

def save_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Dataset saved to: {output_path}")

def load_examples(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- File paths ---
input_path = "examples_5_vrstic_sloberta99_for_finetune.json"

output_path = "input_" + input_path

# --- Load and format examples ---
examples = load_examples(input_path)
text_data = convert_to_text_format(examples)

# --- Save dataset ---
save_to_json(text_data, output_path)
