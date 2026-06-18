import pandas as pd
import torch
import json
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 1. Configuration & File Paths
INPUT_CSV = "your_input_data.csv"
OUTPUT_CSV = "news_sentiment_results.csv"
CHECKPOINT_DIR = "checkpoints/"
MODEL_ID = "Rexceslum/Llama-3.2-3B-Finance"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 2. 4-Bit Quantization Config (Crucial for 4GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. The Prompt Strategy
system_prompt = """You are a highly accurate financial sentiment analysis AI. 
Analyze the provided financial news text and output the results STRICTLY as a JSON object. 
Do not include any markdown, conversational text, or explanations. 
Use this exact structure:
{
  "llm_positive": <float 0.0-1.0>,
  "llm_negative": <float 0.0-1.0>,
  "llm_neutral": <float 0.0-1.0>,
  "llm_label": "<string: positive, negative, or neutral>",
  "llm_confidence": <float 0.0-1.0>,
  "llm_score": <float -1.0 to 1.0>,
  "llm_relevance": <float 0.0-1.0>,
  "llm_impact_magnitude": <float 0.0-1.0>,
  "llm_uncertainty": <float 0.0-1.0>
}"""


# Robust JSON extraction to handle LLMs that occasionally hallucinate markdown
def extract_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
    # Fallback structure if parsing completely fails
    return {
        "llm_positive": None, "llm_negative": None, "llm_neutral": None,
        "llm_label": "error", "llm_confidence": None, "llm_score": None,
        "llm_relevance": None, "llm_impact_magnitude": None, "llm_uncertainty": None,
        "raw_error_output": text  # Keep the raw text to debug later
    }


# 4. Load Data
df = pd.read_csv(INPUT_CSV)
results = []

print(f"Starting inference on {len(df)} rows...")

# 5. The Inference Loop
for index, row in tqdm(df.iterrows(), total=len(df)):
    news_text = str(row.get('news_text', ''))

    if not news_text.strip():
        results.append(extract_json(""))  # Handle empty rows gracefully
        continue

    # Apply Llama 3 Chat Template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": news_text}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=150,  # Keep short to save time; JSON is brief
            temperature=0.1,  # Low temp for deterministic, consistent JSON
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated portion
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    # Parse and merge with original row data
    parsed_json = extract_json(response)
    combined_row = {**row.to_dict(), **parsed_json}
    results.append(combined_row)

    # 6. Checkpointing (Save every 100 rows)
    if (index + 1) % 100 == 0:
        checkpoint_df = pd.DataFrame(results)
        checkpoint_df.to_csv(f"{CHECKPOINT_DIR}checkpoint_{(index + 1)}.csv", index=False)

# 7. Final Save
final_df = pd.DataFrame(results)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"Complete! Results saved to {OUTPUT_CSV}")