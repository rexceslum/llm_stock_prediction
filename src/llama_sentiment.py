import pandas as pd
import torch
import json
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Increase timeout to 300 seconds (5 minutes) instead of the default 10 seconds
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = "300"

# 1. Configuration & File Paths
TICKER = "NVDA"
INPUT_CSV = f"../data/cleaned_merged_{TICKER.lower()}_news.csv"
OUTPUT_CSV = f"../data/llm_{TICKER.lower()}_news_sentiment.csv"
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
system_prompt = f"""You are a highly accurate financial sentiment analysis AI. 
Analyze the provided financial news text from the perspective of {TICKER} stock. Respond STRICTLY with a JSON object. 
Do not include any other text. 
Use this exact structure:
{{
  "llm_label": "<string: positive, negative, or neutral>",
  "llm_score": <integer from -5 (very negative) to 5 (very positive)>,
  "llm_confidence": <integer from 1 (unsure) to 10 (highly confident)>
}}"""


# Robust JSON extraction to handle LLMs that occasionally hallucinate markdown
def extract_json(text):
    # Default fallback if the LLM completely fails
    data = {
        "llm_label": "error", "llm_score": 0, "llm_confidence": 0
    }

    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            # Merge parsed data into our default dictionary
            data.update(parsed)
    except Exception:
        pass  # If parsing fails, we stick with the default 'error' data

    # 1. Normalize the core metrics back to your desired float scales
    label = str(data.get("llm_label", "neutral")).lower().strip()
    raw_score = float(data.get("llm_score", 0))  # Expected -5 to 5
    raw_conf = float(data.get("llm_confidence", 5))  # Expected 1 to 10

    # Scale score to -1.0 to 1.0
    normalized_score = round(max(min(raw_score / 5.0, 1.0), -1.0), 2)

    # Scale confidence to 0.0 to 1.0
    normalized_conf = round(max(min(raw_conf / 10.0, 1.0), 0.0), 2)

    # 2. Mathematically derive the positive/negative/neutral probabilities based on the score
    pos_prob = 0.0
    neg_prob = 0.0
    neu_prob = 0.0

    if normalized_score > 0.2:
        pos_prob = abs(normalized_score)
        neu_prob = round(1.0 - pos_prob, 2)
    elif normalized_score < -0.2:
        neg_prob = abs(normalized_score)
        neu_prob = round(1.0 - neg_prob, 2)
    else:
        neu_prob = 1.0 - abs(normalized_score)

    # 3. Build the final 9-metric dictionary you originally wanted
    return {
        "llm_positive": round(pos_prob, 2),
        "llm_negative": round(neg_prob, 2),
        "llm_neutral": round(neu_prob, 2),
        "llm_label": label if label in ['positive', 'negative', 'neutral'] else "neutral",
        "llm_confidence": normalized_conf,
        "llm_score": normalized_score,
        "llm_relevance": 0.9,  # Hardcoded: if it contains the ticker, it's highly relevant
        "llm_impact_magnitude": round(abs(normalized_score), 2),  # Impact is the absolute value of the score
        "llm_uncertainty": round(1.0 - normalized_conf, 2)  # Uncertainty is the inverse of confidence
    }

# 4. Load Data
df = pd.read_csv(INPUT_CSV).head(10)
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

    # Get the formatted string first (tokenize=False)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize explicitly into a dictionary
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Keep short to save time; JSON is brief
            temperature=0.3,  # Low temp for deterministic, consistent JSON
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Slice the output using the input_ids length explicitly
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

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