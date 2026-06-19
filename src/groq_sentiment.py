import requests
import os

api_key = ""
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)
print(response.json())

sorted_data = sorted(response.json()["data"], key=lambda x: x['name'])
for model in sorted_data:
    print("-" * 50)
    print(f"model id: {model['id']}")
    print(f"active: {model['active']}")
    print(f"context window: {model['context_window']}")
    print(f"max completion tokens: {model['max_completion_tokens']}")
    if model.get("hugging_face_id"):
        print(f"hugging face id: {model['hugging_face_id']}")
    print(f"input modalities: {model['input_modalities']}")
    print(f"output modalities: {model['output_modalities']}")
    print(f"context length: {model['context_length']}")
    print(f"max output length: {model['max_output_length']}")
    if model.get("pricing"):
        print(f"pricing: {model['pricing']}")
    if model.get("supported_features"):
        print(f"supported features: {model['supported_features']}")