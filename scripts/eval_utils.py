import openai
import backoff
import time
import torch
from openai import AzureOpenAI
from openai import RateLimitError
from openai import OpenAI

# we use the engine provided by our university
# you may need to modify the prompting function if you use the openai API
# openai.api_key = ''
# Correct endpoint format (remove the extra path and query parameters)
endpoint = "https://sebas-m88z4ckk-eastus2.cognitiveservices.azure.com/"
deployment = "gpt-35-turbo"
api_version = "2024-12-01-preview"
subscription_key = "QcCAEjEb4AjlMq3HXAqQh6dOUB7Ft6A5sWVz1ODKUU5TsVlCtkaOJQQJ99BCACHYHv6XJ3w3AAAAACOGe30X"

"""
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)
"""

client = OpenAI(
  api_key="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"
)

def prompt_chatgpt(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-35-turbo",  # Using the GPT-3.5 model
            messages=[
                {"role": "user", "content": prompt}
            ],
                max_tokens=10,
                temperature=0.0,
                top_p=1.0
        )
    except Exception as e:
        print(str(e))
        time.sleep(20)
        completion = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
                max_tokens=10,
                temperature=0.0,
                top_p=1.0
        )
    return completion.choices[0].message.content

def prompt_gpt4(prompt):
    try:
        completion = client.chat.completions.create(
            model="GPT-4",  # Using the GPT-4 model
            messages=[
                {"role": "user", "content": prompt}
            ],
                max_tokens=10,
                temperature=0.0,
                top_p=1.0
        )
    except Exception as e:
        print(str(e))
        time.sleep(20)
        completion = client.chat.completions.create(
            model="GPT-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
                max_tokens=10,
                temperature=0.0,
                top_p=1.0
        )
    return completion.choices[0].message.content


def prompt_gpt4o(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            store=True,
            messages=[
                {"role": "user", "content": prompt}
            ],
                    max_tokens=10,
                    temperature=0.0,
                    top_p=1.0
        )
    except Exception as e:
        print(str(e))
        time.sleep(20)
        completion = client.chat.completions.create(
            model="gpt-4o",
            store=True,
            messages=[
                {"role": "user", "content": prompt}
            ],
                    max_tokens=10,
                    temperature=0.0,
                    top_p=1.0
        )
    return completion.choices[0].message.content


@backoff.on_exception(backoff.expo, RateLimitError)
def prompt_chatgpt_with_backoff(prompt):
    return prompt_chatgpt(prompt)

@backoff.on_exception(backoff.expo, RateLimitError)
def prompt_gpt4_with_backoff(prompt):
    return prompt_gpt4(prompt)

@backoff.on_exception(backoff.expo, RateLimitError)
def prompt_gpt4o_with_backoff(prompt):
    return prompt_gpt4o(prompt)


def prompt_llama_like_model(prompt, model, tokenizer, max_new_tokens=100):
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move model to appropriate device
    model = model.to(device)
    
    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"], 
            max_new_tokens=max_new_tokens
        )
        outputs_string = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    return outputs_string