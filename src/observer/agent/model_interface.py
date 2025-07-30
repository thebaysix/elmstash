from openai import OpenAI
import os

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY")
)

# TODO: Switch to batch in an async flow https://platform.openai.com/docs/guides/batch
def query_model(prompt, model="gpt-4o-mini", temperature=0.7, system_msg=None):
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
