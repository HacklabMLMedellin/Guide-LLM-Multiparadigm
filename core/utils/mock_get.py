from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1/openai", api_key="not-needed")
resp = client.chat.completions.create(
    model="my-model", messages=[{"role": "user", "content": "Hello"}]
)
print(resp.choices[0].message.content)
