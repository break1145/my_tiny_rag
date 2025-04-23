import json
import os

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

class OpenAIModel:
    def __init__(self):
        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def get_response(self, prompt: str):
        """ 返回一个 raw json str """
        completion = self.client.chat.completions.create(
            model="qwen-plus",
            # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[{"role": "user", "content": f"{prompt}"}],
        )
        return completion.model_dump_json()

    def get_embedding_json(self, text, model="text-embedding-v3"):
        """ 返回一个 raw json str """
        text = text.replace("n", " ")
        return self.client.embeddings.create(input=[text], model=model).model_dump_json()

    def get_one_embedding(self, text) -> list[float]:
        """ 返回单个query的embedding"""
        res = self.client.embeddings.create(model="text-embedding-v3", input=[text])
        return res.data[0].embedding