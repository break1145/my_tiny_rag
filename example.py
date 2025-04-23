import json

from tqdm import tqdm

from OpenaiModel import OpenAIModel
from VectorStore import PineconeVS
from utils import file_to_list

model = OpenAIModel()

data1 = file_to_list("test_data1.json")

vectors = []
for line in data1[0]:
    emb_obj = json.loads(model.get_embedding_json(line))
    emb = emb_obj['data'][0]
    vectors.append({
        'values': emb['embedding'],
        'id': emb_obj['id'],
        'metadata': {'text': line}
    })

dimension = len(vectors[0]['values'])
index_name = 'xiaomi-15-info-test'

pc = PineconeVS()
index = pc.create_index(index_name=index_name, dimension=dimension)

batch_size = 50
for start in tqdm(range(0, len(vectors), batch_size), "Upserting records batch"):
    batch = vectors[start:start + batch_size]
    index.upsert(vectors=batch)
print(index.describe_index_stats())


query = """小米15用的什么操作系统"""
emb = model.get_one_embedding(query)
res = index.query(vector=emb, top_k=2, include_metadata=True)
print(res)
prompt = f"""
    对于问题：{query} 参考以下资料，给出精简、准确的回答。如果以下资料中没有可以回答问题的相关信息，直接表明资料缺失，不可以胡编乱造。
    资料：{res}.
    输出：
"""
print(model.get_response(prompt=prompt))