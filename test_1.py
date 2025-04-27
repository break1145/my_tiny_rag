from datasets import load_dataset
from tqdm import tqdm

from OpenaiModel import OpenAIModel
from VectorStore import PineconeVS
from indexing.embedding import PineconeEmbedding


# https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia?library=datasets

def text_emb():
    # text embedding
    ds_text = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")

    vs = PineconeVS('test-wiki')
    vs.delete_index('test-wiki')
    vs.create_index('test-wiki', 1024)
    emb = PineconeEmbedding(vectorstore=vs)

    vectors = []
    for x in tqdm(ds_text['passages'], desc="embedding"):
        vectors += emb.text_embedding(x['passage'])
    vs.upsert_in_batch(vectors, 50)

def query(q: str):
    # question test
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    model = OpenAIModel()
    q_vc = model.get_one_embedding(q)
    vs = PineconeVS('test-wiki')
    res = vs.query(vector=q_vc, top_k=5, include_metadata=True)
    prompt = f"""
        you are now a useful assistant,answer the question below according to given materials correctly.
        if you can't get relevant information from the given materials, reply "Ciallo, I can't answer the question yet."
        question: {q},  
        materials: {res},
        your answer: 
        """
    print(model.get_response(prompt=prompt))

if __name__ == '__main__':

    text_emb()

    query("Did Lincoln sign the National Banking Act of 1863?")# yes