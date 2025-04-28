from datasets import load_dataset
from tqdm import tqdm

from OpenaiModel import OpenAIModel
from VectorStore import PineconeVS
from indexing.embedding import PineconeEmbedding
from retrieval.query import query
from retrieval.search import split_query_with_function_call


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



if __name__ == '__main__':

    # text_emb()

    # query("Did Lincoln sign the National Banking Act of 1863?")# yes
    print(query("丁真的电子烟和王源的传统烟哪个吸了肺痒痒？"))