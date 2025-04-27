from datasets import load_dataset
from tqdm import tqdm

from OpenaiModel import OpenAIModel
from VectorStore import PineconeVS
from indexing.embedding import PineconeEmbedding
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

def query(q: str):
    """
    Given a question `q`, split it into sub-questions, retrieve matching results from vector store,
    select the top 5 most frequent matches, and generate an answer using LLM based on these materials.
    """

    # Initialize models
    model = OpenAIModel()
    vs = PineconeVS('test-wiki')

    # Step 1: Split the question into multiple queries
    queries = split_query_with_function_call(q)
    print(f"Split queries: {queries}")

    # Step 2: Query the vector store and count match frequencies
    matches = {}
    match_counts = {}

    for query_part in queries:
        embedding = model.get_one_embedding(query_part)
        result = vs.query(vector=embedding, top_k=5, include_metadata=True)

        for match in result['matches']:
            match_id = match['id']
            matches[match_id] = match
            match_counts[match_id] = match_counts.get(match_id, 0) + 1

    # Step 3: Sort matches by frequency and select top 5
    top_match_ids = sorted(match_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    top_matches = [matches[match_id] for match_id, _ in top_match_ids]
    print(f"Top matches: {top_matches}")

    # Step 4: Build prompt for LLM
    prompt = f"""
        You are now a helpful assistant. Answer the following question based on the provided materials.
        If the materials do not contain relevant information, reply: "Ciallo, I can't answer the question yet."
        
        Question: {q}
        
        Materials: {top_matches}
        
        Your Answer:
    """
    # Step 5: Get and print LLM response
    response = model.get_response(prompt=prompt)
    print(response)


if __name__ == '__main__':

    # text_emb()

    query("Did Lincoln sign the National Banking Act of 1863?")# yes