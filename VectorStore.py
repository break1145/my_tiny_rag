import os
from typing import Optional

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
class PineconeVS:
    def __init__(self, index_name):
        self.index = None
        self.index_name = index_name
        self.pc = Pinecone(api_key=pinecone_api_key)

    def create_index(self, index_name: str, dimension: int):
        if not self.pc.has_index(name=index_name):
            # If it does not exist, create index
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index = self.pc.Index(name=index_name)
        print(self.index.describe_index_stats())
        return self.pc.Index(name=index_name)

    def upsert_in_batch(self, vectors, batch_size):
        for start in tqdm(range(0, len(vectors), batch_size), "Upserting records batch"):
            batch = vectors[start:start + batch_size]
            self.index.upsert(vectors=batch)

    def query(self, vector, top_k=2, include_metadata=True):
        return self.index.query(vector=vector, top_k=top_k, include_metadata=include_metadata)