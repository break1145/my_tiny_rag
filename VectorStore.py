import os
from typing import Optional

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
class PineconeVS:
    def __init__(self):
        self.index = {}
        self.pc = Pinecone(api_key=pinecone_api_key)

    def create_index(self, index_name: str, dimension: int):
        if not self.pc.has_index(name=index_name):
            # If it does not exist, create index
            self.pc.create_index(
                name=index_name,
                dimension=dimension,  # dimensionality of text-embedding-ada-002
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index[index_name] = self.pc.Index(name=index_name)
        print(self.index[index_name].describe_index_stats())
        return self.pc.Index(name=index_name)