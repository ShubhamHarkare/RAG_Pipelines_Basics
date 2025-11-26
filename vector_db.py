from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SearchRequest


class QdrantStorage:
    def __init__(self, url='http://localhost:6333', collection='docs', dim=384):
        """
        Initialize Qdrant vector database
        
        Args:
            url: Qdrant server URL
            collection: Collection name
            dim: Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        self.client = QdrantClient(host='localhost', port=6333)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert(self, ids, vectors, payloads):
        """Insert or update vectors in the collection"""
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) 
            for i in range(len(ids))
        ]
        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    def search(self, query_vector, top_k: int = 5):
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding vector (list of floats)
            top_k: Number of results to return
            
        Returns:
            Dictionary with contexts and sources
        """
        # Use query_points instead of search
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        contexts, sources = [], set()
        
        # query_points returns a QueryResponse object with 'points' attribute
        points = results.points if hasattr(results, 'points') else results
        
        for point in points:
            # Access payload from each point
            payload = point.payload if hasattr(point, 'payload') else {}
            text = payload.get("text", "")
            source = payload.get('source', "")

            if text:
                contexts.append(text)
                if source:
                    sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}