import faiss
import numpy as np
import os
from dotenv import load_dotenv
from OpenAIEmbedding import OpenAIEmbedder

load_dotenv()


class FAISSVectorStore:
    def __init__(self, dimension):
        self.dimentsion = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []
        self.document_ids = []


    def add_documents(self, documents, embeddings, document_ids=None):
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)

        if document_ids is None:
            document_ids = [f"doc_{len(self.documents) - len(documents) + i}"
                            for i in range(len(documents))]
        
        self.document_ids.extend(document_ids)

        print(f"{len(documents)}개 문서가 FAISS 인덱스에 추가되었습니다.")
    

    def search(self, query_embedding, top_k=5):
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if dix < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'document_id': self.document_ids[idx],
                    'index': int(idx)
                })
        
        return results
    

if __name__ == "__main__":
    sample_docs = [
        "딥러닝은 인공신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 머신러닝 기법입니다.",
        "머신러닝은 데이터로부터 패턴을 학습하여 예측이나 분류를 수행하는 인공지능 기술입니다.",
        "자연어처리는 컴퓨터가 인간의 언어를 이해하고 처리할 수 있게 하는 기술 분야입니다.",
        "컴퓨터 비전은 컴퓨터가 이미지나 비디오를 해석하고 이해할 수 있게 하는 기술입니다.",
        "강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 방향으로 학습하는 방법입니다."
    ]

    api_key = os.getenv("OPENAI_API_KEY")
    embedder = OpenAIEmedder(api_key)

    print("문서 임베딩 생성 중...")
    doc_embeddings_list = embedder.get_batch_embeddings(sample_docs)

    if doc_embeddings_list:
        doc_embeddings = np.array(doc_embeddings_list)
        faiss_store = FAISSVectorStore(dimension=doc_embeddings.shape[1])

        faiss_store.add_documents(sample_docs, doc_embeddings)

        print("\n검색 실행 중...")
        query_embedding_list = embedder.get_embedding("딥러닝 학습 방법")

        if query_embedding_list:
            query_embedding = np.array(query_embedding_list)

            fast_results = faiss_store.search(query_embedding, top_k=3)

            print("\nFAISS 빠른 검색 결과:")
            for i, result in enumerate(fast_results, 1):
                print(f"{i}. 점수: {result['score']:.3f}")
                print(f"    문서: {result['document']}")
                print(f"    ID: {result['document_id']}")
                print()
        else:
            print("질문 임베딩 생성에 실패했습니다.")
    else:
        print("문서 임베딩 생성에 실패했습니다.")