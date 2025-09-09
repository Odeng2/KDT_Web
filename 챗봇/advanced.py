# # 의미 기반 청킹 ================================
# import re


# def simple_sentence_split(text):
#     sentences = re.split(r'[.!?]+', text)
#     sentences = [s.strip() for s in sentences if s.strip()]

#     return sentences


# def sementic_chunking(text, max_sentences=3):
#     sentences = simple_sentence_split(text)
#     chunks = []
#     current_chunk = []

#     for sentence in sentences:
#         current_chunk.append(sentence)

#         if len(current_chunk) >= max_sentences:
#             chunks.append('. '.join(current_chunk) + '.')
#             current_chunk = []

#     if current_chunk:
#         chunks.append('. '.join(current_chunk) + '.')

#     return chunks
# sample_text = """
#     머신러닝은 데이터를 기반으로 하는 컴퓨터 시스템의 학습 능력을 개선하는 혁신적인 기술로, 
#     명시적인 프로그래밍 없이도 데이터의 패턴을 찾아내어 예측과 분류 작업을 수행할 수 있습니다.
    
#     전통적인 프로그래밍에서는 개발자가 모든 규칙과 조건을 직접 코딩해야 했지만, 머신러닝은 
#     대량의 데이터로부터 자동으로 규칙을 학습하여 새로운 데이터에 대한 인사이트를 제공합니다.
    
#     머신러닝의 주요 분야로는 지도학습, 비지도학습, 강화학습이 있으며, 각각 다른 접근 방식으로 
#     문제를 해결합니다. 지도학습은 레이블이 있는 데이터를 사용하여 분류나 회귀 작업을 수행하고,
#     비지도학습은 레이블 없는 데이터에서 숨겨진 패턴을 발견하며, 강화학습은 환경과의 상호작용을 
#     통해 최적의 행동 전략을 학습합니다.
    
#     딥러닝은 머신러닝의 한 분야로, 인공 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 
#     기술입니다. 이미지 인식, 자연어 처리, 음성 인식 등의 분야에서 획기적인 성과를 보여주며,
#     현재 AI 기술 발전의 핵심 동력이 되고 있습니다. 특히 트랜스포머 아키텍처의 등장으로 
#     대화형 AI와 생성형 AI 기술이 급속도로 발전하고 있습니다.
#     """

# sementic_chunks = sementic_chunking(sample_text, max_sentences=2)
# for i, chunk in enumerate(sementic_chunks):
#     print(f"의미 청크 {i+1}: {chunk}")








# 컨텍스트 관리 ==============================================
import os
import numpy as np
from openai import OpenAI
import json
from typing import List, Dict, Optional
import dotenv
dotenv.load_dotenv()

from FAISSVectorStore import FAISSVectorStore
from OpenAIEmbedding import OpenAIEmbedder


class RAGPromptBuilder:
    def __init__(self):
        self.system_prompt = """
        당신은 주어진 문서들을 바탕으로 정확하고 도움이 되는 답변을 제공하는 AI 어시스턴트입니다.

        지침:
        1. 주어진 문서 내용만을 바탕으로 답변하세요
        2. 문서에 없는 내용은 추측하지 마세요
        3. 답변에 근거가 되는 문서를 명시하세요
        4. 확실하지 않은 내용은 "문서에서 찾을 수 없습니다"라고 말하세요
        5. 답변은 명확하고 구체적으로 작성하세요
        """
    

    def build_prompt(self, query, retrieved_docs, include_sources=True):
        context = "=== 참고 문서 ===\n"
        for i, doc in enumerate(retrieved_docs, 1):
            if isinstance(doc, dict):
                doc_text = doc.get('document', doc.get('text', str(doc)))
                source = doc.get('source', f'문서 {i}')
            else:
                doc_text = str(doc)
                source = f'문서 {i}'

            context += f"\n[{source}]\n{doc_text}\n"

        prompt = f"""
        {self.system_prompt}

        {context}

        === 질문 ===
        {query}

        === 답변 ===
        위의 참고 문서를 바탕으로 질문에 대한 답변을 작성해주세요.
        """

        return prompt
    

    def build_conversational_prompt(self, query, retrieved_docs, chat_history=None):
        """
        대화형 RAG 프롬프트 생성
        """
        context = "=== 참고 문서 ===\n"
        for i, doc in enumerate(retrieved_docs, 1):
            doc_text = doc.get('document', str(doc)) if isinstance(doc, dict) else str(doc)
            # .get('document', str(doc)) : 딕셔너리에서 'document' 키가 있으면 해당 값을 반환, 없으면 str(doc) 반환
            context += f"\n[문서 {i}]\n{doc_text}\n"
        
        # 대화 기록 추가
        conversation = ""
        if chat_history:
            conversation = "\n=== 이전 대화 ===\n"
            for turn in chat_history[-3:]:  # 최근 3턴만 포함
                conversation += f"사용자: {turn.get('user', '')}\n"
                conversation += f"어시스턴트: {turn.get('assistant', '')}\n\n"
        
        prompt = f"""
        {self.system_prompt}

        {context}
        {conversation}
        === 현재 질문 ===
        {query}

        === 답변 ===
        참고 문서와 이전 대화를 고려하여 답변해주세요.
        """
        
        return prompt


class VectorStore:
    
    def __init__(self, api_key: Optional[str] = None):
        self.embedder = OpenAIEmbedder(api_key or os.getenv('OPENAI_API_KEY'))
        
        self.faiss_store = None
        self.documents_data = []  
    
    def add_documents(self, documents: List[Dict]):
        print("문서 임베딩 생성 중...")
        
        doc_texts = []
        for doc in documents:
            text = doc.get('text', doc.get('document', str(doc)))
            doc_texts.append(text)
        
        embeddings_list = self.embedder.get_batch_embeddings(doc_texts)
        
        if not embeddings_list:
            print("임베딩 생성 실패!")
            return
        
        embeddings = np.array(embeddings_list)
        
        if self.faiss_store is None:
            dimension = embeddings.shape[1]  
            self.faiss_store = FAISSVectorStore(dimension)
            print(f"FAISS 인덱스 초기화 완료 (차원: {dimension})")
        
        self.faiss_store.add_documents(doc_texts, embeddings)
        
        self.documents_data.extend(documents)
        
        print(f"{len(documents)}개 문서가 FAISS 벡터 스토어에 추가되었습니다.")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """FAISS 기반 의미적 검색"""
        print(f"\n검색 쿼리: '{query}'")
        
        if self.faiss_store is None:
            print("FAISS 인덱스가 초기화되지 않았습니다!")
            return []
        
        print(f"총 문서 수: {len(self.documents_data)}")
        
        print("쿼리 임베딩 생성 중...")
        query_embedding_list = self.embedder.get_embedding(query)
        
        if not query_embedding_list:
            print("쿼리 임베딩 생성 실패!")
            return []
        
        query_embedding = np.array(query_embedding_list)
        faiss_results = self.faiss_store.search(query_embedding, top_k=top_k)
        
        print(f"FAISS 검색 완료: {len(faiss_results)}개 결과")
        
        results = []
        for faiss_result in faiss_results:
            doc_index = faiss_result['index']
            
            if doc_index < len(self.documents_data):
                original_doc = self.documents_data[doc_index].copy()
                original_doc['score'] = faiss_result['score']
                results.append(original_doc)
                
                print(f"선택된 문서: {original_doc.get('title', '제목없음')} "
                      f"(유사도: {faiss_result['score']:.3f})")
        
        print(f"반환할 문서 수: {len(results)}")
        return results


class RAGChatbot:
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
        self.prompt_builder = RAGPromptBuilder()
        self.vector_store = VectorStore(api_key)  
        self.chat_history = []
    
    def load_documents(self, documents: List[Dict]):
        self.vector_store.add_documents(documents)
    
    def chat(self, query: str, use_history: bool = True) -> str:

        try:
            print(f"\n사용자 질문: '{query}'")
            retrieved_docs = self.vector_store.search(query, top_k=3)
            
            print(f"검색된 문서 수: {len(retrieved_docs)}")
            
            if not retrieved_docs:
                print("검색된 문서가 없습니다!")
                return "죄송합니다. 관련된 정보를 찾을 수 없습니다."
            
            if use_history and self.chat_history:
                prompt = self.prompt_builder.build_conversational_prompt(
                    query, retrieved_docs, self.chat_history
                )
            else:
                prompt = self.prompt_builder.build_prompt(query, retrieved_docs)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            self.chat_history.append({
                "user": query,
                "assistant": answer,
                "retrieved_docs": len(retrieved_docs)
            })
            
            return answer
            
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"
    
    def get_chat_history(self) -> List[Dict]:
        return self.chat_history
    
    def clear_history(self):
        self.chat_history = []



if __name__ == "__main__":

    sample_documents = [
        {
            'text': '인공지능(AI)은 인간의 지능을 모방하는 컴퓨터 시스템입니다. AI는 학습, 추론, 문제해결 능력을 갖추고 있습니다.',
            'source': 'AI 기초 교재',
            'title': 'AI 개념'
        },
        {
            'text': '머신러닝은 AI의 한 분야로, 데이터로부터 패턴을 학습하여 예측을 수행합니다. 지도학습, 비지도학습, 강화학습으로 나뉩니다.',
            'source': 'ML 가이드북',
            'title': '머신러닝 기초'
        },
        {
            'text': '딥러닝은 인공신경망을 이용한 머신러닝 기법입니다. 이미지 인식, 자연어 처리 등에서 뛰어난 성능을 보입니다.',
            'source': 'DL 논문',
            'title': '딥러닝 개념'
        },
        {
            'text': 'ChatGPT는 OpenAI에서 개발한 대화형 AI 모델입니다. GPT 아키텍처를 기반으로 하며, 자연스러운 대화가 가능합니다.',
            'source': 'OpenAI 블로그',
            'title': 'ChatGPT 소개'
        },
        {
            'text': 'RAG(Retrieval-Augmented Generation)는 외부 지식을 검색하여 답변 생성에 활용하는 기법입니다.',
            'source': 'RAG 연구논문',
            'title': 'RAG 기법'
        }
    ]
    

    # RAG 챗봇 초기화
    chatbot = RAGChatbot()

    chatbot.load_documents(sample_documents)
    
    print("=== RAG 챗봇 테스트 ===\n")
    
    questions = [
        "인공지능이 뭐야?",
        "머신러닝과 딥러닝의 차이는?",
        "RAG가 뭔지 설명해줄래?",
        "ChatGPT에 대해 알려줘"
    ]
    
    for question in questions:
        print(f"질문: {question}")
        answer = chatbot.chat(question)
        print(f"답변: {answer}\n")