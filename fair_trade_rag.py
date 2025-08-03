import json
import os
from typing import List, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

class FairTradeRAG:
    """
    공정거래 법령 분석을 위한 RAG 시스템
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        RAG 시스템 초기화
        
        Args:
            openai_api_key (str): OpenAI API 키 (환경변수에서 자동으로 가져옴)
        """
        # OpenAI API 키 설정
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # 임베딩 모델 초기화 (한국어 지원)
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1
        )
        
        # 텍스트 분할기
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # 컬렉션 이름
        self.collection_name = "fair_trade_laws"
        
    def load_law_data(self, filename: str = "fair_trade_laws.json") -> List[Dict]:
        """
        법령 데이터를 파일에서 로드합니다.
        
        Args:
            filename (str): 법령 데이터 파일명
            
        Returns:
            List[Dict]: 로드된 법령 데이터
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"파일 {filename}을 찾을 수 없습니다.")
            return []
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {e}")
            return []
    
    def prepare_documents(self, laws: List[Dict]) -> List[Dict]:
        """
        법령 데이터를 RAG용 문서로 변환합니다.
        
        Args:
            laws (List[Dict]): 법령 데이터
            
        Returns:
            List[Dict]: RAG용 문서 목록
        """
        documents = []
        
        for law in laws:
            # 법령 전체 내용
            if law.get('content'):
                documents.append({
                    'text': f"법령명: {law['title']}\n\n내용: {law['content']}",
                    'metadata': {
                        'type': 'law_full',
                        'title': law['title'],
                        'keyword': law.get('keyword', ''),
                        'url': law.get('url', '')
                    }
                })
            
            # 조문별 분리
            for article in law.get('articles', []):
                article_text = f"법령명: {law['title']}\n조문: 제{article['number']}조"
                if article.get('title'):
                    article_text += f" ({article['title']})"
                article_text += f"\n\n내용: {article['content']}"
                
                documents.append({
                    'text': article_text,
                    'metadata': {
                        'type': 'article',
                        'law_title': law['title'],
                        'article_number': article['number'],
                        'article_title': article.get('title', ''),
                        'keyword': law.get('keyword', ''),
                        'url': law.get('url', '')
                    }
                })
        
        return documents
    
    def create_vector_database(self, documents: List[Dict]):
        """
        문서들을 벡터 데이터베이스에 저장합니다.
        
        Args:
            documents (List[Dict]): 저장할 문서 목록
        """
        # 기존 컬렉션이 있으면 삭제
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        # 새 컬렉션 생성
        collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 문서들을 청크로 분할
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for i, doc in enumerate(documents):
            # 텍스트 분할
            chunks = self.text_splitter.split_text(doc['text'])
            
            for j, chunk in enumerate(chunks):
                chunk_id = f"doc_{i}_chunk_{j}"
                all_chunks.append(chunk)
                all_metadatas.append(doc['metadata'])
                all_ids.append(chunk_id)
        
        # 벡터 임베딩 생성 및 저장
        embeddings = self.embedding_model.encode(all_chunks)
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        print(f"벡터 데이터베이스에 {len(all_chunks)}개의 청크를 저장했습니다.")
    
    def search_relevant_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        쿼리와 관련된 문서들을 검색합니다.
        
        Args:
            query (str): 검색 쿼리
            n_results (int): 반환할 결과 수
            
        Returns:
            List[Dict]: 관련 문서 목록
        """
        collection = self.client.get_collection(self.collection_name)
        
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_model.encode([query])
        
        # 유사도 검색
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        documents = []
        for i in range(len(results['documents'][0])):
            documents.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return documents
    
    def analyze_case(self, case_description: str) -> str:
        """
        케이스 분석을 수행합니다.
        
        Args:
            case_description (str): 분석할 케이스 설명
            
        Returns:
            str: 분석 결과
        """
        # 관련 법령 검색
        relevant_docs = self.search_relevant_documents(case_description, n_results=8)
        
        # 컨텍스트 구성
        context = "관련 법령 정보:\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"{i}. {doc['text']}\n\n"
        
        # 시스템 프롬프트
        system_prompt = """당신은 공정거래 전문 변호사입니다. 
주어진 케이스와 관련 법령을 바탕으로 다음과 같이 분석해주세요:

1. **관련 법령 식별**: 케이스와 가장 관련성이 높은 법령과 조문을 찾아주세요.
2. **법적 쟁점 분석**: 케이스에서 발생할 수 있는 법적 쟁점을 분석해주세요.
3. **위반 가능성 평가**: 공정거래법 위반 가능성을 평가해주세요.
4. **권고사항**: 기업이 취해야 할 조치사항을 제시해주세요.

분석은 객관적이고 실무적으로 작성해주세요."""

        # 사용자 메시지
        user_message = f"""케이스 설명:
{case_description}

{context}

위 케이스를 분석해주세요."""

        # LLM 호출
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def get_law_summary(self, law_name: str = None) -> str:
        """
        특정 법령 또는 전체 법령 요약을 제공합니다.
        
        Args:
            law_name (str): 특정 법령명 (None이면 전체 요약)
            
        Returns:
            str: 법령 요약
        """
        if law_name:
            # 특정 법령 검색
            relevant_docs = self.search_relevant_documents(law_name, n_results=10)
        else:
            # 전체 법령 요약
            relevant_docs = self.search_relevant_documents("공정거래 하도급 상생협력", n_results=15)
        
        context = "법령 정보:\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"{i}. {doc['text']}\n\n"
        
        system_prompt = """당신은 법령 전문가입니다. 
주어진 법령 정보를 바탕으로 다음과 같이 요약해주세요:

1. **법령 개요**: 각 법령의 목적과 주요 내용
2. **핵심 조문**: 가장 중요한 조문들과 그 의미
3. **적용 범위**: 어떤 기업이나 거래에 적용되는지
4. **주요 제재**: 위반 시 어떤 제재가 있는지

요약은 일반인이 이해하기 쉽게 작성해주세요."""

        user_message = f"""다음 법령 정보를 요약해주세요:

{context}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = self.llm.invoke(messages)
        return response.content

def main():
    """
    메인 실행 함수
    """
    print("공정거래 RAG 시스템을 초기화합니다...")
    
    # RAG 시스템 초기화
    rag = FairTradeRAG()
    
    # 법령 데이터 로드
    laws = rag.load_law_data()
    
    if not laws:
        print("법령 데이터가 없습니다. 먼저 law_data_collector.py를 실행해주세요.")
        return
    
    print(f"{len(laws)}개의 법령을 로드했습니다.")
    
    # 벡터 데이터베이스 생성
    print("벡터 데이터베이스를 생성합니다...")
    documents = rag.prepare_documents(laws)
    rag.create_vector_database(documents)
    
    # 테스트 케이스 분석
    test_case = """
    A기업은 자동차 부품 제조업체로, B기업으로부터 하도급 작업을 받아왔습니다. 
    최근 B기업이 갑자기 하도급 대금을 30% 삭감하겠다고 통보했고, 
    계약서에는 "원청의 요청에 따라 단가 조정 가능"이라는 조항이 있습니다. 
    A기업은 이에 반대했지만 B기업은 "계약서에 명시되어 있다"며 강행하려고 합니다.
    """
    
    print("\n=== 케이스 분석 테스트 ===")
    print("케이스:", test_case.strip())
    print("\n분석 결과:")
    analysis = rag.analyze_case(test_case)
    print(analysis)
    
    print("\n=== 법령 요약 ===")
    summary = rag.get_law_summary()
    print(summary)

if __name__ == "__main__":
    main() 