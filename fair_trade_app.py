import streamlit as st
import json
import os
from law_data_collector import LawDataCollector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 페이지 설정
st.set_page_config(
    page_title="공정거래 법령 분석 시스템",
    page_icon="⚖️",
    layout="wide"
)

# API 키 설정 함수
def setup_api_key():
    """API 키를 안전하게 설정"""
    # Streamlit Cloud Secrets에서 가져오기 (배포용)
    try:
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
            return True
    except:
        pass
    
    # 환경변수에서 가져오기 (로컬용)
    if os.environ.get('OPENAI_API_KEY'):
        return True
    
    return False

# 사이드바
st.sidebar.title("⚖️ 공정거래 법령 분석")
st.sidebar.markdown("---")

# 메인 기능 선택
page = st.sidebar.selectbox(
    "기능 선택",
    ["🏠 홈", "📊 데이터 수집", "🔍 케이스 분석", "📋 법령 요약", "⚙️ 설정"]
)

class SimpleFairTradeRAG:
    """간단한 법령 분석 시스템"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.documents = []
        self.doc_vectors = None
        
    def load_law_data(self, filename="fair_trade_laws.json"):
        """법령 데이터 로드"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
            
    def prepare_documents(self, laws):
        """법령을 검색 가능한 문서로 변환"""
        documents = []
        for law in laws:
            if law.get('content'):
                documents.append({
                    'text': f"{law['title']} {law['content']}",
                    'title': law['title'],
                    'type': 'full_law'
                })
            
            for article in law.get('articles', []):
                documents.append({
                    'text': f"{law['title']} 제{article['number']}조 {article.get('title', '')} {article['content']}",
                    'title': f"{law['title']} 제{article['number']}조",
                    'type': 'article'
                })
        
        self.documents = documents
        if documents:
            texts = [doc['text'] for doc in documents]
            self.doc_vectors = self.vectorizer.fit_transform(texts)
        
        return len(documents)
    
    def search_relevant_documents(self, query, n_results=5):
        """관련 문서 검색"""
        if not self.documents or self.doc_vectors is None:
            return []
            
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        top_indices = similarities.argsort()[-n_results:][::-1]
        results = []
        
        for idx in top_indices:
            results.append({
                'text': self.documents[idx]['text'],
                'title': self.documents[idx]['title'],
                'similarity': similarities[idx]
            })
        
        return results
    
    def analyze_case_simple(self, case_description):
        """간단한 케이스 분석 (API 없이)"""
        relevant_docs = self.search_relevant_documents(case_description, n_results=3)
        
        if not relevant_docs:
            return "관련 법령 데이터가 없습니다. 먼저 데이터를 수집해주세요."
        
        result = "## 📊 관련 법령 분석\n\n"
        
        for i, doc in enumerate(relevant_docs, 1):
            result += f"### {i}. {doc['title']}\n"
            result += f"**유사도:** {doc['similarity']:.3f}\n\n"
            result += f"**내용:** {doc['text'][:300]}...\n\n"
            result += "---\n\n"
        
        result += "## 💡 분석 요약\n\n"
        result += "위의 관련 법령들을 검토하여 케이스의 법적 쟁점을 파악하시기 바랍니다.\n"
        result += "더 정확한 분석을 위해서는 OpenAI API 키를 설정하여 AI 분석 기능을 활용하세요."
        
        return result
    
    def analyze_case_ai(self, case_description):
        """AI를 활용한 상세 케이스 분석"""
        if not os.environ.get('OPENAI_API_KEY'):
            return self.analyze_case_simple(case_description)
        
        relevant_docs = self.search_relevant_documents(case_description, n_results=5)
        
        if not relevant_docs:
            return "관련 법령 데이터가 없습니다. 먼저 데이터를 수집해주세요."
        
        context = "관련 법령:\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"{i}. {doc['title']}\n{doc['text'][:400]}...\n\n"
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 공정거래 전문 변호사입니다. 주어진 케이스와 관련 법령을 바탕으로 법적 분석을 제공해주세요."},
                    {"role": "user", "content": f"케이스: {case_description}\n\n{context}\n\n위 케이스를 분석해주세요."}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI 분석 중 오류 발생: {str(e)}\n\n" + self.analyze_case_simple(case_description)

def home_page():
    """홈 페이지"""
    st.title("공정거래 법령 분석 시스템")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 시스템 개요")
        st.markdown("""
        이 시스템은 공정거래 관련 법령들을 분석하여 
        기업의 공정거래 케이스를 평가하는 AI 도구입니다.
        
        **주요 기능:**
        - 📊 법령 데이터 수집 및 저장
        - 🔍 케이스별 법적 분석
        - 📋 법령 요약 및 설명
        - ⚖️ 위반 가능성 평가
        """)
    
    with col2:
        st.subheader("📈 지원 법령")
        st.markdown("""
        **핵심 법령:**
        - 공정거래법
        - 하도급법
        - 상생협력법
        
        **관련 법령:**
        - 독점규제 및 공정거래에 관한 법률
        - 하도급거래 공정화에 관한 법률
        - 대·중소기업 상생협력 촉진에 관한 법률
        """)
    
    st.markdown("---")
    
    # 시스템 상태 확인
    st.subheader("🔧 시스템 상태")
    
    # 데이터 파일 확인
    if os.path.exists("fair_trade_laws.json"):
        with open("fair_trade_laws.json", 'r', encoding='utf-8') as f:
            laws = json.load(f)
        st.success(f"✅ 법령 데이터 로드됨 ({len(laws)}개 법령)")
    else:
        st.warning("⚠️ 법령 데이터가 없습니다. '데이터 수집' 페이지에서 데이터를 수집해주세요.")
    
    # API 키 확인
    if setup_api_key():
        st.success("✅ OpenAI API 키 설정됨 (AI 분석 가능)")
    else:
        st.info("💡 OpenAI API 키 미설정 (기본 분석만 가능)")

def data_collection_page():
    """데이터 수집 페이지"""
    st.title("📊 법령 데이터 수집")
    st.markdown("---")
    
    st.markdown("""
    국가법령정보센터에서 공정거래 관련 법령들을 수집합니다.
    """)
    
    if st.button("🚀 데이터 수집 시작", type="primary"):
        with st.spinner("법령 데이터를 수집하고 있습니다..."):
            try:
                collector = LawDataCollector()
                laws = collector.collect_fair_trade_laws()
                
                if laws:
                    collector.save_laws_to_file(laws)
                    st.success(f"✅ {len(laws)}개의 법령을 성공적으로 수집했습니다!")
                    
                    # 수집된 법령 목록 표시
                    st.subheader("📋 수집된 법령 목록")
                    for i, law in enumerate(laws, 1):
                        with st.expander(f"{i}. {law['title']}"):
                            st.write(f"**키워드:** {law.get('keyword', 'N/A')}")
                            st.write(f"**조문 수:** {len(law.get('articles', []))}")
                            st.write(f"**URL:** {law.get('url', 'N/A')}")
                else:
                    st.error("❌ 데이터 수집에 실패했습니다.")
                    
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")

def case_analysis_page():
    """케이스 분석 페이지"""
    st.title("🔍 케이스 분석")
    st.markdown("---")
    
    # RAG 시스템 초기화
    if 'rag' not in st.session_state:
        st.session_state.rag = SimpleFairTradeRAG()
    
    rag = st.session_state.rag
    
    # 법령 데이터 확인 및 로드
    laws = rag.load_law_data()
    if not laws:
        st.warning("⚠️ 법령 데이터가 없습니다. 먼저 데이터를 수집해주세요.")
        return
    
    # 문서 준비
    if not rag.documents:
        with st.spinner("법령 데이터를 준비하고 있습니다..."):
            doc_count = rag.prepare_documents(laws)
            st.success(f"✅ {doc_count}개의 문서가 준비되었습니다!")
    
    st.markdown("---")
    
    # 분석 모드 선택
    analysis_mode = st.radio(
        "분석 모드 선택:",
        ["🤖 AI 분석 (OpenAI API 필요)", "📝 기본 분석 (API 불필요)"]
    )
    
    # 케이스 입력
    st.subheader("📝 케이스 입력")
    
    # 예시 케이스들
    example_cases = {
        "하도급 대금 삭감": """
        A기업은 자동차 부품 제조업체로, B기업으로부터 하도급 작업을 받아왔습니다. 
        최근 B기업이 갑자기 하도급 대금을 30% 삭감하겠다고 통보했고, 
        계약서에는 "원청의 요청에 따라 단가 조정 가능"이라는 조항이 있습니다. 
        A기업은 이에 반대했지만 B기업은 "계약서에 명시되어 있다"며 강행하려고 합니다.
        """,
        "독점적 지위 남용": """
        대기업 C는 특정 시장에서 80% 이상의 점유율을 가지고 있습니다. 
        최근 C기업이 중소기업 D에게 "우리 제품만 사용하라"며 
        다른 업체 제품 사용을 금지하고, 이를 어길 경우 거래 중단을 위협하고 있습니다.
        """,
        "불공정 거래 조건": """
        대기업 E는 중소기업 F와 거래하면서 다음과 같은 조건을 제시했습니다:
        - 90일 후 지급 조건 (기존 30일에서 변경)
        - 품질 보증금 20% 예치 요구
        - 계약 해지 시 30일 전 통보
        F기업은 이러한 조건들이 너무 까다롭다고 생각합니다.
        """
    }
    
    selected_example = st.selectbox(
        "예시 케이스 선택 (선택사항):",
        ["직접 입력"] + list(example_cases.keys())
    )
    
    if selected_example != "직접 입력":
        case_description = st.text_area(
            "케이스 설명:",
            value=example_cases[selected_example],
            height=200
        )
    else:
        case_description = st.text_area(
            "케이스 설명:",
            placeholder="분석하고 싶은 공정거래 관련 케이스를 자세히 설명해주세요...",
            height=200
        )
    
    if st.button("🔍 분석 시작", type="primary") and case_description.strip():
        with st.spinner("케이스를 분석하고 있습니다..."):
            try:
                if "🤖 AI 분석" in analysis_mode:
                    analysis_result = rag.analyze_case_ai(case_description)
                else:
                    analysis_result = rag.analyze_case_simple(case_description)
                
                st.subheader("📊 분석 결과")
                st.markdown(analysis_result)
                
                # 분석 결과 다운로드
                st.download_button(
                    label="📥 분석 결과 다운로드",
                    data=analysis_result,
                    file_name="공정거래_케이스_분석.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"❌ 분석 중 오류 발생: {str(e)}")

def law_summary_page():
    """법령 요약 페이지"""
    st.title("📋 법령 요약")
    st.markdown("---")
    
    if 'rag' not in st.session_state:
        st.session_state.rag = SimpleFairTradeRAG()
    
    rag = st.session_state.rag
    laws = rag.load_law_data()
    
    if not laws:
        st.warning("⚠️ 법령 데이터가 없습니다. 먼저 데이터를 수집해주세요.")
        return
    
    # 법령 목록 표시
    st.subheader("📋 수집된 법령 목록")
    for i, law in enumerate(laws, 1):
        with st.expander(f"{i}. {law['title']}"):
            st.write(f"**키워드:** {law.get('keyword', 'N/A')}")
            if law.get('content'):
                st.write(f"**내용:** {law['content'][:500]}...")
            st.write(f"**조문 수:** {len(law.get('articles', []))}")

def settings_page():
    """설정 페이지"""
    st.title("⚙️ 설정")
    st.markdown("---")
    
    st.subheader("🔑 OpenAI API 키 설정")
    
    # API 키 입력
    api_key = st.text_input(
        "OpenAI API 키:",
        type="password",
        help="OpenAI API 키를 입력하세요. AI 분석 기능을 사용하려면 필요합니다."
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ API 키가 설정되었습니다.")
    elif setup_api_key():
        st.success("✅ API 키가 이미 설정되어 있습니다.")
    else:
        st.info("💡 API 키가 없어도 기본 분석 기능은 사용 가능합니다.")
    
    st.markdown("---")
    
    st.subheader("🗂️ 데이터 관리")
    
    if st.button("🗑️ 법령 데이터 삭제"):
        if os.path.exists("fair_trade_laws.json"):
            os.remove("fair_trade_laws.json")
            st.success("✅ 법령 데이터가 삭제되었습니다.")
            # 세션 상태 초기화
            if 'rag' in st.session_state:
                del st.session_state.rag
        else:
            st.warning("⚠️ 삭제할 법령 데이터가 없습니다.")
    
    st.markdown("---")
    
    st.subheader("📊 시스템 정보")
    
    # 파일 상태 확인
    if os.path.exists("fair_trade_laws.json"):
        file_size = os.path.getsize("fair_trade_laws.json") / 1024  # KB
        st.info(f"📄 법령 데이터 파일: {file_size:.1f} KB")
    else:
        st.warning("📄 법령 데이터 파일: 없음")

# 페이지 라우팅
if page == "🏠 홈":
    home_page()
elif page == "📊 데이터 수집":
    data_collection_page()
elif page == "🔍 케이스 분석":
    case_analysis_page()
elif page == "📋 법령 요약":
    law_summary_page()
elif page == "⚙️ 설정":
    settings_page()