import requests
from bs4 import BeautifulSoup
import json
import time
import re
from typing import List, Dict, Optional

class LawDataCollector:
    """
    국가법령정보센터에서 공정거래 관련 법령 데이터를 수집하는 클래스
    """
    
    def __init__(self):
        self.base_url = "https://www.law.go.kr"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_laws(self, keyword: str) -> List[Dict]:
        """
        키워드로 법령을 검색합니다.
        
        Args:
            keyword (str): 검색할 키워드
            
        Returns:
            List[Dict]: 검색된 법령 목록
        """
        search_url = f"{self.base_url}/lsSc.do?menuId=0&p1=&subMenu=1"
        
        params = {
            'query': keyword,
            'tabNo': '1'
        }
        
        try:
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            laws = []
            
            # 검색 결과에서 법령 정보 추출
            law_items = soup.find_all('div', class_='law_item')
            
            for item in law_items:
                title_elem = item.find('a', class_='law_title')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href')
                    
                    # 법령 상세 정보 추출
                    detail_info = item.find('div', class_='law_info')
                    info_text = detail_info.get_text(strip=True) if detail_info else ""
                    
                    laws.append({
                        'title': title,
                        'link': link,
                        'info': info_text
                    })
            
            return laws
            
        except Exception as e:
            print(f"법령 검색 중 오류 발생: {e}")
            return []
    
    def get_law_content(self, law_url: str) -> Optional[Dict]:
        """
        특정 법령의 상세 내용을 가져옵니다.
        
        Args:
            law_url (str): 법령 상세 페이지 URL
            
        Returns:
            Optional[Dict]: 법령 상세 정보
        """
        try:
            full_url = f"{self.base_url}{law_url}" if law_url.startswith('/') else law_url
            response = self.session.get(full_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 법령 제목
            title = soup.find('h1', class_='law_title')
            title_text = title.get_text(strip=True) if title else "제목 없음"
            
            # 법령 내용
            content_div = soup.find('div', class_='law_content')
            content = content_div.get_text(strip=True) if content_div else ""
            
            # 조문별 분리
            articles = self._extract_articles(content)
            
            return {
                'title': title_text,
                'url': full_url,
                'content': content,
                'articles': articles
            }
            
        except Exception as e:
            print(f"법령 내용 가져오기 중 오류 발생: {e}")
            return None
    
    def _extract_articles(self, content: str) -> List[Dict]:
        """
        법령 내용에서 조문을 추출합니다.
        
        Args:
            content (str): 법령 전체 내용
            
        Returns:
            List[Dict]: 조문 목록
        """
        articles = []
        
        # 조문 패턴 (제1조, 제2조 등)
        article_pattern = r'제(\d+)조\s*[\(\[（]?([^\)\]）]*)[\)\]）]?\s*(.*?)(?=제\d+조|$)'
        matches = re.finditer(article_pattern, content, re.DOTALL)
        
        for match in matches:
            article_num = match.group(1)
            article_title = match.group(2).strip()
            article_content = match.group(3).strip()
            
            articles.append({
                'number': article_num,
                'title': article_title,
                'content': article_content
            })
        
        return articles
    
    def collect_fair_trade_laws(self) -> List[Dict]:
        """
        공정거래 관련 법령들을 수집합니다.
        
        Returns:
            List[Dict]: 수집된 법령 목록
        """
        # 검색할 키워드들
        keywords = [
            "공정거래법",
            "하도급법", 
            "상생협력법",
            "독점규제",
            "공정거래",
            "하도급거래",
            "상생협력"
        ]
        
        all_laws = []
        
        for keyword in keywords:
            print(f"'{keyword}' 검색 중...")
            laws = self.search_laws(keyword)
            
            for law in laws:
                print(f"  - {law['title']} 처리 중...")
                detail = self.get_law_content(law['link'])
                if detail:
                    detail['keyword'] = keyword
                    all_laws.append(detail)
                
                # 서버 부하 방지를 위한 대기
                time.sleep(1)
        
        return all_laws
    
    def save_laws_to_file(self, laws: List[Dict], filename: str = "fair_trade_laws.json"):
        """
        수집된 법령 데이터를 파일로 저장합니다.
        
        Args:
            laws (List[Dict]): 저장할 법령 데이터
            filename (str): 저장할 파일명
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(laws, f, ensure_ascii=False, indent=2)
            print(f"법령 데이터가 {filename}에 저장되었습니다.")
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")

def main():
    """
    메인 실행 함수
    """
    print("공정거래 관련 법령 데이터 수집을 시작합니다...")
    
    collector = LawDataCollector()
    
    # 법령 수집
    laws = collector.collect_fair_trade_laws()
    
    print(f"\n총 {len(laws)}개의 법령을 수집했습니다.")
    
    # 파일로 저장
    collector.save_laws_to_file(laws)
    
    # 수집된 법령 목록 출력
    print("\n수집된 법령 목록:")
    for i, law in enumerate(laws, 1):
        print(f"{i}. {law['title']}")
        print(f"   키워드: {law['keyword']}")
        print(f"   조문 수: {len(law['articles'])}")
        print()

if __name__ == "__main__":
    main() 