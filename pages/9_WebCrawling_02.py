import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse, quote
import os
from datetime import datetime
import streamlit as st
import time

# URL 패턴을 정의하는 정규식
HTTP_URL_PATTERN = r'^http[s]*://.+'

# 크롤링할 도메인과 전체 URL을 정의
domain = "developer.samsung.com/conference/sdc23/"
full_url = "https://developer.samsung.com/conference/sdc23/"

# HTML 파서를 정의하는 클래스
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hyperlinks = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

# URL에서 하이퍼링크를 가져오는 함수
def get_hyperlinks(url):
    try:
        with urllib.request.urlopen(url) as response:
            if not response.info().get('Content-Type').startswith("text/html"):
                return []
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []
    parser = HyperlinkParser()
    parser.feed(html)
    return parser.hyperlinks

# 같은 도메인 내의 하이퍼링크를 가져오는 함수
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None
        if re.search(HTTP_URL_PATTERN, link):
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + local_domain + "/" + link
        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)
    return list(set(clean_links))

# 파일 이름을 안전하게 변환하는 함수
def safe_filename(url):
    return quote(url[8:].replace("/", "_"), safe="")

# 크롤링 함수
def crawl(url):
    local_domain = urlparse(url).netloc
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    text_dir = f"text_{local_domain}_{timestamp}"
    
    queue = deque([url])
    seen = set([url])

    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    start_time = time.time()
    page_count = 0

    while queue:
        url = queue.pop()
        st.write(f"Crawling: {url}")
        
        # 특정 경로가 포함된 경우 스킵
        if any(exclude in url for exclude in ["/file/", "/../../../../../", "login?redirectURL=", "/search?query="]):
            st.write(f"Skipping URL due to specific path: {url}")
            continue

        # 특정 도메인 경로 포함 여부 확인
        if "developer.samsung.com/conference/sdc23/" not in url:
            st.write(f"Skipping URL as it does not contain the required path: {url}")
            continue

        try:
            response = requests.get(url)
            response.raise_for_status()  # 추가된 에러 처리
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            
            if "You need to enable JavaScript to run this app." in text:
                st.write(f"Unable to parse page {url} due to JavaScript being required")
                continue
                
            # 연속된 줄 바꿈이 3개 이상일 경우에만 줄 바꿈 2개로 변환
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            with open(os.path.join(text_dir, safe_filename(url) + ".txt"), "w", encoding="UTF-8") as f:
                # URL 정보를 key-value 형식으로 쓰기
                f.write(f"url: {url}\n")
                # 텍스트 쓰기
                f.write(text)
            
            page_count += 1
        except Exception as e:
            st.write(f"Failed to process URL {url}: {e}")
            continue

        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen and "developer.samsung.com/conference/sdc23/" in link:
                queue.append(link)
                seen.add(link)

    end_time = time.time()
    elapsed_time = end_time - start_time

    st.write(f"Crawling completed! Total pages crawled: {page_count}")
    st.write(f"Time taken: {elapsed_time:.2f} seconds")

# Streamlit UI
st.title('Samsung Developer Conference Crawler')
st.write('This application will crawl the URL: https://developer.samsung.com/conference/sdc23/')

if st.button('Crawl'):
    crawl(full_url)
