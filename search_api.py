import requests
import re
from langchain_community.tools import DuckDuckGoSearchResults
from newspaper import Article
import xml.etree.ElementTree as ET
import io
from pypdf import PdfReader

# 配置请求头模拟浏览器访问
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def parse_arxiv_xml(xml_data,MODE: bool):
    root = ET.fromstring(xml_data)

    # arXiv 使用了命名空间，需要声明
    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

    entry = root.find('atom:entry', ns)

    # 提取字段
    title = entry.find('atom:title', ns).text.strip()
    summary = entry.find('atom:summary', ns).text.strip()
    published = entry.find('atom:published', ns).text
    authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
    pdf_link = None
    for link in entry.findall('atom:link', ns):
        if link.attrib.get('title') == 'pdf':
            pdf_link = link.attrib['href']
    if MODE:
        response = requests.get(pdf_link)
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        # print(text)
        return text
    else:
        # print(summary)
        return summary

def fetch_webpage(url):
    """改进的网页获取函数，支持自动重试"""
    if "arxiv.org" in url:
        # 使用arxiv官方API获取结构化数据
        api_url = f"http://export.arxiv.org/api/query?id_list={url.split('/')[-1]}"
        response = requests.get(api_url)
        return response.text
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        # 检查是否是验证页面（如知乎的反爬机制）
        if "安全检查" in response.text:
            print(f"触发验证页面: {url}")
            return None
        return response.text
    except requests.exceptions.HTTPError as e:
        print(f"HTTP错误 {e.response.status_code} at {url}")
    except Exception as e:
        print(f"获取页面失败: {str(e)}")
    return None

def extract_content(html, url):
    """使用newspaper3k提取正文内容"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error parsing {url}: {str(e)}")
        return None

def parse_custom_format(raw_str):
    """解析DuckDuckGo返回的非标准格式字符串"""
    results = []
    # 使用正则表达式匹配每个结果项
    pattern = r'\[(snippet: (.*?), title: (.*?), link: (.*?))\]'
    matches = re.findall(pattern, raw_str)

    for match in matches:
        try:
            result = {
                'snippet': match[1].strip(),
                'title': match[2].strip(),
                'link': match[3].strip()
            }
            # 处理arXiv的特殊标题格式
            if "arxiv.org" in result['link']:
                result['title'] = re.sub(r'\[\d+\.\d+\]', '', result['title']).strip()
            results.append(result)
        except IndexError:
            continue
    return results

def search_results(query):
    """执行搜索并爬取结果页面"""
    search = DuckDuckGoSearchResults(output_format='list')
    results = search.invoke(query)

    references = []

    for i, result in enumerate(results, 1):
        print(f"\n{'='*50}\nProcessing result {i}/{len(results)}")
        print(f"Title: {result['title']}")
        print(f"URL: {result['link']}")
        if result['title'] == 'EOF':
            return references

        html = fetch_webpage(result['link'])
        if html:
            if 'arxiv.org' in result['link']:
                references.append({
                    'title': result['title'],
                    'link': result['link'],
                    'content': parse_arxiv_xml(html,False)
                })
                continue
            content = extract_content(html, result['link'])
            if content:
                # print(f"\nExtracted content ({len(content)} characters):")
                # print(content[:500] + "...")  # 显示前500个字符
                references.append({
                    'title': result['title'],
                    'link': result['link'],
                    'content': content
                })
            # else:
                # print("Failed to extract content")
        # else:
            # print("Failed to fetch webpage")
    return references

if __name__ == "__main__":
    query = "自我原则点评调优（SPCT）与元奖励模型（Meta Reward Model）"
    all_references = search_results(query)  # Store the returned list
    print("\n--- Extracted References ---")
    if all_references:
        for ref in all_references:
            print(f"Title: {ref['title']}")
            print(f"Link: {ref['link']}")
            print(f"Content (first 500 chars):\n{ref['content'][:500]}...\n")
            print("-" * 30)
    else:
        print("No references found or extracted.")