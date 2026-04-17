import os
import re
import time
import html
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"

HEADERS = {
    "User-Agent": "qa-agent-kb-builder/1.0 (academic-use)"
}


DOMAIN_TOPICS = {
    "machine_learning": [
        "Neural network",
        "Overfitting",
        "Regularization (mathematics)",
        "Cross-validation (statistics)",
        "Gradient descent",
        "Transformer (deep learning)",
        "Support vector machine",
        "Random forest",
    ],
    "networking": [
        "Transmission Control Protocol",
        "Domain Name System",
        "Internet Protocol",
        "Hypertext Transfer Protocol",
        "Routing",
        "Packet switching",
        "Subnet",
        "Border Gateway Protocol",
    ],
    "healthcare": [
        "diabetes mellitus",
        "hypertension",
        "cardiovascular disease",
        "asthma",
        "chronic kidney disease",
        "depression",
        "obesity",
        "stroke",
    ],
    "cybersecurity": [
        "phishing",
        "encryption",
        "authentication",
        "authorization",
        "malware",
        "ransomware",
        "least privilege",
        "zero trust",
    ],
}


def safe_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "document"


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def write_text(domain: str, title: str, content: str) -> None:
    folder = KB_DIR / domain
    folder.mkdir(parents = True, exist_ok = True)
    path = folder / f"{safe_filename(title)}.txt"
    path.write_text(content.strip() + "\n", encoding = "utf-8")


def fetch_wikipedia_summary(title: str) -> Optional[str]:
    """
    Uses Wikimedia REST API summary endpoint.
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title, safe = '')}"
    resp = requests.get(url, headers=HEADERS, timeout=30)

    if resp.status_code != 200:
        return None

    data = resp.json()
    extract = data.get("extract")
    page_title = data.get("title", title)

    if not extract:
        return None

    return f"Title: {page_title}\nSource: Wikipedia\n\n{clean_text(extract)}"


def pubmed_search(term: str, retmax: int = 3) -> List[str]:
    """
    Search PubMed IDs using NCBI E-utilities.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
    }
    resp = requests.get(url, params = params, headers = HEADERS, timeout = 30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def pubmed_fetch_abstracts(pmids: List[str]) -> List[Dict[str, str]]:
    """
    Fetch PubMed abstracts using efetch in XML.
    """
    if not pmids:
        return []

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    
    resp = requests.get(url, params = params, headers = HEADERS, timeout = 30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "xml")
    results = []

    for article in soup.find_all("PubmedArticle"):
        title_tag = article.find("ArticleTitle")
        abstract_tags = article.find_all("AbstractText")

        title = clean_text(title_tag.get_text(" ", strip = True)) if title_tag else "Untitled"
        abstract = " ".join(
            clean_text(tag.get_text(" ", strip = True)) for tag in abstract_tags
        ).strip()

        if abstract:
            results.append({
                "title": title,
                "abstract": abstract,
            })

    return results


def build_healthcare_from_pubmed(topics: List[str], per_topic: int = 2) -> None:
    for topic in tqdm(topics, desc = "Healthcare KB", unit = "topic"):
        try:
            pmids = pubmed_search(topic, retmax = per_topic)
            articles = pubmed_fetch_abstracts(pmids)

            for idx, article in enumerate(articles, start = 1):
                title = f"{topic}_{idx}_{article['title'][:60]}"
                content = (
                    f"Topic: {topic}\n"
                    f"Source: PubMed\n"
                    f"Article Title: {article['title']}\n\n"
                    f"{article['abstract']}"
                )

                write_text("healthcare", title, content)

            time.sleep(0.4)

        except Exception as e:
            print(f"[healthcare] failed for topic '{topic}': {e}")


def fetch_nist_glossary_term(term: str) -> Optional[str]:
    """
    Simple scrape of the NIST glossary search page by term.
    """
    url = "https://csrc.nist.gov/glossary"
    params = {"term": term}
    resp = requests.get(url, params = params, headers = HEADERS, timeout = 30)

    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text("\n", strip = True)
    text = clean_text(text)

    if not text:
        return None

    snippet = text[:3000]
    return f"Term: {term}\nSource: NIST Glossary\n\n{snippet}"


def fetch_owasp_top10() -> Optional[str]:
    url = "https://owasp.org/www-project-top-ten/"
    resp = requests.get(url, headers = HEADERS, timeout = 30)

    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text("\n", strip = True)
    text = clean_text(text)

    if not text:
        return None

    return f"Title: OWASP Top 10\nSource: OWASP\n\n{text[:5000]}"


def build_cybersecurity_kb(topics: List[str]) -> None:
    for topic in tqdm(topics, desc = "Building Cybersecurity KB", unit = "topic"):
        try:
            nist_text = fetch_nist_glossary_term(topic)

            if nist_text:
                write_text("cybersecurity", f"nist_{topic}", nist_text)

            time.sleep(0.3)

        except Exception as e:
            print(f"[cybersecurity] NIST failed for '{topic}': {e}")

    try:
        owasp_text = fetch_owasp_top10()

        if owasp_text:
            write_text("cybersecurity", "owasp_top_10", owasp_text)

    except Exception as e:
        print(f"[cybersecurity] OWASP fetch failed: {e}")


def build_wikipedia_domain(domain: str, topics: List[str]) -> None:
    for topic in tqdm(topics, desc = f"Building {domain} KB"):
        try:
            text = fetch_wikipedia_summary(topic)

            if text:
                write_text(domain, topic, text)

            time.sleep(0.3)

        except Exception as e:
            print(f"[{domain}] failed for topic '{topic}': {e}")


def main() -> None:
    KB_DIR.mkdir(parents=True, exist_ok=True)

    # Machine learning from Wikipedia
    build_wikipedia_domain("machine_learning", DOMAIN_TOPICS["machine_learning"])

    # Networking from Wikipedia
    build_wikipedia_domain("networking", DOMAIN_TOPICS["networking"])

    # Healthcare from PubMed
    build_healthcare_from_pubmed(DOMAIN_TOPICS["healthcare"], per_topic=2)

    # Cybersecurity from NIST + OWASP
    build_cybersecurity_kb(DOMAIN_TOPICS["cybersecurity"])

    print(f"Knowledge base created at: {KB_DIR}")


if __name__ == "__main__":
    main()