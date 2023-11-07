from Bio import Entrez
import requests
import json

class PubMedXrayScraper:
    def __init__(self, email):
        self.email = email
        self.api_key = None  # Set your API key here if needed
        Entrez.email = email

    def search_xray_articles(self, query, max_results=10):
        search_results = self._perform_search(query, max_results)
        articles = self._fetch_articles_metadata(search_results)
        self._fetch_full_articles(articles)
        return articles

    def _perform_search(self, query, max_results):
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()
            return record["IdList"]
        except Exception as e:
            print(f"Error performing PubMed search: {e}")
            return []

    def _fetch_articles_metadata(self, pmids):
        articles = []
        for pmid in pmids:
            article_data = self._fetch_article_metadata(pmid)
            if article_data:
                articles.append(article_data)
        return articles

    def _fetch_article_metadata(self, pmid):
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="text")
            record = Entrez.read(handle)
            handle.close()
            article_data = self._parse_article_metadata(record)
            return article_data
        except Exception as e:
            print(f"Error fetching article metadata for PMID {pmid}: {e}")
            return None

    def _parse_article_metadata(self, record):
        article_data = {
            "Title": record['PubmedArticle'][0]['MedlineCitation']['Article']['ArticleTitle'],
            "Authors": [author['LastName'] + ' ' + author['Initials'] for author in record['PubmedArticle'][0]['MedlineCitation']['Article']['AuthorList']],
            "Abstract": record['PubmedArticle'][0]['MedlineCitation']['Article']['Abstract']['AbstractText'],
            "PMID": record['PubmedArticle'][0]['PubmedData']['ArticleIdList'][0],
            "PublicationDate": record['PubmedArticle'][0]['PubmedData']['History'][0]['Year'],
        }
        return article_data

    def _fetch_full_articles(self, articles):
        for article in articles:
            pmid = article['PMID']
            article['FullText'] = self._fetch_full_article(pmid)

    def _fetch_full_article(self, pmid):
        try:
            bioC_url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/unicode"
            response = requests.get(bioC_url)
            if response.status_code == 200:
                return response.text
            else:
                print(f"Error fetching full article for PMID {pmid}: {response.status_code}")
        except Exception as e:
            print(f"Error fetching full article for PMID {pmid}: {e}")
        return None

    def save_to_json(self, data, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    email = "msspam132@gmail.com"  # Replace with your email address
    xray_scraper = PubMedXrayScraper(email)
    xray_articles = xray_scraper.search_xray_articles("X-ray AND open access", max_results=10)
    xray_scraper.save_to_json(xray_articles, "xray_articles.json")