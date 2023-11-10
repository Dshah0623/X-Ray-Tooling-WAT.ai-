from dotenv import load_dotenv
import os
import requests
import json
from bs4 import BeautifulSoup
load_dotenv()


class SpringerXrayScraper:
    """
    class for Springer API scraper
    """

    def __init__(self, api_key):
        self.api_key = api_key

    def search_springer_articles(self, query, max_results=10, save_folder="/Users/jeevanparmar/Desktop/co_ops/WAT.ai/X-Ray-Tooling/Webscraper/Springer_PDF"):
        """
        Retrieve up to max_results articles retrieved from query and 
        return a list of dictionaries of the metadata along with the full text if available.
        """
        article_metadata = self._perform_search(query, max_results)
        for article in article_metadata:
            # Assuming 'url' is the key for the full text link
            # and 'openaccess' indicates if the full text is freely available
            if 'openaccess' in article and article['openaccess']:
                full_text_link = self._get_full_text_link(article)
                if full_text_link:
                    article['FullText'] = self._fetch_full_article(
                        full_text_link, save_folder)
        return article_metadata

    def _get_full_text_link(self, article):
        """
        Extract the full text link list from the article metadata.
        """
        return article.get('url', [])

    def _perform_search(self, query, max_results):
        """
        search query into the Springer API and return up to max_results
        """
        try:
            url = "http://api.springer.com/metadata/json"
            params = {
                "q": query,
                "p": max_results,
                "api_key": self.api_key,
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            articles = response.json()["records"]
            return articles
        except requests.exceptions.HTTPError as e:
            print(f"HTTPError performing Springer search: {e}")
        except Exception as e:
            print(f"Error performing Springer search: {e}")
        return []

    def _fetch_full_article(self, url_list, save_folder):
        """
        Download the PDF of the article from the given URL list and save it in the specified folder.
        """
        # Extract the URL from the list of URL dictionaries
        actual_url = next(
            (link['value'] for link in url_list if link.get('format') == ''), None)
        if not actual_url:
            print("No valid URL found in URL list.")
            return None

        try:
            pdf_url = self._get_pdf_download_link(actual_url)

            if pdf_url:
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()

                # Extract filename from URL or set a default one
                filename = pdf_url.split('/')[-1] or 'downloaded_article.pdf'
                file_path = os.path.join(save_folder, filename)

                with open(file_path, 'wb') as f:
                    f.write(response.content)

                print(f"PDF downloaded successfully from URL: {pdf_url}")
                print(f"PDF downloaded successfully to: {file_path}")
                return file_path
        except Exception as e:
            print(f"Error downloading PDF: {e}")
        return None

    def _get_pdf_download_link(self, article_url):
        """
        Scrape the article page to find the PDF download link.
        """
        try:
            response = requests.get(article_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            pdf_link_element = soup.find('span', class_='c-pdf-download__text')

            if pdf_link_element and pdf_link_element.parent and pdf_link_element.parent.has_attr('href'):
                relative_pdf_url = pdf_link_element.parent['href']

                # Check if the URL is already absolute
                if relative_pdf_url.startswith(('http', 'https')):
                    return relative_pdf_url

                # Correctly concatenate the base URL and the relative URL
                base_url = 'https://link.springer.com'  # Assuming this is the correct base URL
                if relative_pdf_url.startswith('/'):
                    return base_url + relative_pdf_url
                else:
                    return base_url + '/' + relative_pdf_url

        except Exception as e:
            print(f"Error while scraping PDF link: {e}")

        return None

    def search_and_fetch_full_texts(self, query, max_results=10):
        """
        Search articles and fetch their full text if available.
        """
        articles_metadata = self.search_springer_articles(query, max_results)
        for article in articles_metadata:
            full_text_links = self._get_full_text_link(article)
            if full_text_links:  # Check if there are any full text links
                article['FullText'] = self._fetch_full_article(full_text_links)
        return articles_metadata

    def save_to_json(self, data, filename):
        """
        save the list of dictionaries to a json file
        """
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    api_key = os.environ.get("SPRINGER_API_KEY")
    springer_scraper = SpringerXrayScraper(api_key)
    springer_articles = springer_scraper.search_springer_articles(
        'title:"X-ray" AND openaccess:"true"', max_results=10)
    springer_scraper.save_to_json(
        springer_articles, "/Users/jeevanparmar/Desktop/co_ops/WAT.ai/X-Ray-Tooling/Webscraper/springer_articles.json")
