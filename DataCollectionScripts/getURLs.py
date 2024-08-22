# https://stackoverflow.com/questions/59347372/how-extract-all-urls-in-a-website-using-beautifulsoup
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://starwars.fandom.com/wiki/Special:AllPages"
df = pd.DataFrame()
links = []
def extract_links(url):
    print("source url", url)
    if "/Template:" in url or "/Forum:" in url or "file=" in url or ".jpg" in url or ".png" in url or "/Category:" in url or "/Wookieepedia:" in url or "/WP:" in url:
        return
    global links
    source_url = requests.get(url)
    soup = BeautifulSoup(source_url.content,"html.parser")
    if soup.find_all('div',{"class":"mw-allpages-body"}):
        for link in soup.find_all('div',{"class":"mw-allpages-body"})[0].find_all('a',href=True):
            link['href'] = "https://starwars.fandom.com"+link.get('href')
            try:
                if link.get('href').startswith("https://") and link.get("href") not in links:
                    # Redundant check, but it's fine
                    if "/Template:" not in str(link.get("href")) and "/Forum:" not in str(link.get("href")) and "file=" not in str(link.get("href")) and ".jpg" not in str(link.get("href")) and ".png" not in str(link.get("href")) and "/Category:" not in str(link.get("href")) and "/Wookieepedia:" not in str(link.get("href")) and "/WP:" not in str(link.get("href")):
                        # check if https://starwars.fandom.com/wiki/ is in the url
                        if "https://starwars.fandom.com/wiki/" in str(link.get("href")):
                            links.append(link.get('href'))
                            extract_links(link.get('href'))
            except Exception as e:
                print("Unhandled exception",e)
    if soup.find_all('div',{"class":"mw-allpages-nav"}):
        for link in soup.find_all('div',{"class":"mw-allpages-nav"})[0].find_all('a',href=True):
            link['href'] = "https://starwars.fandom.com"+link.get('href')
            try:
                if link.get('href').startswith("https://") and link.get("href") not in links:
                    # Redundant check, but it's fine
                    if "/Template:" not in str(link.get("href")) and "/Forum:" not in str(link.get("href")) and "file=" not in str(link.get("href")) and ".jpg" not in str(link.get("href")) and ".png" not in str(link.get("href")) and "/Category:" not in str(link.get("href")) and "/Wookieepedia:" not in str(link.get("href")) and "/WP:" not in str(link.get("href")):
                        # check if https://starwars.fandom.com/wiki/ is in the url
                        if "https://starwars.fandom.com/wiki/" in str(link.get("href")):
                            links.append(link.get('href'))
                            extract_links(link.get('href'))
            except Exception as e:
                print("Unhandled exception",e)

extract_links(url)
df = pd.DataFrame({"links":links})
df.to_csv("allpages.csv", index=False)