import wikipedia
import tensorflow as tf
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import word_tokenize
import os
import warnings
#nltk.download('punkt')
DATA_PATH = "../../data/"
# queries = ["machine learning", "networks", "artificial intelligence", "computer science", "neural networks", "natural language processing"]
queries = ["Wireless network"]
#download punkt
def wiki_scrape(queries):
    """Downloads and scrapes queries from wikipedia to data folder"""
    warnings.catch_warnings()
    warnings.simplefilter("ignore")
    wikipedia.set_rate_limiting(0.01) #set limit so the all powerful webmaster doesn't block us
    #get a random page content
    print(f"queries: {queries}")
    for query in queries:
        result = wikipedia.search(query) 
        print(f"query: {query}, result: {result}")
        for i in range(len(result)):
            try:
                page = wikipedia.page(result[i])
                content = page.content.lower()
                title = page.title.replace(" ","_")
                if title not in os.listdir(DATA_PATH):
                    with open(f"{DATA_PATH}{title}.txt", "w") as infile:
                        infile.write(content)
                        infile.close()
                    print(f"Added: {title}")
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"error: {e}")
                continue
            except wikipedia.exceptions.PageError as e:
                print(f"error: {e}")
                continue
            except FileNotFoundError as e:
                print(f"error: {e}")
                continue
            
# wiki_scrape(queries)
        
