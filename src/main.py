import scrapping.wiki_scrape as ws
import os
from nltk.tokenize import word_tokenize
from queries.txt_query import txt_to_query
QUERY_FILE = "math.txt"
def main():
    # word_tokenize("")
    # query = input("Enter your domain of interest: \n")
    os.chdir("queries")
    query = txt_to_query(QUERY_FILE)
    os.chdir("../scrapping")
    ws.wiki_scrape(query)

if (__name__ == "__main__"):
    main()