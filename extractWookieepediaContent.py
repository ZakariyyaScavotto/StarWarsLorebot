import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def extractContent(url):
    # Get all the content from the page, which is all of the text displayed in the mw-parser-output div's p, h2, ul and li tags 
    # that is not in the "toc" class, "references-small" class, "p-lang" id, "Notes_and_references" id, either directly or having a parent/grandparent with that class or id
    source_url = requests.get(url)
    soup = BeautifulSoup(source_url.content,"html.parser")
    content = ""
    if soup.find_all('div',{"class":"mw-parser-output"}) == []:
        return ""
    for tag in soup.find_all('div',{"class":"mw-parser-output"})[0].find_all(['p','ul','li']):
        text = tag.get_text()
        parents = [parent for parent in tag.parents]
        grandParents = [parent for parent in parents[0].parents]
        usable = True
        for parent in parents:
            if parent.has_attr('class') and "toc" in parent['class']:
                usable = False
                break
            if parent.has_attr('class') and "references-small" in parent['class']:
                usable = False
                break
            if parent.has_attr('id') and "Notes_and_references" in parent['id']:
                usable = False
                break
            if parent.has_attr('id') and "p-lang" in parent['id']:
                usable = False
                break
        for grandParent in grandParents:
            if grandParent.has_attr('class') and "toc" in grandParent['class']:
                usable = False
                break
            if grandParent.has_attr('id') and "Notes_and_references" in grandParent['id']:
                usable = False
                break
            if grandParent.has_attr('class') and "references-small" in grandParent['class']:
                usable = False
                break
            if grandParent.has_attr('id') and "p-lang" in grandParent['id']:
                usable = False
                break
        if usable and text not in content:
            content += text
    return content

def combineContent():
    # take all of the files in AllInfo and combine them into one file by separating them with two newlines
    files = os.listdir("AllInfo")
    content = ""
    for file in files:
        f = open("AllInfo/"+file,"r")
        content += f.read()
        content += "\n\n"
        f.close()
    f = open("CompiledInfo.txt","x")
    f.write(content)
    f.close()

def main():
    # Read in the csv file
    df = pd.read_csv("allpages.csv")
    # iterate through the links and extract the content, writing them to a file in AllInfo, with the name of the file being the name of the page
    for index, row in df.iterrows():
        # print(row['links'])
        content = extractContent(row['links'])
        # strip out all double or more newlines, replacing them with a single newline
        content = content.replace("\n\n\n","\n")
        content = content.replace("\n\n","\n")
        #print(content)
        # write the content to a file, with the name of the file being the name of the page after the last / in the url
        filename = "AllInfo/"+row['links'].split("/")[-1]+".txt"
        # if the file exists, delete it
        try:
            f = open(filename,"r")
            f.close()
            os.remove(filename)
        except:
            pass
        if content != "":
            try:
                f = open(filename,"x")
                f.write(content)
                f.close()
                print("Wrote "+filename)
            except Exception as e:
                print(e)
                print("Error writing "+filename)
        else:
            print("No content for "+filename)

    # combine all of the files into one file
    combineContent()

if __name__ == "__main__":
    main()