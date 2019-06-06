import requests
from bs4 import BeautifulSoup

FRONT = 'https://www.washington.edu'
web_url1 = requests.get('https://en.wikipedia.org/wiki/University_of_Washington').text
web_url2 = requests.get('https://www.washington.edu/').text
web_url3 = requests.get('https://www.washington.edu/students/gencat/degree_programsTOC.html').text #all the major information
soup = BeautifulSoup(web_url3, 'lxml')
count = 0
#print(soup.prettify())
for link in soup.findAll('a'):
    tail = link['href']
    if(tail.find('/students/gencat/academic') != -1):
        url = FRONT + tail
        web_url = requests.get(url).text
        soup_link = BeautifulSoup(web_url, 'lxml')
        content = soup_link.find('td', attrs={'class':'mainContent'})
        print(link.text,'-------------------flag--------------------------')
        print (content.text)
        with open("Output.txt", "a") as text_file:
            text_file.write(link.text)
            text_file.write(content.text)
