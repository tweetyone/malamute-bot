import os,requests,json
from io import StringIO

try:
    while 1:
        searchQuery = str(input("enter a search query : "))
        print ("\n")
        wikiLink = "https://en.wikipedia.org/w/api.php?format=json&action=query&generator=search&gsrnamespace=0&gsrlimit=10&prop=pageimages|extracts&pilimit=max&exintro&explaintext&exsentences=1&exlimit=max&gsrsearch="+searchQuery
        wikiId = "https://en.wikipedia.org/?curid="
        jsonReq = requests.get(wikiLink).content.decode('utf-8')
        io = StringIO(jsonReq)
        jsonLoad = json.load(io)
        for a in jsonLoad["query"]["pages"] :

            urlId = jsonLoad["query"]["pages"][a]["pageid"]
            urlIdOpen = requests.get(wikiId+str(urlId))
            urlIdOpen = urlIdOpen.url
            title = jsonLoad["query"]["pages"][a]["title"]
            print ("\t\t" + title)

            print("\t\t" + str(urlId))

            urltitle = "_".join(title.split(" "))
            urlIdOpen = "https://en.wikipedia.org/wiki/"+urltitle
            print ("\t\t"+urlIdOpen)

            print ("\t\t"+jsonLoad["query"]["pages"][a]["extract"])
            print ("\n")

except KeyboardInterrupt:
    print ("GOOD BYE DUDE :)")

except:
    print ("something went wrong :(")
    print ("please re execute the program")
