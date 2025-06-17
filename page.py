import requests
from pyquery import PyQuery as pq
import json
import time

PAGE_URL = "https://collegedunia.com/college/13730-hitkarini-college-of-engineering-and-technology-hcet-jabalpur"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36",
    "Accept": "application/json"
}


def main():
    response = requests.get(PAGE_URL, headers=HEADERS)
    
    if response.status_code == 200:
            data = response.text
            src = pq(data)
            jsonData = src.find('script:last')
            
            final = pq(jsonData).text()
            print(final)
            print(type(final))
            
            with open("ind_college.json", "w", encoding="utf-8") as f:
                json.dump(json.loads(final), f, indent=4)
            # with open("ind_college.html", "w") as f:
            #     f.write(data)
            
    else:
        print("Page not loaded")

if __name__ == "__main__":
    main()
