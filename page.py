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
    # for i <= 10: Time.sleep() # user-agent
    
    # data = output.get('props')
    # dataInitial = data.get('intit')
    response = requests.get(PAGE_URL, headers=HEADERS)
    
    if response.status_code == 200:
            data = response.text
            src = pq(data)
            jsonData = src.find('script:last')
            
            final = pq(jsonData).text()
            print(final)
            print(type(final))
            
            # json create using  final
            
            with open("ind_college.json", "w", encoding="utf-8") as f:
                json.dump(json.loads(final), f, indent=4)
            # with open("ind_college.html", "w") as f:
            #     f.write(data)
            
    else:
        print("Page not loaded")

if __name__ == "__main__":
    main()
