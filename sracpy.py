import requests
import json
import time

# Base endpoint
BASE_URL = "https://collegedunia.com/web-api/nc/global-search"

# Headers to mimic a real browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

# Search terms
SEARCH_TERMS = ["IIT", "NIT", "BITS", "IIIT", "AIIMS", "VIT", "SRM", "Manipal", "Technological", "Engineering"]

# Store data
DATASET = {}
DATASET_URLS = []

def fetch_college_data(term):
    params = {
        "page_type": "in",
        "countryId": 2,
        "item_type": "college",
        "term": term
    }

    response = requests.get(BASE_URL, headers=HEADERS, params=params)
    url = response.url
    DATASET_URLS.append(url)
    print(f" Fetching data from: {url}")

    if response.status_code == 200:
        data = response.json()
        output = data.get("output", [])

        for item in output:
            if item.get("item_type") == "college":
                college_id = item.get("entity_id")
                if college_id not in DATASET:
                    DATASET[college_id] = {
                        "name": item.get("name"),
                        "rating": item.get("rating"),
                        "score": item.get("score"),
                        "url": f"https://collegedunia.com/{item.get('url', '')}",
                        "logo": item.get("logo")
                    }
    else:
        print(f"Error {response.status_code} for term: {term}")

def main():
    print("Starting college data collection...\n")

    for term in SEARCH_TERMS:
        fetch_college_data(term)
        time.sleep(2)

    print(f"\n Total colleges fetched: {len(DATASET)}")

    with open("college_dataset.json", "w") as f:
        json.dump(DATASET, f, indent=4)

    with open("college_dataset_urls.json", "w") as f:
        json.dump({"urls": list(set(DATASET_URLS))}, f, indent=4)

    print("\n Data saved to 'college_dataset.json' and 'college_dataset_urls.json'.")

if __name__ == "__main__":
    main()