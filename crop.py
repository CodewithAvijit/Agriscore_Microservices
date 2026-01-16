import requests
import time

# Your URLs
urls = [
    "https://farmplanner-ha0o.onrender.com/",
    "https://agriyieldpro-dy2t.onrender.com/",
    "https://plantcure-cen9.onrender.com/",
]

def ping_urls():
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            print(f"[OK] {url} -> {response.status_code}")
        except Exception as e:
            print(f"[ERROR] {url} -> {e}")

if __name__ == "__main__":
    while True:
        print("\n---- Pinging URLs ----")
        ping_urls()
        print("Sleeping for 10 minutes...\n")
        time.sleep(600)  # 600 seconds = 10 minutes
