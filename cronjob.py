import requests
import time

# List of your API URLs
api_urls = [
    'https://farmplanner-ra1x.onrender.com',
    'https://plantcure.onrender.com',
    'https://agriyieldpro.onrender.com',
]

print("Starting API pings...")

for url in api_urls:
    try:
        response = requests.get(url)
        # You can add a check for the status code
        response.raise_for_status()
        print(f"✅ Successfully pinged {url} (Status: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to ping {url}: {e}")
    # Optional: Add a small delay between requests
    time.sleep(1)

print("Finished API pings.")

