from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.imdb.com/title/tt16311594/reviews/?ref_=tt_ov_ql_2"

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get(url)

time.sleep(10)

last_count = 0

while True:
    try:
        load_more = driver.find_element(By.XPATH, '//button[contains(text(),"Load More")]')
        driver.execute_script("arguments[0].click();", load_more)
        time.sleep(9)
    except:
        pass

    reviews = driver.find_elements(By.CLASS_NAME, "ipc-list-card__content")
    current_count = len(reviews)

    print(f"Review loaded: {current_count}")

    if current_count == last_count:
        break

    last_count = current_count
    time.sleep(9)

print("Semua review sudah muncul")

driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(12)

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

reviews = soup.find_all('div', class_="ipc-list-card__content")

data = []

for review in reviews:
    title = review.find('h3')
    title = title.text.strip() if title else None

    rating = review.find('span', class_="ipc-rating-star--rating")
    rating = rating.text.strip() if rating else None

    content = review.find('div', class_="ipc-html-content-inner-div")
    content = content.get_text(strip=True) if content else None

    parent = review.find_parent('article')

    author = parent.find('a', {'data-testid': 'author-link'})
    author = author.text.strip() if author else None

    date = parent.find('li', class_="review-date")
    date = date.text.strip() if date else None

    data.append({
        "title": title,
        "rating": rating,
        "content": content,
        "author": author,
        "date": date
    })

driver.quit()

# 💾 SAVE CSV
df = pd.DataFrame(data)
df.to_csv("imdb_reviews_full.csv", index=False, encoding='utf-8-sig')

print("🔥 Total review:", len(data))
print("💾 File: imdb_reviews_full.csv")

df.head()