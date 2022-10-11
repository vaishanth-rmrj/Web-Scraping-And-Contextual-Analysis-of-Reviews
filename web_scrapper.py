import requests
from bs4 import BeautifulSoup




class Scrapper:
    def __init__(self):
        self.HEADERS = ({'User-Agent':'Mozilla/5.0 \
                        (Windows NT 10.0; Win64; x64) \
                        AppleWebKit/537.36 (KHTML, like Gecko) \
                        Chrome/90.0.4430.212 Safari/537.36',
                        'Accept-Language': 'en-US, en;q=0.5'})

        # variables
        self.reviews = {}

    def scrap_url(self, url):
        html_data = self.get_html_data(url)
        b_sup = BeautifulSoup(html_data, 'html.parser')
        return b_sup

    def get_html_data(self, url):
        req = requests.get(url, headers=self.HEADERS)
        return req.text
    
    def get_customer_reviews(self, url):
        b_sup = self.scrap_url(url)
        for id, review_item in enumerate(b_sup.find_all("div", class_="a-section review aok-relative")):
            cust_name = review_item.find("span", class_="a-profile-name")
            cust_review = review_item.find("div", class_="review-text-content")
            cust_rating = review_item.find("i", class_="review-rating")
            # print(customer_name.text)
            # print(cust_review.text)
            self.reviews[str(id)] = {
                "customer_name":  cust_name.text,
                "customer_rating": cust_rating.text,
                "customer_review": cust_review.text,
            }
        
            print(self.reviews[str(id)])

if __name__ == "__main__":
    url = "https://www.amazon.com/Ninja-NJ601AMZ-Professional-1000-Watt-Dishwasher-Safe/dp/B098RD17LG?ref_=Oct_DLandingS_D_929b9bab_60&smid=ATVPDKIKX0DER"
  
    scrapper = Scrapper()
    scrapped_html = scrapper.get_customer_reviews(url)
    print(scrapped_html)
