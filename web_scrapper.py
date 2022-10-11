import requests
from bs4 import BeautifulSoup
import pandas as pd




class Scrapper:
    def __init__(self):
        self.HEADERS = ({'User-Agent':'Mozilla/5.0 \
                        (Windows NT 10.0; Win64; x64) \
                        AppleWebKit/537.36 (KHTML, like Gecko) \
                        Chrome/90.0.4430.212 Safari/537.36',
                        'Accept-Language': 'en-US, en;q=0.5'})

        # variables
        self.reviews = {
            "customer_name": [],
            "customer_rating": [],
            "review_date": [],
            "customer_review": [],
        }

    def scrap_url(self, url):
        html_data = self.get_html_data(url)
        b_sup = BeautifulSoup(html_data, 'html.parser')
        return b_sup

    def get_html_data(self, url):
        req = requests.get(url, headers=self.HEADERS)
        return req.text
    
    def get_customer_reviews(self, url):
        b_sup = self.scrap_url(url)
        cust_name_lst, cust_rating_lst, review_date_lst, cust_review_lst = self.reviews["customer_name"], self.reviews["customer_rating"],self.reviews["review_date"], self.reviews["customer_review"]
        for id, review_item in enumerate(b_sup.find_all("div", class_="a-section review aok-relative")):
            cust_name = review_item.find("span", class_="a-profile-name")
            cust_review = review_item.find("span", class_="review-text-content")
            cust_rating = review_item.find("i", class_="review-rating")
            review_date = review_item.find('span', attrs={'data-hook':'review-date'})

            # processing ratings text
            cust_rating = self.process_ratings_txt(cust_rating.text)
            review_date = self.process_review_date(review_date.text)

            cust_name_lst.append(cust_name.text)
            cust_rating_lst.append(cust_rating)
            review_date_lst.append(review_date)
            if cust_review is not None:
                cust_review_lst.append(cust_review.text)  
            else:
                cust_review_lst.append("None")  

        self.reviews["customer_name"] = cust_name_lst
        self.reviews["customer_rating"] = cust_rating_lst
        self.reviews["review_date"] = review_date_lst
        self.reviews["customer_review"] = cust_review_lst
        

        
    def process_ratings_txt(self, txt):
        return txt.split(" ")[0]
    
    def process_review_date(self, date_txt):
        date_txt = date_txt.split(" ")[::-1]
        date_txt = date_txt[:3]
        date_txt[1] = date_txt[1].replace(',', "")
        return " ".join(date_txt)
    
    def save_as_csv(self, file_name):
        reviews_df = pd.DataFrame(self.reviews)
        print(reviews_df)
        reviews_df.to_csv(file_name, index=False)
    
    def get_total_reviews_count(self, url):
        b_sup = self.scrap_url(url)
        total_reviews_count = b_sup.find('div', attrs={'data-hook':'cr-filter-info-review-rating-count'})
        count = list(total_reviews_count.stripped_strings)[0].split(" ")[3]
        return int(count)
    
    def get_all_reviews(self, url):
        b_sup = self.scrap_url(url)
        all_reviews_link = b_sup.find('a', attrs={'data-hook':'see-all-reviews-link-foot'})
        all_reviews_link = "https://www.amazon.com" + all_reviews_link['href']

        # get the total reviews count to calc number of pages
        total_reviews_count = self.get_total_reviews_count(all_reviews_link)
        num_pages = (total_reviews_count // 10) + 1

        # get first page reviews
        self.get_customer_reviews(all_reviews_link)
        
        # loop through all review pages
        for page_id in range(2, num_pages+1):
            reviews_nxt_page_link = all_reviews_link + "&pageNumber=" + str(page_id)
            self.get_customer_reviews(reviews_nxt_page_link)

        

if __name__ == "__main__":
    url = "https://www.amazon.com/Ninja-NJ601AMZ-Professional-1000-Watt-Dishwasher-Safe/dp/B098RD17LG?ref_=Oct_DLandingS_D_929b9bab_60&smid=ATVPDKIKX0DER"
  
    scrapper = Scrapper()
    scrapped_html = scrapper.get_all_reviews(url)
    scrapper.save_as_csv("data/reviews.csv")
    # print(scrapped_html)
