import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

class Scrapper:
    """
    Scrapper class to fetch all the reviews from the amazon product page
    """
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

    def get_all_reviews(self, url):
        """
        Fetches all the reviews by looping over different review pages

        Args:
            url (string): url for the product page
        """
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
            print("Fetching reviews from Page {}".format(page_id))
            reviews_nxt_page_link = all_reviews_link + "&pageNumber=" + str(page_id)
            self.get_customer_reviews(reviews_nxt_page_link)
        
        print("Total reviews fetched : {}".format(len(self.reviews["customer_name"])))

    def scrap_url(self, url):
        """
        Processes the url to fetch the html file

        Args:
            url (string): url for the product page

        Returns:
            BeautifulSoup: BeautifulSoup object with html parsed
        """
        html_data = self.get_html_data(url)
        b_sup = BeautifulSoup(html_data, 'html.parser')
        return b_sup

    def get_html_data(self, url):
        """
        makes url request to get html

        Args:
            url (string): url for the product page

        Returns:
            text: html as text
        """
        req = requests.get(url, headers=self.HEADERS)
        return req.text
    
    def get_total_reviews_count(self, url):
        """
        to fetch total number of amazon reviews

        Returns:
            int: total number of amazon reviews
        """
        b_sup = self.scrap_url(url)
        total_reviews_count = b_sup.find('div', attrs={'data-hook':'cr-filter-info-review-rating-count'})
        count = list(total_reviews_count.stripped_strings)[0].split(" ")[3]
        print("Total number of reviews: {}".format(int(count)))
        return int(count)
    
    def get_customer_reviews(self, url):
        """
        loops through the html tags to get the review data and stores it
        """
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
        """
        removes unwanted text from the string

        Args:
            txt (string): ratings text from html tag

        Returns:
            string: ratings
        """
        return txt.split(" ")[0]
    
    def process_review_date(self, date_txt):
        """
        removes unwanted text from the string

        Args:
            date_txt (string): html date string

        Returns:
            string: date of review
        """
        date_txt = date_txt.split(" ")[::-1]
        date_txt = date_txt[:3]
        date_txt[1] = date_txt[1].replace(',', "")
        return " ".join(date_txt)
    
    def save_as_csv(self, file_name):
        """
        saves the reviews data to csv file in data folder

        Args:
            file_name (string): csv file name
        """
        reviews_df = pd.DataFrame(self.reviews)
        reviews_df.to_csv(file_name, index=False)        
        file_size = Path(file_name).stat().st_size
        print("Saving data to csv file, {} , Memory {}mb".format(reviews_df.shape, file_size/pow(10, 6)))  

import os
import glob
import pandas as pd


if __name__ == "__main__":
    urls = [" https://www.amazon.com/Moto-Alexa-Hands-Free-camera-included/productreviews/B07N9255CG?ie=UTF8&reviewerType=all_reviews"]
    # "https://www.amazon.com/Ninja-NJ601AMZ-Professional-1000-Watt-Dishwasher-Safe/dp/B098RD17LG?ref_=Oct_DLandingS_D_929b9bab_60&smid=ATVPDKIKX0DER", 
    # "https://www.amazon.com/Sony-PlayStation-Pro-1TB-Console-4/dp/B07K14XKZH/", 
    i = 1
    for url in urls:
        scrapper = Scrapper()
        scrapped_html = scrapper.get_all_reviews(url)
        scrapper.save_as_csv("data/reviews_data"+str(i)+".csv")
        i+=1
    
    os.chdir("/home/pranav/Fall22/809K/Web-Scraping-And-Contextual-Analysis-of-Reviews/data")
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    #export to csv
    combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
    # # reviews_df = pd.read_csv('/content/drive/MyDrive/Project_809K/data/reviews_data.csv')
