from classifier.utils import load_data, encode_text
from web_scrapper import Scrapper


if __name__ == "__main__":

    # params
    mode = "eval" # mode-> ['train', 'eval']
    train_data_dir = "datasets/train"
    test_data_dir = "datasets/test/"

    # test_df = load_data(test_data_dir)

    # sample_txt = test_df.iloc[2, 1]

    # encoded_text, attention_mask = encode_text(sample_txt, max_encoding_len=128)

    # print(encode_text)

    # webscrapping
    # url = "https://www.amazon.com/Nine-West-Womens-Silver-Tone-Black/dp/B0721SGTGY/?_encoding=UTF8&pd_rd_w=6eJqF&content-id=amzn1.sym.e4bd6ac6-9035-4a04-92a6-fc4ad60e09ad&pf_rd_p=e4bd6ac6-9035-4a04-92a6-fc4ad60e09ad&pf_rd_r=TFW8DVB8ETT6J8CWEAP6&pd_rd_wg=TLy4F&pd_rd_r=2b9d59aa-aba1-431c-86e8-085e83989c2f&ref_=pd_gw_ci_mcx_mr_hp_atf_m&th=1"
  
    # scrapper = Scrapper()
    # scrapped_html = scrapper.get_all_reviews(url)
    # scrapper.save_as_csv("cache/scrapped_data/reviews.csv")

    review_df = 




