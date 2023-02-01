# Web-Scraping-And-Contextual-Analysis-of-Reviews
A Natural Language Processing (NLP) application that analyzes customer reviews on Amazon and provides a recommendation score based on the sentiment of the reviews. The application reduces the time customers spend reading through reviews and helps them make an informed choice.

## Project Objective 
The goal of this project is to develop an application that can analyze reviews posted by numerous customers on Amazon for a product and make recommendations based on those reviews.

## Algorithm Flow Diagram
The project consists of two phases: the training phase and the prediction phase.

1. Training Phase
- Pre-process the Amazon reviews that have been extracted.
- Clean and map the data.
- Split the dataset into train, validation, and test data.
- Pass the training and validation data through a Pre-trained BERT tokenizer.
- Pass the data through an implementation of the BERT model.
- Calculate the loss and update the weights.
- Store the model with the best accuracy.

<img src="https://github.com/vaishanth-rmrj/Web-Scraping-And-Contextual-Analysis-of-Reviews/blob/main/extras/training_flow.png" width=400 height=600/>

2. Prediction Phase
- Use the best classifier model to generate predictions on users' sentiments towards a particular product.
- Generate a product recommendation score based on the average predicted sentiment of all reviews for the searched product.
- Pass the positive classified reviews through a Pegasus Tokenizer to generate a summary of all classified reviews.

<img src="https://github.com/vaishanth-rmrj/Web-Scraping-And-Contextual-Analysis-of-Reviews/blob/main/extras/prediction_flow.png" width=400 height=600/>

3. GUI interaction with backend.
<img src="https://github.com/vaishanth-rmrj/Web-Scraping-And-Contextual-Analysis-of-Reviews/blob/main/extras/gui_flow.png" width=400/>

## GUI Implementation
- A GUI has been developed that allows users to search for a particular product on Amazon.
- The GUI displays the positive sentiment score which indicates how positively received the product is on Amazon.
- It also provides a summary of all extracted reviews from the Amazon URL.

<img src="https://github.com/vaishanth-rmrj/Web-Scraping-And-Contextual-Analysis-of-Reviews/blob/main/extras/gui.png" width=500/>

## Conclusion

In conclusion, the Web-Scraping-And-Contextual-Analysis-of-Reviews project presents a powerful solution for customers who wish to make informed choices when shopping on Amazon. By automating the process of analyzing product reviews, the application saves time and provides a clear recommendation score based on the sentiment of the reviews. This project demonstrates the capability of NLP algorithms in processing large amounts of text data and generating meaningful insights. The implementation of BERT and Pegasus Tokenizer makes the sentiment analysis robust and accurate, while the GUI provides an intuitive and user-friendly interface. Overall, the project provides a valuable tool for customers who are looking to make informed purchase decisions on Amazon.




