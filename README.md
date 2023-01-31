# Web-Scraping-And-Contextual-Analysis-of-Reviews
Implementation of NLP algorithm to analyse the product reviews

### Project Objective 
The goal of this project is to develop an application that can analyze reviews posted by numerous customers on Amazon for a product and make recommendations based on those reviews. When a customer searches for a product on Amazon, they take a lot of time reading the reviews and determining whether the item is as good as claimed. In order to help the customer make an informed choice, our application would reduce that time by automatically recommending the product to them.

### Algorithm Flow Diagram
1. Training Phase
In our prediction pipeline, we first pre-process the amazon reviews that have been extracted, then clean and map. We then split the dataset into train validation and test data. The training and validation data is passed through a Pre-trained Bert tokenizer and Then passed through our implementation of the BERT model, where the loss is calculated and the weights are updated. Once our model is done training we use the test data to evaluate our accuracy. Over multiple epochs, the model which achieves the best accuracy is then stored.

<img src="https://github.com/vaishanth-rmrj/Web-Scraping-And-Contextual-Analysis-of-Reviews/blob/main/extras/training_flow.png" width=400 height=600/>

2. Prediction Phase
The best classifier model is used to generate a prediction on users’ sentiments towards a particular product, in the Fig: we can see how the best classification model from our prediction pipeline is used to generate a Product recommendation score which is based on the average predicted sentiment of all the reviews of the searched product.Furthermore, the positive classified reviews are then passed through a Pegasus Tokenizer which generates a summary of all the classified reviews.

<img src="https://github.com/vaishanth-rmrj/Web-Scraping-And-Contextual-Analysis-of-Reviews/blob/main/extras/prediction_flow.png" width=400 height=600/>

3. GUI interaction with backend.
<img src="https://github.com/vaishanth-rmrj/Web-Scraping-And-Contextual-Analysis-of-Reviews/blob/main/extras/gui_flow.png" width=400/>

### GUI Implementation

We have developed a GUI that allows users to search for a particular product on amazon. Then it displays the positive sentiment score which indicates how positively received is the amazon listing. The GUI also provides a summary of all the extracted reviews from the amazon URL.

<img src="https://github.com/vaishanth-rmrj/Web-Scraping-And-Contextual-Analysis-of-Reviews/blob/main/extras/gui.png" width=500/>




