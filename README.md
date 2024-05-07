# Sentiment_Analysis_IMDB_Reviews

Project Objective : To analyze a dataset of IMDB reviews and develop a sentiment analysis model that accurately classifies the sentiment of each review as positive or negative. The goal is to explore different techniques such as natural language processing (NLP) and machine learning to achieve high accuracy in sentiment classification.

Data Sources: This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. .Link(https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Libraries : Pandas, Numpy, SK-learn, Streamlit, NLTK(Word_tokenization, Stopwords_removal, Stemming, Text_vectorization)

Text Preprocessing: Here, on the review column we applied word tokenization, stopwords removal, and lemmatization to clean the data after that we applied text vectorization.

Model Building: We have created multiple models-

Model Name - Logistics regression | Accuracy - 0.8933 | Precision - 0.8935 |

Model Name - Decision tree classifier | Accuracy - 0.717 | Precision - 0.717 |

Model Name - RandomForestClassifier | Accuracy - 0.8565 | Precision - 0.8566 |

Model Selection: As we can see there are 2 models which are performing very well in terms of precision RandomForestClassifier and Logistics regression but when it comes to accuracy we will go ahead with Logistics regression.

Here are the screenshot of the live app classifying about Positive & Negative reviews -

![image](https://github.com/deependra-eng/Sentiment_Analysis_IMDB_Reviews/assets/56891041/b797b2ba-8890-4ce9-91b1-74974a145f4f)
![image](https://github.com/deependra-eng/Sentiment_Analysis_IMDB_Reviews/assets/56891041/165aa5c3-cf8b-4341-bf9d-fd331b5cb0eb)

