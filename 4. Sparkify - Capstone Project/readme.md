# Sparkify - Capstone Project

## Table of contents

- [Installation](#installation)
- [Tools](#Tools)
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Reflections](#reflections)
- [References](#references)


## Installation

In order to execute the Jupyter Notebook we should have **Anaconda 5.1** and **python 3.6** installed. 

- Clone the repo: `git clone https://github.com/devranjan/Udacity-Data-Scientist-Nanodegree.git`
- Run the Jupyter Notebook

## Tools
Python,
Spark,
PySpark,
pandas,
numpy,
matplotlib,
Seaborn. 

These are available as part of Anaconda installation and don't need any additional installation.

## Project Overview

Sparkify is a fictional music streaming app created by Udacity that allows its users to listen and manage their favorite music. Sparkify offers its services in two tiers. The free tier where users can use Sparkify services for free but songs are interspersed with commercials. The paid tier, on the other hand, is a premium service that plays ad-free music at a monthly subscription fee.

Users can add songs to their play list, thumb-up or thumb-down songs, add other users as friends (possibly for sharing playlists) along with upgrading, downgrading or cancelling subscription. Each of the action taken by user is added to an event log which contains event type, timestamp of the event along with user name,  subscription tier and other event specific details.

Each of the action taken by the user is added to an event log which contains event type, timestamp of the event along with user name, subscription tier, and other event-specific details. A user can contain many entries. In the data, a part of the user is churned, through the cancellation of the account behavior can be distinguished.

## Problem Statement
When users cancel the subscription or downgrade their account, it is a potential revenue loss for the business. Sparkify would like to avoid and minimize users canceling or downgrading the subscription.
The goal of this project is to apply data analysis and machine learning to predict if a user is at risk of cancelling the subscription. 

- The data is available in form of user events in json format
- Data needs pre-processing in form of
	- Clean up of missing or unusable data
	- Feature extraction to extract meaningful information
- On the next step, the pre-processed data is used to train several supervised machine learning model.
- The best suited model is then chosen to predict future data

## Project Structure
```text
Sparkify/
├── Sparkify.html
├── Sparkify.ipynb
├── mini_sparkify_event_data.json
└── README.md
```

- Sparkify.html                 --- Generated report from notebook
- Sparkify.ipynb                --- Notebook
- mini_sparkify_event_data.json --- Sparkify user event log in json format

## Reflections
- After feature engineering, I used the same dataframe for further test/train split and subsequent model training. It took long time even for a small amount of data. It seems very obvious now but it was interesting to realize that since Spark procrastinates the operations on DataFrames, training iterations were taking long as the entire transformation pipeline was run every time. Persisting the dataset with engineered features and reloading it quickened up the model training.

I post a blog about the complete detail, you may find it [here.](https://medium.com/@devranjan/how-did-i-predict-user-churn-using-pyspark-in-sparkify-music-app-ed334e2fb2c9)

## References
1. https://spark.apache.org/docs/latest/api/python/
1. https://mapr.com/blog/churn-prediction-pyspark-using-mllib-and-ml-packages/
1. https://www.kaggle.com/blastchar/telco-customer-churn
1. https://www.kaggle.com/c/kkbox-churn-prediction-challenge
