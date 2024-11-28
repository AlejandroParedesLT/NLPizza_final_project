# Setup
To run the scripts and models in this folder without any problems, ensure to download all the necessary python libraries in requirements.txt. We suggest running notebooks on a machine that has high GPU computational power. We have used Kaggle and Google Colab to train, test and validate our models. The datasets are large and the models are complex, requiring more time to run through entire scripts.

# Datasets
2017_1.csv \
2017_2.csv \
2018_1.csv \
2018_2.csv \
2019_1.csv \
2019_2.csv \
Each of the CSV files above contains the necessary data to train and test the models. Each year between 2017 and 2019 have two datasets with subsets of data from that year.

predictions.csv - This csv file contains the input data observation and output predictions from distilBert model with the final layer as LORA.

# Python Notebooks
**LDA_political_leanings.ipynb** - This contains the method for initial data exploration using LDA to understand the word distributions between different labels.

**NaiveBayes.ipynb** - This is our base model Naive Bayes for classification.

**distilBERT_NN.ipynb** - This is distilBERT with a Neural Network as the final layer for classification.
**CosineSimilarity.ipynb** - This is the cosine similarity final head for classification on top of the embeddings from distilBERT.

**distilBERT_LoRA_body_LRonly.ipynb** - This model uses distilBERT with LORA to classify full texts as either Left or Right (not four classes). \
**distilBERT_LoRA_body.ipynb** - This model uses distilBERT with LORA to classify full texts across the four classes. \
**distilBERT_LoRA_headline.ipynb** - This model uses distilBERT with LORA to classify headlines. \
**distilBERT_LoRA_healine_LRonly.ipynb** - This model uses distilBERT with LORA to classify headlines as Left or Right only (not four classes). \
**distilBERT_LoRA_lead.ipynb** - This model uses distilBERT with LORA to classify th lead across the four labels. \
**distilBERT_LoRA_summary.ipynb** - This model uses distilBERT with LORA to classify a summary of the text across the four labels. \

# Additional Analysis
the **analysis** folder contains additional analysis, models and results we used to develop out methodology. It can also be used to generate specific results for models.
