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
distilBERT_NN.ipynb
BERT_PEFT_LoRA_BODY.ipynb
