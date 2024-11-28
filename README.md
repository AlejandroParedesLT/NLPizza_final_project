# Setup
To run the scripts and models in this folder without any problems, ensure to download all the necessary python libraries in requirements.txt. We suggest running notebooks on a machine that has high GPU computational power. We have used Kaggle and Google Colab to train, test and validate our models. The datasets are large and the models are complex, requiring more time to run through entire scripts.

# Datasets
2017_1.csv \
2017_2.csv
2018_1.csv
2018_2.csv
2019_1.csv
2019_2.csv

Each of the CSV files above contains the necessary data to train and test the models. Each year between 2017 and 2019 have two datasets with subsets of data from that year.

predictions.csv - This csv file contains the input data observation and output predictions from distilBert model with the final layer as LORA.

# Python Notebooks
BERT_NN.ipynb
BERT_PEFT.ipynb
BERT_PEFT_LoRA_BODY.ipynb


Trained model(Nov.19)

(BERT trained with BODY) https://duke.box.com/s/6clw9gx2vqpu26s4p7yh7z64gg3vi8yg

(BERT trained with HEADLINE) https://duke.box.com/s/8bax76my1wfbdu2715xtrp662zu6fl72

(BERT trained with SUMMARY) https://duke.box.com/s/zq5bu72d29k83tbuen5b24deqojzhiaj
 -> The accuracy is 62%. Using 10% (15K lines) of the full dataset (150K)
    Summarization was conduced with the T-5 'small' model. Would using the T-5 'Base' model improve the results......?


https://ground.news/

https://www.kaggle.com/code/mikiota/data-augmentation-csv-txt-using-back-translation

https://www.kaggle.com/code/nkitgupta/text-representations

https://huggingface.co/docs/transformers/en/tasks/summarization
