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

Described in the document supporting this implementation, most of the models were trained using 2017_1 and tested on 2017_2 and 2019_1.

# Results

predictions.csv - This csv file contains the input data observation and output predictions from distilBert model with the final layer as LORA.


# Python Notebooks

In reference to the document the following notebooks were used to model the desired classification task

**3.1. Topic Modeling for Exploratory Data Analysis**  
[LDA_political_leanings.ipynb](./LDA_political_leanings.ipynb) - This contains the method for initial data exploration using LDA to understand the word distributions between different labels.  

**4.1. Naive Bayes Classifier**  
[NaiveBayes.ipynb](./NaiveBayes.ipynb) - This is our base model Naive Bayes for classification.  

**4.2.1 distilBERT embeddings and Cosine Similarity classification head**  
[CosineSimilarity.ipynb](./CosineSimilarity.ipynb) - This is the cosine similarity final head for classification on top of the embeddings from distilBERT.

**4.2.2. Fully connected layer with neural networks**  
[distilBERT_NN.ipynb](./distilBERT_NN.ipynb) - This is distilBERT with a Neural Network as the final layer for classification.

**4.2.3 LoRA(Low-Rank Adaptation) Integration with distilBERT**  
[distilBERT_LoRA_body.ipynb](./distilBERT_LoRA_body.ipynb) - This model uses distilBERT with LORA to classify full texts across the four classes.  

**5.1. Using headlines, lead and summarization**  
[distilBERT_LoRA_headline.ipynb](./distilBERT_LoRA_headline.ipynb) - This model uses distilBERT with LORA to classify headlines.  
[distilBERT_LoRA_lead.ipynb](./distilBERT_LoRA_lead.ipynb) - This model uses distilBERT with LORA to classify the lead across the four labels.  
[distilBERT_LoRA_summary.ipynb](./distilBERT_LoRA_summary.ipynb) - This model uses distilBERT with LORA to classify a summary of the text across the four labels.  

**5.2 Filtering Left-Right classes only**  
[distilBERT_LoRA_body_LRonly.ipynb](./distilBERT_LoRA_body_LRonly.ipynb) - This model uses distilBERT with LORA to classify full texts as either Left or Right (not four classes).  
[distilBERT_LoRA_healine_LRonly.ipynb](./distilBERT_LoRA_healine_LRonly.ipynb) - This model uses distilBERT with LORA to classify headlines as Left or Right only (not four classes).  



#Final pretrained models
The pretrained models can be found in this [directory](https://duke.box.com/s/rc2h1fhsj5a59q42i1pijh64ikczhxeg). 

# Additional Analysis
the **analysis** folder contains additional analysis, models and results we used to develop out methodology. It can also be used to generate specific results for models.
