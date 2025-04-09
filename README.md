# Political Leaning Classification with BERT-based Models

## Project Overview
This project explores different approaches for classifying the political leaning of news articles using BERT-based architectures, with a focus on DistilBERT for its versatility and computational efficiency. The implementation compares several classification methods:

- Naive Bayes (baseline)
- DistilBERT with cosine similarity classification
- DistilBERT with neural network fully connected layers
- DistilBERT with Low-Rank Adaptation (LoRA)

## Dataset
The project utilizes the POLUSA Dataset, containing approximately 0.9 million political news articles from 2017-2019. Articles are classified into four political leaning categories:
- LEFT
- RIGHT
- CENTER
- UNDEFINED

Features used include the article's main body text, headline, and political leaning classification. Data from 2017 was primarily used for training, with testing performed on both 2017 and 2019 data to evaluate model performance across time periods.

## Methodology

### 1. Naive Bayes Classifier
Serves as a baseline model, providing a simple yet interpretable framework for text classification.

### 2. DistilBERT with Cosine Similarity
Uses pre-trained DistilBERT to generate document embeddings, then classifies based on cosine similarity between the document embedding and mean class embeddings.

### 3. DistilBERT with Neural Network Layers
Enhances the model by adding fully connected neural network layers to the DistilBERT backbone to capture complex relationships between embeddings and classifications.

### 4. DistilBERT with LoRA
Integrates Low-Rank Adaptation (LoRA) into the DistilBERT architecture, optimizing the attention mechanism with trainable query layers while significantly reducing computational costs.

## Experiments and Results

### Overall Model Performance
| Model | Accuracy on 2017 | Accuracy on 2019 |
|-------|------------------|------------------|
| Naive Bayes | 0.70 | 0.53 |
| DistilBERT + Cosine Similarity | 0.45 | 0.37 |
| DistilBERT + FC Neural Network | 0.65 | 0.42 |
| DistilBERT + LoRA | 0.91 | 0.79 |
| DistilBERT + LoRA (LEFT/RIGHT only) | 0.92 | 0.91 |

### Alternative Data Configurations
The project also explored classification using limited text inputs:
- Headlines only
- Leading paragraphs only
- Summarized text
- Binary classification (LEFT/RIGHT only)

### Key Findings
1. DistilBERT with LoRA delivered the best performance, achieving 91% accuracy on 2017 data and 79% accuracy on 2019 data.
2. Binary classification (LEFT/RIGHT only) further improved accuracy to 92% on 2017 data and 91% on 2019 data.
3. Models performed better with full article text than with headlines or summarized content.
4. Performance generally decreased when testing on 2019 data, suggesting temporal data drift.

## Implementation Details

### DistilBERT + LoRA Architecture
- **Tokenizer**: Maximum length 512 with padding
- **DistilBERT Encoder**: Pretrained weights frozen during training
- **LoRA Configuration**: Applied to query projection (q_lin), rank=4, alpha=32
- **Training**: Learning rate 1e-3, batch size 10, 10 epochs
- **Optimization**: Adam optimizer with weight decay 0.01

### Preprocessing
- Text standardization to lowercase
- De-contractions of common English contractions
- Removal of non-ASCII characters
- Filtering URLs and emojis
- Class weight balancing to address imbalanced data

## Future Work
1. Exploring headline-specific performance challenges
2. Testing alternative architectures beyond DistilBERT
3. Investigating more advanced fine-tuning techniques
4. Addressing data imbalances more effectively
5. Comparing with other pre-trained models like RoBERTa

## Requirements
- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- PEFT (for LoRA implementation)
- scikit-learn
- numpy
- pandas

# Trained models
Trained model(Nov.19)

(BERT trained with BODY) https://duke.box.com/s/6clw9gx2vqpu26s4p7yh7z64gg3vi8yg

(BERT trained with HEADLINE) https://duke.box.com/s/8bax76my1wfbdu2715xtrp662zu6fl72

(BERT trained with SUMMARY) https://duke.box.com/s/zq5bu72d29k83tbuen5b24deqojzhiaj
 -> The accuracy is 62%. Using 10% (15K lines) of the full dataset (150K)
    Summarization was conduced with the T-5 'small' model. Would using the T-5 'Base' model improve the results......?

## Citation
If you use this code or methodology in your research, please cite:
```
Chen, J., Lee, I., Maddhuri, J., & Paredes, A. (2025). Political Leaning in News with BERT.
```

References:

https://ground.news/

https://www.kaggle.com/code/mikiota/data-augmentation-csv-txt-using-back-translation

https://www.kaggle.com/code/nkitgupta/text-representations

https://huggingface.co/docs/transformers/en/tasks/summarization
