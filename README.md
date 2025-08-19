#  NLP Capstone: Sentiment Analysis & Topic Modeling for BikeEase

**Author:** Carllos Watts-Nogueira  
**Course:** Natural Language Processing  
**Due Date:** August 23, 2025  
**Focus:** Sentiment Classification · Topic Modeling · Model Deployment  

---

##  Badges

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) 
![NLP](https://img.shields.io/badge/NLP-Sentiment%20%26%20Topics-purple.svg) 
![Traditional Models](https://img.shields.io/badge/Models-LogReg%2C%20NaiveBayes-blue.svg)
![Deep Learning](https://img.shields.io/badge/Models-LSTM%2C%20BERT-purple.svg)
![Deployment](https://img.shields.io/badge/Deployment-Joblib%20%2F%20Pipeline-green.svg) 
![Status](https://img.shields.io/badge/Stage-Capstone%20Complete-brightgreen.svg)

---

##  Project Overview

This capstone project analyzes customer reviews from **BikeEase**, a bike rental platform, using Natural Language Processing (NLP). The goal is to:

- Classify sentiment (positive, neutral, negative)  
- Extract key themes from reviews  
- Help BikeEase improve customer experience through data-driven insights  

---

##  Objectives

- Build an end-to-end NLP pipeline  
- Perform sentiment classification using traditional and pretrained models  
- Apply topic modeling to uncover customer pain points  
- Evaluate model performance and robustness  
- Save and deploy the best-performing model  

---

##  Dataset

Customer reviews collected from multiple platforms.  
 [Access the dataset](https://drive.google.com/drive/folders/13-g3jxhPR0btN_s77KbcempcFXZ0RoqT)

---

##  Models Used

### Traditional Machine Learning
- Logistic Regression  
- Naive Bayes  

### Pretrained NLP Models
- TextBlob  
- VADER  
- Flair  
- BERT (via Hugging Face Transformers)

### Deep Learning
- LSTM (Bidirectional, with dropout and batch normalization)

---

##  Preprocessing Steps

- Lowercasing  
- Punctuation removal  
- Tokenization  
- Lemmatization  
- Stopword removal  
- TF-IDF vectorization  

---

##  Evaluation Metrics

- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  
- Cross-Validation  

---

##  Topic Modeling

- **Latent Dirichlet Allocation (LDA)**  
- Interactive visualization with **pyLDAvis**

---

##  Model Saving

- Traditional models and TF-IDF vectorizer saved using `joblib`
- LSTM model saved in native Keras format (`.keras`)
- Tokenizer and label encoder saved as `.pkl` files
- Training history saved for performance tracking

---

##  Results Summary

| Model               | Accuracy | Neutral F1 | Notes                                 |
|---------------------|----------|------------|----------------------------------------|
| Naive Bayes         | 1.00     | 1.00       | Perfect scores on clean data           |
| Logistic Regression | 1.00     | 1.00       | Same as Naive Bayes                    |
| LSTM                | 1.00     | 1.00       | Deep learning model with perfect scores|
| TextBlob            | 0.67     | 0.28       | Fast, but weak on neutral              |
| VADER               | 0.67     | 0.43       | Better, but still biased               |
| Flair               | 0.67     | 0.00       | Ignores neutral class                  |
| BERT                | 0.67     | 0.00       | Powerful, but needs fine-tuning        |
---

## LSTM Insights
- Achieved perfect classification across all sentiment classes
- Early convergence with extremely low loss
- Sudden drop in validation accuracy at epoch 4 suggests potential overfitting or data anomaly
- Highlights importance of monitoring both training and validation metrics

##  Reflections

- Neutral sentiment is the hardest to classify  
- Pretrained models require fine-tuning for 3-class tasks  
- Traditional models excel on clean, balanced datasets  
- Error analysis and cross-validation are essential for robustness

## Evaluation Tools
- `sklearn.metrics` for precision, recall, F1-score
- Confusion matrices visualized with `seaborn` and `matplotlib`

---

##  Future Work

- Fine-tune BERT for improved neutral detection  
- Build a Streamlit or Flask app for real-time sentiment prediction  
- Explore ensemble methods to combine model strengths  

---

##  Contact

**Author:** Carllos Watts-Nogueira  
**Email:** [carlloswattsnogueira@gmail.com](mailto:carlloswattsnogueira@gmail.com)  
**LinkedIn:** [linkedin.com/in/carlloswattsnogueira](https://www.linkedin.com/in/carlloswattsnogueira/)

---


