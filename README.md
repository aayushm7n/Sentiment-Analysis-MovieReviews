# Sentiment Analysis of Movie Reviews 🎬
> Author : Aayushman Singh Chandel
**Machine Learning Project** | December 2025

> Teaching computers to understand whether movie reviews are positive or negative using classical ML and cutting-edge deep learning.

---

## About

This is a project that tackles **sentiment analysis** - the task of automatically determining whether a piece of text expresses positive or negative sentiment. Specifically, I have built and compared **6 different models** (from simple to sophisticated) to classify 50,000 IMDb movie reviews.

### Relevance
- Companies use this tech everywhere (Netflix, Amazon, every streaming service)
- Demonstrates both classical ML fundamentals AND modern deep learning
- Real production considerations: not just "which is most accurate" but "which should I use when?"
- Great portfolio piece with actual depth to discuss in interviews

### Quick Navigation
- [What Problem We're Solving](#problem-statement) - The why and the what
- [The Dataset](#dataset-50000-imdb-reviews) - What we're working with
- [How to Run This](#quick-start-3-steps) - Get it running in minutes
- [The Notebooks](#the-4-notebooks) - Where the magic happens
- [Results Summary](#results-at-a-glance) - TL;DR of what worked
- [Honest Limitations](#limitations) - What doesn't work (yet)
- [Technical Details](#technical-deep-dive) - For the ML nerds

**Layman-readable version** Check out [PROJECT_STORY.md](PROJECT_STORY.md) for a more easy-going walkthrough.

---

## Problem Statement

### The Challenge
Can we teach a computer to read movie reviews and understand if they're positive or negative? Sounds simple, but:

- **Sarcasm exists**: "Oh great, another masterpiece" is actually negative
- **Context matters**: "not good" vs "good" - one word changes everything
- **People write differently**: From Shakespeare-level prose to "meh"
- **Nuance is hard**: Mixed feelings, backhanded compliments, cultural references

### What I Set Out to Do

**Primary Goal**: Build models that accurately classify reviews as positive or negative

**But also:**
- Compare 6 different approaches (classical ML through cutting-edge transformers)
- Understand the speed vs accuracy trade-offs
- Figure out which model you'd actually use in production
- Document everything so others can learn from it

### How We Measure Success

| Metric | What It Means | Why It Matters |
|--------|---------------|----------------|
| **Accuracy** | % of correct predictions | Overall performance |
| **F1-Score** | Balance of precision & recall | Handles false positives/negatives |
| **ROC-AUC** | How well model separates classes | Model discrimination ability |
| **Inference Time** | Speed per prediction | Production feasibility |

**Target**: Beat the baseline (random guessing = 50%) by a significant margin. Industry standard is 85%+.

---

## 📊 Dataset: 50,000 IMDb Reviews

### The Data
- **Source**: [Stanford's IMDb Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- **Size**: 50,000 movie reviews (25k training, 25k test)
- **Balance**: Perfect 50-50 split (no class imbalance issues)
- **Labels**: Binary - Positive (≥7/10 stars) or Negative (≤4/10 stars)
- **Language**: English
- **Format**: Plain text reviews with sentiment labels

**Variety in Length**:
- Short: "Loved it!" (2 words)
- Long: Multi-paragraph essays (2000+ words)
- Average: ~230 words per review

**Real-World Messiness**:
- HTML artifacts (`<br />` tags everywhere)
- Typos, slang, internet speak
- CAPSLOCK SHOUTING
- Sarcasm and nuance
- Mixed sentiments ("great acting but terrible plot")

**Why This Dataset?**
- Industry standard benchmark for sentiment analysis
- Large enough to train deep learning models
- Diverse vocabulary (~100k unique words)
- Real reviews from real people (not synthetic data)
- Challenging enough to be interesting, tractable enough to complete in 2 weeks

### Example Reviews

**Positive** ⭐⭐⭐⭐⭐:
> "This film was absolutely phenomenal! The cinematography was breathtaking, and the performances were outstanding. A must-watch masterpiece that will stay with you long after the credits roll."

**Negative** ⭐:
> "What a complete waste of time. Terrible acting, predictable plot, and painfully slow pacing. I can't believe I sat through the entire thing. Avoid at all costs."

---

## Quick Start (3 Steps)

### Prerequisites
- Python 3.9+ (tested with 3.14)
- 8GB RAM minimum (16GB recommended for BERT)
- GPU optional (makes deep learning training faster, but not required)

### Installation

```bash
# 1. Navigate to project directory
cd Sentiment-Analysis-MovieReviews

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Download NLTK data (one-time setup)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# 4. Launch Jupyter
jupyter notebook
```

### Run the Notebooks in Order

Open and run these notebooks sequentially:

1. **`01_eda_preprocessing.ipynb`** - Data exploration and cleaning (~15 min)
2. **`02_classical_ml_models.ipynb`** - Train 4 ML models (~20 min)
3. **`03_deep_learning_models.ipynb`** - LSTM + BERT (~1-2 hours with GPU, longer without)
4. **`04_model_comparison_results.ipynb`** - See all results (~10 min)

**That's it!** The notebooks are self-contained - just run cells top to bottom.

---

## The Notebooks

### Notebook 1: EDA & Preprocessing
**File**: `01_eda_preprocessing.ipynb`  
**Time**: ~15 minutes  
**What it does**:
- Loads 50k reviews from HuggingFace
- Visualizes sentiment distribution, review lengths
- Creates word clouds (positive vs negative vocabularies)
- Cleans text (removes HTML, lemmatizes, removes stopwords)
- Saves processed data for modeling

**Key outputs**: Clean CSV files, visualization charts, preprocessing insights

### Notebook 2: Classical ML Models
**File**: `02_classical_ml_models.ipynb`  
**Time**: ~20 minutes  
**What it does**:
- TF-IDF vectorization (convert text to numbers)
- Trains 4 models:
  - Logistic Regression (fast baseline)
  - Naive Bayes (probabilistic)
  - SVM (max-margin classifier)
  - Random Forest (ensemble)
- Evaluates each with accuracy, F1, ROC-AUC
- Compares performance side-by-side

**Key outputs**: 4 trained models, performance metrics, comparison charts

### Notebook 3: Deep Learning Models
**File**: `03_deep_learning_models.ipynb`  
**Time**: ~1-2 hours (with GPU)  
**What it does**:
- **LSTM**: Builds bidirectional LSTM from scratch with PyTorch
  - Custom vocabulary building
  - Embedding layer (128-dim)
  - 2-layer BiLSTM (256 hidden units)
  - 5 epochs of training
- **DistilBERT**: Fine-tunes pre-trained transformer
  - HuggingFace transformers library
  - AdamW optimizer, 2e-5 learning rate
  - 3 epochs of fine-tuning
- Full training loops with progress bars

**Key outputs**: LSTM model, fine-tuned BERT, training curves

### Notebook 4: Final Comparison & Results
**File**: `04_model_comparison_results.ipynb`  
**Time**: ~10 minutes  
**What it does**:
- Loads results from all 6 models
- Creates comprehensive comparison visualizations:
  - Bar charts, heatmaps, radar charts
  - Model rankings by F1-score
  - Classical vs Deep Learning comparison
- Production recommendations (when to use which model)
- Discusses limitations honestly
- Future work suggestions

**Key outputs**: Final comparison charts, insights document, recommendations

---

## Results

### Model Performance Summary

| Model | Accuracy | F1-Score | Speed | Memory | Best For |
|-------|----------|----------|-------|--------|----------|
| **Logistic Regression** | ~88% | ~0.88 | ⚡⚡⚡⚡⚡ | 10 MB | Real-time APIs |
| **Naive Bayes** | ~85% | ~0.85 | ⚡⚡⚡⚡⚡ | 5 MB | Resource-constrained |
| **SVM** | ~89% | ~0.89 | ⚡⚡⚡⚡ | 50 MB | **Production** |
| **Random Forest** | ~86% | ~0.86 | ⚡⚡⚡ | 100 MB | Feature importance |
| **LSTM** | ~87% | ~0.87 | ⚡⚡ | 500 MB | Learning sequences |
| **DistilBERT** | ~92% | ~0.92 | ⚡ | 1.5 GB | **Maximum accuracy** |

**Legend**: ⚡⚡⚡⚡⚡ = Ultra-fast (1000s/sec) ... ⚡ = Slow (~50/sec on GPU)

### Key Takeaways

1. **Winner by Accuracy**: DistilBERT at ~92%
   - But: Slow and resource-intensive
   - Use when: Accuracy is critical, speed isn't (batch processing)

2. **Best Balance**: SVM at ~89%
   - Fast enough for production, accurate enough to trust
   - Use when: Need real-world deployment

3. **Fastest**: Logistic Regression at ~88%
   - Only 3% less accurate than BERT, 100x faster
   - Use when: Sub-second response times required

4. **Most Interpretable**: Logistic Regression
   - Can see exactly which words influence decisions
   - Use when: Need to explain predictions to humans


## Project Structure

```
Sentiment-Analysis-MovieReviews/
│
├── README.md                   # (technical docs)
├── PROJECT_STORY.md            # Layman-friendly walkthrough
├── QUICK_START.md              # Getting started guide
├── requirements.txt            # All Python dependencies
├── .gitignore                  # Git ignore rules
│
├── notebooks/                  # The main deliverable - 4 complete notebooks
│   ├── 01_eda_preprocessing.ipynb          # EDA & data cleaning
│   ├── 02_classical_ml_models.ipynb        # 4 ML models (TF-IDF based)
│   ├── 03_deep_learning_models.ipynb       # LSTM & BERT implementations
│   └── 04_model_comparison_results.ipynb   # Final analysis & insights
│
├── data/                       # Dataset storage (auto-created)
│   ├── raw/                    # Original data from HuggingFace
│   └── processed/              # Cleaned CSV files
│
├── models/                     # Saved models (auto-created)
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   ├── random_forest.pkl
│   ├── lstm_sentiment.pth
│   └── distilbert_sentiment/
│
└── results/                    # Outputs (auto-created)
    ├── figures/                # Visualizations and charts
    │   ├── sentiment_distribution.png
    │   ├── word_clouds.png
    │   ├── model_comparison.png
    │   └── confusion_matrices.png
    └── metrics/                # Performance metrics CSV files
```

**Note**: Only `notebooks/`, `README.md`, and `requirements.txt` are included in the repo. The `data/`, `models/`, and `results/` folders are auto-created when you run the notebooks.

---
---

## Technical Details

### Technologies Used

**Core Stack**:
- Python 3.14 (or 3.9+)
- Jupyter Notebooks

**ML/DL Frameworks**:
- scikit-learn 1.4+ → Classical ML models
- PyTorch 2.0+ → Deep learning (LSTM)
- HuggingFace Transformers 4.40+ → BERT fine-tuning

**NLP Libraries**:
- NLTK 3.8+ → Tokenization, lemmatization, stopwords
- HuggingFace datasets → IMDb dataset loading

**Data & Visualization**:
- pandas, numpy → Data manipulation
- matplotlib, seaborn, plotly → Charts and plots
- wordcloud → Word cloud generation
- tqdm → Progress bars

### Model Architectures

**Classical ML Models** (with TF-IDF features):
```python
# TF-IDF Configuration
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),  # unigrams + bigrams
    min_df=5,
    max_df=0.8
)

# Models:
- LogisticRegression(C=1.0, max_iter=1000)
- MultinomialNB(alpha=0.1)
- LinearSVC(C=1.0, max_iter=1000)
- RandomForestClassifier(n_estimators=100)
```

**LSTM Architecture**:
```python
class LSTMSentimentClassifier(nn.Module):
    - Embedding: vocab_size × 128
    - Bidirectional LSTM: 256 hidden units, 2 layers
    - Dropout: 0.5
    - Linear: 256 → 1
    - Sigmoid activation
    
Training:
- Loss: BCELoss
- Optimizer: Adam (lr=0.001)
- Epochs: 5
- Batch size: 32
```

**DistilBERT Fine-tuning**:
```python
Model: DistilBertForSequenceClassification
- Base: distilbert-base-uncased (pre-trained)
- Classification head: 768 → 2 classes

Training:
- Optimizer: AdamW (lr=2e-5)
- Warmup steps: Linear schedule
- Epochs: 3
- Batch size: 16
- Max length: 512 tokens
```

### Evaluation Metrics

**Primary Metrics**:
- Accuracy: Correct predictions / Total predictions
- Precision: True Positives / (True Positives + False Positives)
- Recall: True Positives / (True Positives + False Negatives)
- F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
- ROC-AUC: Area under ROC curve

**F1-Score?**  
Balances precision and recall, important when cost of false positives ≈ cost of false negatives

### Preprocessing Pipeline

```python
1. Load raw text
2. Remove HTML tags (BeautifulSoup or regex)
3. Lowercase everything
4. Remove URLs, emails, mentions
5. Remove punctuation (except necessary ones)
6. Tokenize into words
7. Remove stopwords (optional)
8. Lemmatize (convert to base form)
9. Join back into clean text
```

---

## Key Learnings

### What Worked Well
 TF-IDF + SVM is surprisingly powerful (~89% accuracy)  
 BERT fine-tuning achieves state-of-the-art results  
 Proper preprocessing matters more than fancy models  
 BiLSTM captures context better than unidirectional  
 Lemmatization > Stemming for this task

### What Didn't Work
 Removing all stopwords hurt performance (context matters)  
 Character-level models were too slow without benefit  
 Simple bag-of-words couldn't beat TF-IDF  
 Overly aggressive text cleaning removed important signals

### Surprises
Logistic Regression only 4% behind BERT but 100x faster  
Random Forest underperformed vs simpler models  
BERT doesn't need much fine-tuning (3 epochs sufficient)  
Sarcasm breaks everything (future research area)


### Skills Demonstrated
- Natural Language Processing (NLP)
- Classical Machine Learning (scikit-learn)
- Deep Learning (PyTorch)
- Transfer Learning (BERT fine-tuning)
- Feature Engineering (TF-IDF)
- Model Evaluation & Selection
- Python Programming
- Data Visualization
- Technical Documentation


## References & Resources

### Dataset
- Maas, A. L., et al. (2011). "Learning Word Vectors for Sentiment Analysis." *ACL 2011*

### Key Papers
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." *NAACL*
- Hochreiter & Schmidhuber (1997). "Long Short-Term Memory." *Neural Computation*

### Libraries
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch](https://pytorch.org/docs/)
- [Scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)

---

## Probable FAQs

**Q: Can I use this code for my project?**  
A: Yes! It's open source. Just give credit and don't plagiarize for academic work.

**Q: Why not use GPT-4?**  
A: Overkill for binary classification + expensive. BERT is perfect for this task.

**Q: Which model should I actually use?**  
A: Depends on your use case:
- Real-time app → Logistic Regression or SVM
- Batch processing → BERT
- Resource-constrained → Naive Bayes
- Need explainability → Logistic Regression

**Q: How long did this take?**  
A: About 2 weeks (~40-50 hours total).

**Q: Will this work on tweets/product reviews?**  
A: Probably not without retraining. Different domains have different language patterns.
---

## 📄 License

MIT License - Feel free to use this for learning, projects, or portfolio work.

---

## Acknowledgments

- Stanford AI Lab for the amazing IMDb dataset
- HuggingFace for making transformers accessible
- PyTorch team for the deep learning framework
- The open-source community for all the amazing tools


**Project Timeline**: apx. 2 weeks
