# BERT and Sentence-BERT for Natural Language Inference

A comprehensive implementation of BERT from scratch with Sentence-BERT extensions for Natural Language Inference (NLI) tasks, including a web application for real-time text similarity prediction.

## 🔍 Project Overview

This project implements a complete pipeline for natural language understanding:

1. **Custom BERT Training**: Train BERT from scratch using Masked Language Model (MLM) and Next Sentence Prediction (NSP) objectives on BookCorpus dataset
2. **Sentence-BERT Architecture**: Extend trained BERT with Siamese network structure for sentence embeddings
3. **NLI Classification**: Fine-tune on Stanford Natural Language Inference (SNLI) dataset for entailment, neutral, and contradiction prediction
4. **Web Application**: Deploy interactive web interface for real-time NLI predictions

![BERT Architecture](figures/BERT_embed.png)

## 🏗️ Architecture

### BERT Implementation
- **Layers**: 2 transformer encoder layers
- **Attention Heads**: 4 multi-head attention mechanisms
- **Hidden Dimensions**: 128 (d_model)
- **Feed Forward**: 512 dimensions
- **Maximum Sequence Length**: 64 tokens
- **Vocabulary**: Custom tokenizer trained on BookCorpus

### Sentence-BERT Extension
![Sentence-BERT Architecture](figures/sbert-architecture.png)

The Siamese network structure processes sentence pairs using:
- Mean pooling for sentence embeddings
- SoftmaxLoss objective: `softmax(W^T · (u, v, |u - v|))`
- Classification head for 3-class NLI prediction

## 📊 Dataset Information

### Training Data
- **Primary Corpus**: BookCorpus (5M samples subset)
  - *Source*: [Hugging Face BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus)
  - *Usage*: BERT pre-training with MLM + NSP objectives
- **NLI Dataset**: Stanford Natural Language Inference (SNLI)
  - *Source*: [Hugging Face SNLI](https://huggingface.co/datasets/snli)
  - *Training*: 50,000 premise-hypothesis pairs
  - *Validation*: 5,000 pairs
  - *Test*: 5,000 pairs

### Data Preprocessing
- Sentence segmentation using spaCy with parallel processing
- Text cleaning: lowercase, punctuation removal
- Custom tokenization with special tokens: `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`, `[UNK]`

## 🎯 Performance Results

### Classification Report (SNLI Test Set)

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| entailment  | 0.61      | 0.63   | 0.62     | 1716    |
| neutral     | 0.60      | 0.49   | 0.54     | 1622    |
| contradiction| 0.53     | 0.62   | 0.57     | 1662    |
| **Accuracy** |           |        | **0.58** | **5000** |
| **Macro avg**| **0.58**  | **0.58**| **0.58** | **5000** |
| **Weighted avg**| **0.58**| **0.58**| **0.58** | **5000** |

### Training Configuration

| Parameter | BERT Pre-training | Sentence-BERT Fine-tuning |
|-----------|------------------|---------------------------|
| Batch Size | 8 | 32 |
| Learning Rate | 1e-3 | 2e-5 |
| Epochs | 10 | 10 |
| Optimizer | Adam | Adam (separate for BERT & classifier) |
| Max Sequence Length | 64 | 64 |
| Masking Probability | 15% | - |

### Similarity Example
```
Premise: "Your contribution helped make it possible for us to provide our students with a quality education."
Hypothesis: "Your contributions were of no help with our students' education."
Cosine Similarity: 0.3526
```

## 🌐 Web Application

### [Live Demo](https://do-you-agree.redisland-4f8672a1.southeastasia.azurecontainerapps.io/)

### Features
- **Input Interface**: Two text boxes for premise and hypothesis
- **Real-time Prediction**: Instant NLI classification
- **Visual Feedback**: Color-coded results (green/yellow/red)
- **Model Integration**: Uses custom-trained Sentence-BERT

### Screenshots

#### Entailment Prediction
![Entailment Example](screenshots/Web%20UI%20Entailment.png)

#### Neutral Prediction
![Neutral Example](screenshots/Web%20UI%20Neutral.png)

#### Contradiction Prediction
![Contradiction Example](screenshots/Web%20UI%20Contradiction.png)

### Technical Stack
- **Backend**: Flask with PyTorch integration
- **Frontend**: HTML5, CSS3, JavaScript (async API calls)
- **Model Serving**: Custom BERT + Sentence-BERT pipeline
- **Deployment**: Azure Container Apps ready

## 🚀 Installation & Usage

### Prerequisites
```bash
# Install dependencies
pip install torch transformers datasets spacy scikit-learn flask matplotlib tqdm
python -m spacy download en_core_web_sm
```

### Running the Notebook
```bash
jupyter notebook A4.ipynb
```

### Web Application
```bash
cd app
python app.py
# Navigate to http://localhost:5000
```

### Model Files
- `models/bert_state_dict.pt`: Trained BERT weights
- `models/sbert_state_dict.pt`: Fine-tuned Sentence-BERT + classifier
- `models/tokenizer_metadata.json`: Vocabulary and hyperparameters

## 📈 Analysis & Limitations

### Achievements
✅ **Complete BERT Implementation**: Full transformer architecture from scratch  
✅ **Effective Training Pipeline**: MLM + NSP pre-training followed by NLI fine-tuning  
✅ **Functional Web Application**: Production-ready interface with real-time inference  
✅ **Proper Evaluation**: Comprehensive metrics and confusion matrix analysis  

### Current Limitations
- **Computational Constraints**: Reduced model size (2 layers vs. 12 in BERT-base)
- **Limited Dataset**: Subset training due to local compute limitations
- **Simplified Tokenization**: Basic word-level tokenizer vs. WordPiece/BPE
- **Performance Gap**: Lower accuracy compared to pre-trained models

### Proposed Improvements
1. **Larger Architecture**: Scale to BERT-base dimensions (12 layers, 768 hidden units)
2. **Advanced Tokenization**: Implement WordPiece or Byte-Pair Encoding
3. **Extended Training**: Increase corpus size and training duration
4. **Hard Negative Mining**: Improve contrastive learning for sentence embeddings
5. **Ensemble Methods**: Combine multiple model predictions

## 📚 References

- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)
- Reimers, N., & Gurevych, I. (2019). [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/)
- Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). [A large annotated corpus for learning natural language inference](https://nlp.stanford.edu/pubs/snli_paper.pdf)

## 🤝 Acknowledgments

**Course**: AT82.05 Artificial Intelligence: Natural Language Understanding (NLU)  
**Instructors**: Chaklam Silpasuwanchai, Todsavad Tangtortan  
**Dataset Sources**: Hugging Face (BookCorpus, SNLI)
