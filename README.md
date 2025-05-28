# Fake-News Detection with Deep Learning  
Detecting misinformation in online news using **LSTM + GloVe** and **BERT**.

---

## Dataset

**ISOT Fake News Dataset**  
*23 481 real* & *23 481 fake* English news articles  
Columns: `title`, `text`, `subject`, `date` → merged & labelled (`label = 1` real, `0` fake).

---

## Pre-processing pipeline

1. Merge `True.csv` & `Fake.csv`, shuffle.
2. Text cleaning: lower-case, strip HTML/URLs, punctuation, stop-words.
3. [EDA] Word-clouds before and after cleaning.
4. Train–test split 70 / 30 (stratified).

---

## Models

| Model | Key layers / tricks | Best Val-Acc | Notes |
|-------|---------------------|--------------|-------|
| **Base LSTM** | GloVe-100d (frozen) → 2 × LSTM(64,32) → Dense | **100 %** | Surprising perfect fit; strong generalization |
| **Tuned LSTM** | 3 × LSTM(128,64,32) + more dropout | 90 % | Deeper ≠ better → recall on *fake* dropped |
| **Base BERT** | `bert-base-uncased`, full fine-tune, seq-len 512 | **99.94 %** | Only 8/13 470 errors |
| **Tuned BERT** | LR 3e-5, dropout 0.3, seq-len 256, early-stop | 99.78 % | 30/13 470 errors but faster training |

**Conclusion** : Pre-trained transformers (BERT) give robust, near-perfect results with moderate compute; over-tuning can trade a little accuracy for big speed gains. LSTM with good embeddings can still excel on balanced, monolingual data.

---

## Results snapshot

| Metric | Base LSTM | Tuned LSTM | Base BERT | Tuned BERT |
|--------|-----------|------------|-----------|------------|
| Accuracy | **1.000** | 0.902 | **0.999** | 0.998 |
| Precision (Fake) | 1.00 | **1.00** | 1.00 | 1.00 |
| Recall (Fake) | **1.00** | 0.82 | 1.00 | 0.98 |
| F1-score (macro) | **1.00** | 0.90 | **0.999** | 0.999 |

*(See notebooks for confusion matrices & training curves.)*

---

## Future work

1. Explore RoBERTa-base and DeBERTa-v3 fine-tuning.
2. Add social-context features (propagation graphs, user credibility) à la Faker.
3. Multilingual transfer: fine-tune bert-multilingual-cased on mixed-language fake-news sets.
4. Explainability: integrate SHAP / LIME for article-level rationale.

---

## Duration

- LSTM: 6 minutes (CPU with low RAM)
- BERT fine-tune: 30 minutes (A100 GPU with high RAM)

Note: All models were trained in Google Colab Pro environment. 
