# 📱 iPhone 16 — Public Sentiment & Market Insights

An NLP-based analysis of public discussions around the iPhone 16, designed to go beyond simple sentiment classification and understand **what users like, what frustrates them, and which product attributes shape the conversation**.

The project processes user comments and combines sentiment analysis, text classification, topic modelling and visualization to convert unstructured customer feedback into product-level insights.

---

## 📊 Key Findings

The final analysis contains **1,252 comments**.

| Sentiment | Share |
|-----------|------:|
| 🟦 Positive | 32.1% |
| ⬜ Neutral | 52.5% |
| 🟥 Negative | 15.4% |

Overall sentiment was **slightly positive**, but the most interesting finding was the large neutral segment.

This suggests that the conversation is not simply divided between satisfied and dissatisfied users. A substantial portion of users are discussing the product without expressing a strong positive or negative opinion.

### What users liked

**Camera & Photography** emerged as the strongest positive topic:

- 48.9% positive
- 47.8% neutral
- 3.3% negative

**Software & iOS** was another strong area:

- 42.2% positive
- 50.2% neutral
- 7.6% negative

### What users complained about

**Performance & Speed** had the highest negative sentiment:

- 25.9% negative
- 29.2% positive
- 44.8% neutral

Other areas with relatively high negative sentiment were:

- Display & Design — 18.5%
- Price & Value — 18.1%
- Battery & Charging — 11.0%

The analysis also surfaced recurring terms around:

`battery` · `screen` · `upgrade` · `price` · `camera`

---

# 🎯 Business Questions

The project was designed around four questions:

1. What is the overall public sentiment towards the iPhone 16?
2. Which product features drive positive and negative conversations?
3. What issues should product teams pay attention to?
4. How can unstructured customer feedback support product and marketing decisions?

---

# 🔄 NLP Pipeline

```text
YouTube Reviews
       │
       ▼
Comment Collection
       │
       ▼
Relevance Filtering
       │
       ▼
Text Preprocessing
       │
       ▼
Sentiment Analysis
 ┌─────┴──────┐
 ▼            ▼
VADER       RoBERTa
 └─────┬──────┘
       ▼
 Ensemble Sentiment
       │
       ▼
TF-IDF + ML Models
       │
       ▼
LDA Topic Modeling
       │
       ▼
Business Insights
       │
       ▼
Dashboard + HTML Report