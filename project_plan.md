# IPL 2026 Sentiment Analysis Project

## Phase 1: Project Definition & Setup ✓

### Project Scope
- **Topic**: Indian Premier League (IPL) 2026
- **Objective**: Analyze public sentiment about IPL 2026 across social media
- **Data Sources**: 
  - Primary: X (Twitter) using snscrape
  - Secondary: Reddit (if needed)
  - Backup: Kaggle IPL datasets
- **Sample Size Target**: 500-1000 posts/comments

### Key Research Questions
1. What is the overall public sentiment about IPL 2026?
2. What are the main positive aspects (favorite teams, players, moments)?
3. What are key issues/complaints (ticket prices, match scheduling, streaming)?
4. How can IPL improve fan experience?
5. What are current trends and discussions?

### Data Collection Strategy

#### Option 1: snscrape (Recommended for X - FREE)
```
Search terms: #IPL2026 #IndianPremierLeague2026 IPL OR "IPL 2026"
Date range: Last 1000-2000 posts (to get 500-1000 relevant samples)
Fields to collect: tweet text, timestamp, author, likes, retweets
```

#### Option 2: Reddit PRAW (FREE)
```
Subreddits: r/Cricket, r/IPL, r/IndianCricket
Search: "IPL 2026" posts
Sample size: 300-500 comments
```

#### Option 3: Kaggle Datasets
- Pre-existing IPL tweet datasets (manual collection possible)

### Tools & Libraries Needed
- **Data Collection**: snscrape, praw, tweepy
- **Data Processing**: pandas, numpy
- **Sentiment Analysis**: VADER, TextBlob, transformers
- **Visualization**: matplotlib, seaborn, plotly
- **Utilities**: requests, beautifulsoup4

### Project Directory Structure
```
d:\shraboni_nlp_project\
├── data/
│   ├── raw/
│   │   └── ipl_tweets.csv
│   └── processed/
│       └── ipl_tweets_cleaned.csv
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   └── 04_analysis_visualization.ipynb
├── scripts/
│   ├── collect_tweets.py
│   ├── preprocess.py
│   └── analyze.py
├── results/
│   ├── sentiment_distribution.png
│   ├── wordcloud.png
│   └── analysis_report.md
├── requirements.txt
└── README.md
```

### Next Steps
1. ✓ Define scope (IPL 2026, use snscrape)
2. → Set up Python environment & install libraries
3. → Create data collection script
4. → Execute collection (target 500-1000 samples)

---

**Status**: Ready to proceed to Phase 2
