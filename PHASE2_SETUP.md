# Phase 2 Setup Guide - Python Environment

## Step 1: Create Virtual Environment

### Option A: Using venv (Built-in)
```powershell
# Navigate to project directory
cd d:\shraboni_nlp_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Option B: Using conda
```powershell
conda create -n ipl_sentiment python=3.10
conda activate ipl_sentiment
```

## Step 2: Install Dependencies

```powershell
# Make sure your virtual environment is activated, then:
pip install -r requirements.txt

# If you get errors with torch, install CPU version:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Step 3: Download NLTK Data

Run this in Python:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
```

Or in PowerShell:
```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('stopwords')"
```

## Step 4: (Optional) Set Up Reddit API Credentials

If you want to collect Reddit data:

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create app" → select "script"
3. Fill in details (e.g., app name: "IPL_Sentiment_Analyzer")
4. Get your `client_id` and `client_secret`
5. Create `.env` file in project root:

```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=IPL_Sentiment_Analyzer/1.0
```

## Step 5: Create Required Directories

```powershell
# In PowerShell
mkdir data\raw
mkdir data\processed
mkdir results
mkdir notebooks
```

---

## Data Collection Script Ready! ✓

### File: `scripts\collect_tweets.py`

**Features:**
- ✓ Collects tweets about IPL 2026 using snscrape (FREE, no API key needed)
- ✓ Optional Reddit data collection
- ✓ Saves to CSV format
- ✓ Progress indicators
- ✓ Error handling

### Usage:

```powershell
# Activate virtual environment first
.\venv\Scripts\Activate.ps1

# Run the script
python scripts\collect_tweets.py
```

### What It Does:
1. Searches multiple hashtags and keywords
2. Collects up to 800 tweets
3. Saves to `data/raw/ipl_tweets.csv`
4. Optionally collects Reddit posts
5. Combines all data in `data/raw/ipl_combined_raw.csv`

### Expected Output:
```
============================================================
🚀 IPL 2026 Sentiment Analysis - Data Collection
============================================================
Start time: 2026-04-18 10:30:45

🐦 Collecting tweets about IPL 2026...

📍 Searching: '#IPL2026'
   ✓ Collected 100 tweets...
   ✓ Collected 200 tweets...
   ...

✅ Total tweets collected: 750
💾 Tweets saved to data/raw/ipl_tweets.csv
...
```

---

## Next Steps

After installation and data collection:
1. Run `collect_tweets.py` to gather ~500-1000 samples
2. Proceed to Phase 3: Data Preprocessing
3. Perform sentiment analysis
4. Generate insights and visualizations

---

**Status**: Phase 2 Ready to Execute!
