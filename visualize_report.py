"""
Final Visualization & Business Report — iPhone 16 Sentiment Analysis
=====================================================================
Stage 5 of 5 in the pipeline.

Input  : iphone16_topics.csv       (from topic_modeling.py)
Outputs: dashboard.png             — full multi-panel chart
         executive_summary.html    — self-contained HTML report

Answers all four business questions:
  Q1 — What is the overall public sentiment?
  Q2 — What are the key issues or positive drivers?
  Q3 — How can businesses improve products/services?
  Q4 — What strategic decisions can be made?

Install:
    pip install pandas matplotlib seaborn wordcloud plotly jinja2 tqdm
"""

import warnings
warnings.filterwarnings("ignore")

import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime
from wordcloud import WordCloud

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV        = "iphone16_topics.csv"
DASHBOARD_PNG    = "dashboard.png"
REPORT_HTML      = "executive_summary.html"

LABEL_COL        = "final_label"
TEXT_COL         = "comment_text"
CLEAN_COL        = "clean_text"
TOKENS_COL       = "tokens"
TOPIC_COL        = "dominant_topic"
TOPIC_LABEL_COL  = "topic_label"
TOPIC_CONF_COL   = "topic_confidence"
DATE_COL         = "published_at"           # set to None if no date column

LABEL_ORDER      = ["positive", "neutral", "negative"]
COLORS = {
    "positive": "#3B8BD4",
    "neutral":  "#888780",
    "negative": "#E24B4A",
}
TOPIC_PALETTE = [
    "#185FA5", "#1D9E75", "#BA7517",
    "#A32D2D", "#534AB7", "#993556",
]

# Human-readable topic labels — must match what topic_modeling.py produced.
# Update if you changed NUM_TOPICS or the order in that script.
TOPIC_LABELS = {
    0: "Camera & Photography",
    1: "Battery & Charging",
    2: "Performance & Speed",
    3: "Price & Value",
    4: "Display & Design",
    5: "Software & iOS",
}


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading {path}...")
    df = pd.read_csv(path)

    required = [LABEL_COL, CLEAN_COL]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Run all prior pipeline stages first.\n"
                f"Found columns: {list(df.columns)}"
            )

    df[LABEL_COL] = df[LABEL_COL].str.lower().str.strip()
    df = df[df[LABEL_COL].isin(LABEL_ORDER)]
    df.reset_index(drop=True, inplace=True)

    # Parse dates if available
    global DATE_COL
    if DATE_COL and DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        if df[DATE_COL].isna().all():
            print("  WARNING: date column could not be parsed — time trend disabled.")
            DATE_COL = None
    else:
        DATE_COL = None

    # Resolve topic labels from topic_modeling if not already present
    if TOPIC_LABEL_COL not in df.columns and TOPIC_COL in df.columns:
        df[TOPIC_LABEL_COL] = df[TOPIC_COL].map(TOPIC_LABELS).fillna("Unknown")

    print(f"  {len(df):,} comments loaded.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: COMPUTE CORE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Central place that derives every number needed for charts and the HTML report.
    Returns a dict of metrics so nothing is computed twice.
    """
    total = len(df)
    dist  = df[LABEL_COL].value_counts()
    pct   = (dist / total * 100).round(1)

    # ── Overall sentiment ────────────────────────────────────────────────────
    pos_pct = pct.get("positive", 0.0)
    neu_pct = pct.get("neutral",  0.0)
    neg_pct = pct.get("negative", 0.0)

    if pos_pct >= 50:
        overall_mood = "Mostly Positive"
        mood_color   = COLORS["positive"]
    elif neg_pct >= 50:
        overall_mood = "Mostly Negative"
        mood_color   = COLORS["negative"]
    elif pos_pct > neg_pct:
        overall_mood = "Slightly Positive"
        mood_color   = "#2E7CC4"
    else:
        overall_mood = "Mixed / Polarised"
        mood_color   = "#888780"

    # ── Topic × sentiment cross-tab ──────────────────────────────────────────
    if TOPIC_LABEL_COL in df.columns:
        cross_raw = pd.crosstab(df[TOPIC_LABEL_COL], df[LABEL_COL])
        cross_pct = cross_raw.div(cross_raw.sum(axis=1), axis=0).fillna(0)
        for col in LABEL_ORDER:
            if col not in cross_pct.columns:
                cross_pct[col] = 0.0
        cross_pct = cross_pct[LABEL_ORDER]
        cross_raw = cross_raw.reindex(columns=LABEL_ORDER, fill_value=0)
    else:
        cross_pct = pd.DataFrame()
        cross_raw = pd.DataFrame()

    # ── Top complaint topics ─────────────────────────────────────────────────
    if not cross_pct.empty:
        neg_rank = cross_pct["negative"].sort_values(ascending=False)
        pos_rank = cross_pct["positive"].sort_values(ascending=False)
        top_complaints  = neg_rank.head(3)
        top_drivers     = pos_rank.head(3)
    else:
        top_complaints = pd.Series(dtype=float)
        top_drivers    = pd.Series(dtype=float)

    # ── Top 20 tokens overall, per sentiment ────────────────────────────────
    def top_tokens(subset_df: pd.DataFrame, n: int = 20) -> list[tuple]:
        if TOKENS_COL in subset_df.columns:
            all_tokens = [
                t for row in subset_df[TOKENS_COL].dropna()
                for t in str(row).split("|") if len(t) > 2
            ]
        else:
            all_tokens = [
                t for row in subset_df[CLEAN_COL].dropna()
                for t in str(row).split() if len(t) > 2
            ]
        return Counter(all_tokens).most_common(n)

    token_map = {
        "all":      top_tokens(df),
        "positive": top_tokens(df[df[LABEL_COL] == "positive"]),
        "negative": top_tokens(df[df[LABEL_COL] == "negative"]),
        "neutral":  top_tokens(df[df[LABEL_COL] == "neutral"]),
    }

    # ── Most helpful negative comments (longest, highest topic confidence) ──
    neg_df = df[df[LABEL_COL] == "negative"].copy()
    neg_df["char_len"] = neg_df[TEXT_COL].fillna("").str.len()
    worst_comments = (
        neg_df.nlargest(6, "char_len")[[TEXT_COL, TOPIC_LABEL_COL]]
        if TOPIC_LABEL_COL in neg_df.columns
        else neg_df.nlargest(6, "char_len")[[TEXT_COL]]
    )

    pos_df = df[df[LABEL_COL] == "positive"].copy()
    pos_df["char_len"] = pos_df[TEXT_COL].fillna("").str.len()
    best_comments = (
        pos_df.nlargest(6, "char_len")[[TEXT_COL, TOPIC_LABEL_COL]]
        if TOPIC_LABEL_COL in pos_df.columns
        else pos_df.nlargest(6, "char_len")[[TEXT_COL]]
    )

    return dict(
        total=total,
        dist=dist,
        pct=pct,
        pos_pct=pos_pct,
        neu_pct=neu_pct,
        neg_pct=neg_pct,
        overall_mood=overall_mood,
        mood_color=mood_color,
        cross_raw=cross_raw,
        cross_pct=cross_pct,
        top_complaints=top_complaints,
        top_drivers=top_drivers,
        token_map=token_map,
        worst_comments=worst_comments,
        best_comments=best_comments,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 1 — Overall Sentiment Donut
# ─────────────────────────────────────────────────────────────────────────────

def plot_sentiment_donut(ax: plt.Axes, m: dict) -> None:
    counts = [m["dist"].get(l, 0) for l in LABEL_ORDER]
    colors = [COLORS[l] for l in LABEL_ORDER]
    wedge_props = {"width": 0.52, "edgecolor": "white", "linewidth": 2}

    wedges, _, autotexts = ax.pie(
        counts, labels=LABEL_ORDER,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops=wedge_props,
        pctdistance=0.78,
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")

    # Centre text
    ax.text(0, 0.07, m["overall_mood"], ha="center", va="center",
            fontsize=11, fontweight="bold", color=m["mood_color"])
    ax.text(0, -0.15, f"n = {m['total']:,}", ha="center", va="center",
            fontsize=9, color="gray")
    ax.set_title("Q1 — Overall Public Sentiment", fontweight="bold", pad=12)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 2 — Sentiment Trend Over Time
# ─────────────────────────────────────────────────────────────────────────────

def plot_time_trend(ax: plt.Axes, df: pd.DataFrame) -> None:
    if DATE_COL is None or DATE_COL not in df.columns:
        ax.text(0.5, 0.5,
                "No date column found.\nAdd a 'published_at' column\nto enable trend analysis.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
        ax.set_title("Sentiment Trend Over Time", fontweight="bold")
        return

    df = df.copy()
    df["week"] = df[DATE_COL].dt.to_period("W").dt.start_time
    weekly = (
        df.groupby(["week", LABEL_COL])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABEL_ORDER, fill_value=0)
    )
    weekly_pct = weekly.div(weekly.sum(axis=1), axis=0) * 100

    for label in LABEL_ORDER:
        if label in weekly_pct.columns:
            ax.plot(weekly_pct.index, weekly_pct[label],
                    color=COLORS[label], linewidth=1.8,
                    label=label.capitalize(), marker="o", markersize=3)
            ax.fill_between(weekly_pct.index, weekly_pct[label],
                            alpha=0.12, color=COLORS[label])

    ax.set_xlabel("Week")
    ax.set_ylabel("% of comments")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Sentiment Trend Over Time", fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=8)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 3 — Topic × Sentiment Stacked Horizontal Bar  (Q2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_topic_sentiment(ax: plt.Axes, m: dict) -> None:
    if m["cross_pct"].empty:
        ax.text(0.5, 0.5, "No topic data available.\nRun topic_modeling.py first.",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("Q2 — Sentiment Breakdown per Topic", fontweight="bold")
        return

    cp = m["cross_pct"].copy()
    left = np.zeros(len(cp))
    labels = cp.index.tolist()

    for label in LABEL_ORDER:
        vals = cp[label].values
        bars = ax.barh(labels, vals, left=left,
                       color=COLORS[label], edgecolor="white", linewidth=0.4,
                       label=label.capitalize())
        for bar, val, l in zip(bars, vals, left):
            if val > 0.06:
                ax.text(l + val / 2, bar.get_y() + bar.get_height() / 2,
                        f"{val:.0%}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        left += vals

    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Proportion of comments")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Q2 — Sentiment Breakdown per Topic", fontweight="bold")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(9)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 4 — Top Complaint vs Praise Topics (Q2 / Q3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_issue_drivers(ax: plt.Axes, m: dict) -> None:
    if m["top_complaints"].empty:
        ax.text(0.5, 0.5, "No topic data.", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("Top Issues & Positive Drivers", fontweight="bold")
        return

    topics   = list(m["top_complaints"].index) + list(m["top_drivers"].index)
    neg_vals = [m["top_complaints"][t] * 100 if t in m["top_complaints"] else 0
                for t in topics]
    pos_vals = [m["top_drivers"][t]    * 100 if t in m["top_drivers"]    else 0
                for t in topics]

    x = np.arange(len(topics))
    w = 0.35
    ax.bar(x - w/2, neg_vals, w, color=COLORS["negative"],
           edgecolor="white", label="% Negative")
    ax.bar(x + w/2, pos_vals, w, color=COLORS["positive"],
           edgecolor="white", label="% Positive")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [textwrap.fill(t, 14) for t in topics],
        fontsize=8
    )
    ax.set_ylabel("% of topic comments")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.set_title("Q2 / Q3 — Top Issues & Positive Drivers", fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 5 — Negative Word Cloud  (Q3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_wordcloud(ax: plt.Axes, token_list: list, title: str,
                   colormap: str = "Reds") -> None:
    freq = dict(token_list)
    if not freq:
        ax.text(0.5, 0.5, "No tokens.", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(title, fontweight="bold")
        return

    wc = WordCloud(
        width=500, height=280,
        background_color="white",
        colormap=colormap,
        max_words=60,
        prefer_horizontal=0.9,
        min_font_size=8,
    ).generate_from_frequencies(freq)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 6 — Feature Priority Matrix  (Q4)
#  Plots each topic as a bubble: x = positive %, y = negative %, size = volume
# ─────────────────────────────────────────────────────────────────────────────

def plot_priority_matrix(ax: plt.Axes, df: pd.DataFrame, m: dict) -> None:
    if m["cross_pct"].empty:
        ax.text(0.5, 0.5, "No topic data.", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("Q4 — Strategic Priority Matrix", fontweight="bold")
        return

    topic_counts = df[TOPIC_LABEL_COL].value_counts() if TOPIC_LABEL_COL in df.columns \
                   else pd.Series(dtype=int)

    for i, topic in enumerate(m["cross_pct"].index):
        x   = m["cross_pct"].loc[topic, "positive"] * 100
        y   = m["cross_pct"].loc[topic, "negative"] * 100
        vol = topic_counts.get(topic, 50)
        ax.scatter(x, y, s=vol / max(topic_counts.max(), 1) * 1500 + 80,
                   color=TOPIC_PALETTE[i % len(TOPIC_PALETTE)],
                   alpha=0.75, edgecolors="white", linewidth=1.2, zorder=3)
        ax.annotate(
            textwrap.fill(topic, 14),
            (x, y), textcoords="offset points", xytext=(6, 6),
            fontsize=7.5, ha="left",
        )

    # Quadrant shading
    mid_x, mid_y = 50, 50
    ax.axvline(mid_x, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axhline(mid_y, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.fill_between([0, mid_x],  mid_y, 100, alpha=0.05, color=COLORS["negative"])
    ax.fill_between([mid_x, 100], 0,  mid_y, alpha=0.05, color=COLORS["positive"])

    ax.set_xlabel("% Positive comments", fontsize=9)
    ax.set_ylabel("% Negative comments", fontsize=9)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.text(5,  95, "Pain points\n(fix urgently)",  fontsize=7.5, color=COLORS["negative"],  va="top")
    ax.text(65, 5,  "Strengths\n(market loudly)",   fontsize=7.5, color=COLORS["positive"],  va="bottom")
    ax.text(5,  5,  "Neutral / low-signal",          fontsize=7.5, color="gray",              va="bottom")
    ax.text(65, 95, "Polarising\n(needs attention)", fontsize=7.5, color="#BA7517",            va="top")
    ax.set_title("Q4 — Strategic Priority Matrix", fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 7 — Volume bar chart by topic
# ─────────────────────────────────────────────────────────────────────────────

def plot_topic_volume(ax: plt.Axes, df: pd.DataFrame) -> None:
    if TOPIC_LABEL_COL not in df.columns:
        ax.text(0.5, 0.5, "No topic data.", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("Discussion Volume by Topic", fontweight="bold")
        return

    counts = df[TOPIC_LABEL_COL].value_counts()
    colors = [TOPIC_PALETTE[i % len(TOPIC_PALETTE)] for i in range(len(counts))]
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
    ax.set_xlabel("Number of comments")
    ax.set_title("Discussion Volume by Topic", fontweight="bold")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(9)


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(df: pd.DataFrame, m: dict) -> None:
    print("\nBuilding dashboard...")
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor("#F8F9FA")

    fig.suptitle(
        "iPhone 16 — Public Sentiment Analysis Dashboard",
        fontsize=17, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.52, wspace=0.35,
        top=0.96, bottom=0.04, left=0.07, right=0.97
    )

    # Row 0 ─────────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    plot_sentiment_donut(ax0, m)

    ax1 = fig.add_subplot(gs[0, 1:3])
    plot_time_trend(ax1, df)

    # Row 1 ─────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0:2])
    plot_topic_sentiment(ax2, m)

    ax3 = fig.add_subplot(gs[1, 2])
    plot_topic_volume(ax3, df)

    # Row 2 ─────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0:2])
    plot_issue_drivers(ax4, m)

    ax5 = fig.add_subplot(gs[2, 2])
    plot_priority_matrix(ax5, df, m)

    # Row 3 ─────────────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 0])
    plot_wordcloud(ax6, m["token_map"]["positive"],
                   "Top Positive Words", colormap="Blues")

    ax7 = fig.add_subplot(gs[3, 1])
    plot_wordcloud(ax7, m["token_map"]["negative"],
                   "Top Negative Words", colormap="Reds")

    ax8 = fig.add_subplot(gs[3, 2])
    plot_wordcloud(ax8, m["token_map"]["all"],
                   "All Topics — Top Words", colormap="Purples")

    plt.savefig(DASHBOARD_PNG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Dashboard saved → {DASHBOARD_PNG}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  CONSOLE INSIGHT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_insights(df: pd.DataFrame, m: dict) -> None:
    sep = "=" * 64

    print(f"\n{sep}")
    print("  BUSINESS INSIGHT SUMMARY — iPhone 16 Sentiment")
    print(sep)

    # Q1
    print(f"""
  Q1 ─ OVERALL PUBLIC SENTIMENT
  ─────────────────────────────
  Verdict  : {m['overall_mood']}
  Positive : {m['pos_pct']:.1f}%   Neutral : {m['neu_pct']:.1f}%   Negative : {m['neg_pct']:.1f}%
  Total comments analysed : {m['total']:,}
""")

    # Q2
    print("  Q2 ─ KEY ISSUES & POSITIVE DRIVERS")
    print("  ─────────────────────────────────────────────────────────")
    if not m["top_complaints"].empty:
        print("  Top complaint topics (highest % negative):")
        for topic, val in m["top_complaints"].items():
            bar = "█" * int(val * 20)
            print(f"    {topic:<28}  {val*100:4.1f}% neg  {bar}")
        print()
        print("  Top praise topics (highest % positive):")
        for topic, val in m["top_drivers"].items():
            bar = "█" * int(val * 20)
            print(f"    {topic:<28}  {val*100:4.1f}% pos  {bar}")
    else:
        print("  (No topic data — run topic_modeling.py first)")

    # Q3
    print(f"""
  Q3 ─ PRODUCT IMPROVEMENT OPPORTUNITIES
  ────────────────────────────────────────────────────────
  1. Prioritise fixing the top-negative topics above — they
     represent the loudest user pain points right now.
  2. Read the verbatim complaints in 'worst_comments' in the
     HTML report for specific actionable language.
  3. Topics with high negative % AND high comment volume
     (large bubble in Priority Matrix) need the most urgent
     engineering attention.
  4. Neutral topics are under-tapped — a small UX improvement
     there can shift them into positive territory fast.
""")

    # Q4
    print("  Q4 ─ STRATEGIC DECISIONS")
    print("  ────────────────────────────────────────────────────────")
    if not m["top_drivers"].empty:
        best = m["top_drivers"].index[0]
        worst = m["top_complaints"].index[0] if not m["top_complaints"].empty else "N/A"
        print(f"""
  Marketing   → Lead with '{best}' — it's your highest-
                positive topic. Use in ads and social content.

  R&D / Eng   → Address '{worst}' first. It has the
                highest negative rate and is hurting brand NPS.

  Support     → Identify repeat complaint patterns in the top-
                negative topics and build targeted FAQ / guides.

  Roadmap     → Use the Priority Matrix (bottom-right quadrant =
                strengths) to decide which features to expand
                and which to deprioritise or fix.

  Monitoring  → Re-run this pipeline weekly/monthly and track
                sentiment trend. A rising negative % on any
                topic is an early warning signal.
""")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
#  HTML EXECUTIVE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _comment_rows(comments_df: pd.DataFrame, label: str) -> str:
    rows = []
    color = {"positive": "#3B8BD4", "negative": "#E24B4A", "neutral": "#888780"}.get(label, "#333")
    topic_col = TOPIC_LABEL_COL if TOPIC_LABEL_COL in comments_df.columns else None
    for _, row in comments_df.head(6).iterrows():
        text  = str(row[TEXT_COL])[:280] + ("…" if len(str(row[TEXT_COL])) > 280 else "")
        topic = str(row[topic_col]) if topic_col and topic_col in row.index else ""
        topic_tag = f'<span class="tag">{topic}</span>' if topic else ""
        rows.append(f"""
            <tr>
              <td>{topic_tag}</td>
              <td style="color:{color}">{"▲" if label=="positive" else "▼" if label=="negative" else "–"}</td>
              <td>{text}</td>
            </tr>""")
    return "".join(rows)


def _topic_bars(cross_pct: pd.DataFrame) -> str:
    if cross_pct.empty:
        return "<p>No topic data.</p>"
    rows = []
    for topic in cross_pct.index:
        pos = cross_pct.loc[topic, "positive"] * 100
        neu = cross_pct.loc[topic, "neutral"]  * 100
        neg = cross_pct.loc[topic, "negative"] * 100
        rows.append(f"""
        <div class="topic-row">
          <div class="topic-name">{topic}</div>
          <div class="bar-wrap">
            <div class="bar-pos" style="width:{pos:.1f}%" title="Positive {pos:.1f}%"></div>
            <div class="bar-neu" style="width:{neu:.1f}%" title="Neutral {neu:.1f}%"></div>
            <div class="bar-neg" style="width:{neg:.1f}%" title="Negative {neg:.1f}%"></div>
          </div>
          <div class="bar-label">{pos:.0f}% pos / {neg:.0f}% neg</div>
        </div>""")
    return "".join(rows)


def build_html_report(df: pd.DataFrame, m: dict) -> None:
    print("Building HTML report...")

    now = datetime.now().strftime("%d %b %Y, %H:%M")

    # Top tokens as comma-separated strings
    neg_words = ", ".join(w for w, _ in m["token_map"]["negative"][:15])
    pos_words = ", ".join(w for w, _ in m["token_map"]["positive"][:15])

    complaint_list = "".join(
        f"<li><strong>{t}</strong> — {v*100:.1f}% negative</li>"
        for t, v in m["top_complaints"].items()
    ) if not m["top_complaints"].empty else "<li>No topic data.</li>"

    driver_list = "".join(
        f"<li><strong>{t}</strong> — {v*100:.1f}% positive</li>"
        for t, v in m["top_drivers"].items()
    ) if not m["top_drivers"].empty else "<li>No topic data.</li>"

    neg_rows = _comment_rows(m["worst_comments"], "negative")
    pos_rows = _comment_rows(m["best_comments"],  "positive")
    topic_bars_html = _topic_bars(m["cross_pct"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>iPhone 16 — Sentiment Analysis Report</title>
<style>
  :root {{
    --pos: #3B8BD4; --neu: #888780; --neg: #E24B4A;
    --bg: #F8F9FA; --card: #FFFFFF; --border: #E0E4EA;
    --text: #1A1A2E; --muted: #6B7280;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: var(--bg);
          color: var(--text); padding: 32px 24px; font-size: 14px; }}
  h1   {{ font-size: 1.9em; font-weight: 700; margin-bottom: 4px; }}
  h2   {{ font-size: 1.25em; font-weight: 600; margin-bottom: 14px;
          border-left: 4px solid var(--pos); padding-left: 10px; }}
  h3   {{ font-size: 1.05em; font-weight: 600; margin-bottom: 8px; color: var(--muted); }}
  p    {{ line-height: 1.6; color: var(--muted); margin-bottom: 8px; }}
  .meta {{ color: var(--muted); font-size: 0.85em; margin-bottom: 32px; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
           margin-bottom: 28px; }}
  .card {{ background: var(--card); border: 1px solid var(--border);
           border-radius: 10px; padding: 20px; }}
  .kpi  {{ text-align: center; }}
  .kpi .val {{ font-size: 2.6em; font-weight: 700; display: block; }}
  .kpi .lbl {{ font-size: 0.82em; color: var(--muted); }}
  .pos {{ color: var(--pos); }} .neg {{ color: var(--neg); }} .neu {{ color: var(--neu); }}
  .section {{ margin-bottom: 36px; }}
  .card.wide {{ grid-column: 1 / -1; }}

  /* Stacked topic bars */
  .topic-row  {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
  .topic-name {{ width: 200px; font-size: 0.85em; flex-shrink: 0; }}
  .bar-wrap   {{ flex: 1; display: flex; height: 18px; border-radius: 4px;
                 overflow: hidden; background: #eee; }}
  .bar-pos    {{ background: var(--pos); }}
  .bar-neu    {{ background: var(--neu); }}
  .bar-neg    {{ background: var(--neg); }}
  .bar-label  {{ width: 130px; font-size: 0.78em; color: var(--muted);
                 text-align: right; flex-shrink: 0; }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; font-size: 0.83em; }}
  th    {{ background: #EEF2F8; text-align: left; padding: 8px 10px;
           font-weight: 600; color: var(--muted); }}
  td    {{ padding: 8px 10px; border-bottom: 1px solid var(--border);
           vertical-align: top; line-height: 1.4; }}
  .tag  {{ background: #EEF2F8; border-radius: 4px; padding: 2px 7px;
           font-size: 0.78em; white-space: nowrap; }}

  /* Q-blocks */
  .q-block {{ background: var(--card); border: 1px solid var(--border);
              border-radius: 10px; padding: 22px; margin-bottom: 20px; }}
  .q-block h2 {{ margin-bottom: 10px; }}
  ul {{ margin-left: 18px; }} li {{ margin-bottom: 6px; line-height: 1.5; }}

  /* Keyword chips */
  .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
  .chip  {{ padding: 4px 12px; border-radius: 20px; font-size: 0.80em;
            background: #EEF2F8; color: var(--text); }}
  .chip.neg {{ background: #FDE8E8; color: var(--neg); }}
  .chip.pos {{ background: #E5F1FB; color: var(--pos); }}

  footer {{ text-align: center; color: var(--muted); font-size: 0.78em;
            margin-top: 40px; border-top: 1px solid var(--border); padding-top: 16px; }}
</style>
</head>
<body>

<h1>📱 iPhone 16 — Public Sentiment Analysis</h1>
<p class="meta">Generated : {now} &nbsp;|&nbsp; Pipeline : preprocess → sentiment → model → topics → report &nbsp;|&nbsp; n = {m['total']:,} comments</p>

<!-- ── KPI Row ── -->
<div class="grid">
  <div class="card kpi">
    <span class="val pos">{m['pos_pct']:.1f}%</span>
    <span class="lbl">Positive</span>
  </div>
  <div class="card kpi">
    <span class="val neu">{m['neu_pct']:.1f}%</span>
    <span class="lbl">Neutral</span>
  </div>
  <div class="card kpi">
    <span class="val neg">{m['neg_pct']:.1f}%</span>
    <span class="lbl">Negative</span>
  </div>
</div>

<!-- ── Dashboard image ── -->
<div class="section">
  <div class="card">
    <h2>Full Analysis Dashboard</h2>
    <p>See <strong>{DASHBOARD_PNG}</strong> for the complete chart panel.</p>
  </div>
</div>

<!-- ── Q1 ── -->
<div class="q-block section">
  <h2>Q1 — What is the overall public sentiment?</h2>
  <p>
    Public sentiment for the iPhone 16 is <strong style="color:{m['mood_color']}">{m['overall_mood']}</strong>.
    Out of {m['total']:,} comments analysed,
    <strong class="pos">{m['pos_pct']:.1f}%</strong> are positive,
    <strong class="neu">{m['neu_pct']:.1f}%</strong> neutral, and
    <strong class="neg">{m['neg_pct']:.1f}%</strong> negative.
    {"The majority of users are happy with the device, though a meaningful minority raise concerns."
     if m['pos_pct'] > m['neg_pct'] else
     "Negative comments outnumber positive ones — this is a signal to investigate root causes urgently."}
  </p>
</div>

<!-- ── Q2 ── -->
<div class="q-block section">
  <h2>Q2 — What are the key issues or positive drivers?</h2>

  <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:16px;">
    <div>
      <h3>🔴 Top Complaint Topics</h3>
      <ul>{complaint_list}</ul>
    </div>
    <div>
      <h3>🔵 Top Positive Drivers</h3>
      <ul>{driver_list}</ul>
    </div>
  </div>

  <h3 style="margin-bottom:12px;">Sentiment Breakdown by Topic</h3>
  <div style="padding:4px 0 0 4px;">
    <div style="display:flex; gap:16px; font-size:0.8em; margin-bottom:8px; color:var(--muted);">
      <span>■ <span class="pos">Positive</span></span>
      <span>■ <span class="neu">Neutral</span></span>
      <span>■ <span class="neg">Negative</span></span>
    </div>
    {topic_bars_html}
  </div>

  <h3 style="margin:14px 0 8px;">Top negative keywords</h3>
  <div class="chips">{" ".join(f'<span class="chip neg">{w}</span>' for w in neg_words.split(", "))}</div>

  <h3 style="margin:14px 0 8px;">Top positive keywords</h3>
  <div class="chips">{" ".join(f'<span class="chip pos">{w}</span>' for w in pos_words.split(", "))}</div>
</div>

<!-- ── Q3 ── -->
<div class="q-block section">
  <h2>Q3 — How can businesses improve products / services?</h2>
  <ul>
    <li>Address the highest-negative topics first — these represent the loudest, most pressing user pain points.</li>
    <li>Use verbatim complaints below to understand <em>exactly</em> what language users use when they're unhappy — these map directly to product team action items.</li>
    <li>Topics that are both <strong>high-volume</strong> and <strong>high-negative</strong> (bottom-left of the Priority Matrix) require immediate engineering attention.</li>
    <li>Neutral topics are an opportunity: even a small UX improvement can tip them into positive territory and boost overall NPS.</li>
    <li>Compare negative keyword frequency month-over-month; a declining trend validates that product fixes are working.</li>
  </ul>

  <h3 style="margin:16px 0 8px;">Representative negative comments</h3>
  <table>
    <thead><tr><th>Topic</th><th></th><th>Comment</th></tr></thead>
    <tbody>{neg_rows}</tbody>
  </table>
</div>

<!-- ── Q4 ── -->
<div class="q-block section">
  <h2>Q4 — What strategic decisions can be made?</h2>

  <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
    <div>
      <h3>📣 Marketing</h3>
      <p>Lead with your top positive topics in campaigns and social content — those are your proof points users already believe in.</p>
    </div>
    <div>
      <h3>🔧 R&amp;D / Engineering</h3>
      <p>Prioritise the top complaint topics in the next sprint. Focus on both defect resolution and communication (release notes, support articles).</p>
    </div>
    <div>
      <h3>🎯 Product Roadmap</h3>
      <p>Features with <strong>high positive rate</strong> → invest to expand. Features with <strong>high negative rate</strong> → fix or deprioritise. Use the Priority Matrix quadrants to guide the roadmap quarterly.</p>
    </div>
    <div>
      <h3>📊 Ongoing Monitoring</h3>
      <p>Re-run this pipeline weekly. A rising negative % in any topic is an early warning signal before it becomes a PR issue. Track Cohen's Kappa between VADER and RoBERTa over time as a data quality health check.</p>
    </div>
  </div>

  <h3 style="margin:16px 0 8px;">Representative positive comments (marketing source material)</h3>
  <table>
    <thead><tr><th>Topic</th><th></th><th>Comment</th></tr></thead>
    <tbody>{pos_rows}</tbody>
  </table>
</div>

<footer>
  iPhone 16 Sentiment Analysis Pipeline &nbsp;|&nbsp;
  preprocess_comments.py → sentiment_labeling.py → model_training.py → topic_modeling.py → visualize_report.py
  &nbsp;|&nbsp; Generated {now}
</footer>
</body>
</html>"""

    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML report saved → {REPORT_HTML}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df = load_data(INPUT_CSV)
    m  = compute_metrics(df)

    # ── Console summary ──────────────────────────────────────────────────────
    print_insights(df, m)

    # ── Dashboard PNG ────────────────────────────────────────────────────────
    build_dashboard(df, m)

    # ── HTML executive report ────────────────────────────────────────────────
    build_html_report(df, m)

    print(f"\n✓ Done!")
    print(f"  {DASHBOARD_PNG}    — share with tech leads / data science team")
    print(f"  {REPORT_HTML}  — share with product managers, marketing, C-suite")


if __name__ == "__main__":
    main()