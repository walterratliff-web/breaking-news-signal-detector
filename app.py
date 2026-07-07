import streamlit as st
import feedparser
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime, timezone
from dateutil import parser as dateparser
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import warnings
import re

# Suppress HuggingFace warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

# ─── Page Config ───────────────────────────────────────────────

st.set_page_config(
    page_title="Breaking News Signal Detector",
    page_icon="📡",
    layout="wide",
)

# ─── Topic Feed Definitions ───────────────────────────────────

TOPIC_FEEDS = {
    "Middle East / Geopolitics": {
        "BBC Middle East":      "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
        "Guardian Middle East":  "https://www.theguardian.com/world/middleeast/rss",
        "Al Jazeera":           "https://www.aljazeera.com/xml/rss/all.xml",
        "Al-Monitor":           "https://www.al-monitor.com/rss",
        "Middle East Eye":      "https://www.middleeasteye.net/rss",
        "NPR World":            "https://feeds.npr.org/1004/rss.xml",
        "Reuters World":        "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
    },
    "Business": {
        "CNBC":                 "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "BBC Business":         "https://feeds.bbci.co.uk/news/business/rss.xml",
        "Reuters Business":     "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
        "MarketWatch":          "https://feeds.marketwatch.com/marketwatch/topstories/",
        "NPR Business":         "https://feeds.npr.org/1006/rss.xml",
        "Financial Times":      "https://www.ft.com/rss/home",
        "Guardian Business":    "https://www.theguardian.com/business/rss",
    },
    "US Politics": {
        "Politico":             "https://rss.politico.com/politics-news.xml",
        "The Hill":             "https://thehill.com/feed/",
        "NPR Politics":         "https://feeds.npr.org/1014/rss.xml",
        "BBC North America":    "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
        "Guardian US":          "https://www.theguardian.com/us-news/rss",
        "AP Top News":          "https://rsshub.app/apnews/topics/apf-topnews",
        "Reuters Politics":     "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
    },
    "Religion": {
        "Religion News Service": "https://religionnews.com/feed/",
        "Christianity Today":    "https://www.christianitytoday.com/feed/",
        "National Catholic Reporter": "https://www.ncronline.org/rss.xml",
        "Crux":                  "https://cruxnow.com/feed/",
        "Church Times":          "https://www.churchtimes.co.uk/rss",
        "Al Jazeera":            "https://www.aljazeera.com/xml/rss/all.xml",
        "Guardian Religion":     "https://www.theguardian.com/world/religion/rss",
    },
    "General International": {
        "BBC World":            "https://feeds.bbci.co.uk/news/world/rss.xml",
        "Guardian World":       "https://www.theguardian.com/world/rss",
        "Al Jazeera":           "https://www.aljazeera.com/xml/rss/all.xml",
        "NPR World":            "https://feeds.npr.org/1004/rss.xml",
        "Reuters World":        "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
        "France 24":            "https://www.france24.com/en/rss",
        "DW News":              "https://rss.dw.com/rdf/rss-en-all",
    },
    "Sports": {
        "BBC Sport":            "https://feeds.bbci.co.uk/sport/rss.xml",
        "ESPN":                 "https://www.espn.com/espn/rss/news",
        "CBS Sports":           "https://www.cbssports.com/rss/headlines/",
        "Guardian Sport":       "https://www.theguardian.com/sport/rss",
        "Sky Sports":           "https://www.skysports.com/rss/12040",
        "Yahoo Sports":         "https://sports.yahoo.com/rss/",
        "NPR":                  "https://feeds.npr.org/1001/rss.xml",
    },
}

# Build reverse mapping: outlet name -> set of topic areas
OUTLET_TOPICS = {}
for topic, feeds in TOPIC_FEEDS.items():
    for outlet_name in feeds:
        if outlet_name not in OUTLET_TOPICS:
            OUTLET_TOPICS[outlet_name] = set()
        OUTLET_TOPICS[outlet_name].add(topic)

# Topic emoji mapping
TOPIC_EMOJI = {
    "Middle East / Geopolitics": "🌍",
    "Business": "💼",
    "US Politics": "🇺🇸",
    "Religion": "⛪",
    "General International": "🌐",
    "Sports": "⚽",
}

# ─── Sidebar Controls ─────────────────────────────────────────

st.sidebar.title("⚙️ Parameters")

topic_options = ["📡 All Beats"] + list(TOPIC_FEEDS.keys())
selected_option = st.sidebar.selectbox(
    "Topic",
    options=topic_options,
    index=0,
    help="Scan all beats at once, or drill into a specific topic."
)

is_all_beats = selected_option == "📡 All Beats"
selected_topic = None if is_all_beats else selected_option

SIMILARITY_THRESHOLD = st.sidebar.slider(
    "Similarity threshold",
    min_value=0.20, max_value=0.80, value=0.45, step=0.05,
    help="Minimum cosine similarity to link two headlines. Higher = stricter matches."
)

TIME_WINDOW_HRS = st.sidebar.slider(
    "Time window (hours)",
    min_value=1, max_value=24, value=6, step=1,
    help="Maximum hours between two headlines for them to be linked."
)

RESOLUTION = st.sidebar.slider(
    "Cluster resolution",
    min_value=0.5, max_value=3.0, value=1.5, step=0.25,
    help="Louvain granularity. Higher = more, smaller clusters."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works:** Fetches live headlines from RSS feeds, encodes them "
    "with a sentence transformer, clusters similar headlines published within "
    "the time window, and scores each cluster on outlet breadth, velocity, "
    "and coherence."
)

# ─── Helper Functions ──────────────────────────────────────────

def signal_label(score):
    if score >= 0.7: return "BREAKING"
    if score >= 0.5: return "DEVELOPING"
    if score >= 0.3: return "EMERGING"
    return "LOW"

def signal_color(score):
    if score >= 0.7: return "#DC2626"
    if score >= 0.5: return "#EA580C"
    if score >= 0.3: return "#CA8A04"
    return "#9CA3AF"

def signal_emoji(score):
    if score >= 0.7: return "🔴"
    if score >= 0.5: return "🟠"
    if score >= 0.3: return "🟡"
    return "⚪"

def fetch_feed(outlet_name, feed_url):
    feed = feedparser.parse(feed_url)
    records = []
    for entry in feed.entries:
        pub_date = None
        for date_field in ("published", "updated"):
            raw = getattr(entry, date_field, None)
            if raw:
                try:
                    pub_date = dateparser.parse(raw)
                    break
                except (ValueError, TypeError):
                    continue
        if pub_date is None:
            pub_date = datetime.now(timezone.utc)
        if pub_date.tzinfo is None:
            pub_date = pub_date.replace(tzinfo=timezone.utc)
        records.append({
            "outlet":    outlet_name,
            "title":     entry.get("title", "").strip(),
            "summary":   entry.get("summary", "").strip(),
            "link":      entry.get("link", ""),
            "published": pub_date,
        })
    return records

def get_cluster_topics(cl_outlets):
    """Determine which topic beats a cluster spans based on its outlets."""
    topics = set()
    for outlet in cl_outlets:
        if outlet in OUTLET_TOPICS:
            topics.update(OUTLET_TOPICS[outlet])
    return sorted(topics)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Common words to skip when extracting proper nouns from headlines
STOP_PROPER = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "are", "was", "were", "has", "have", "had", "be", "been", "being",
    "with", "from", "by", "as", "its", "it", "he", "she", "his", "her",
    "they", "their", "this", "that", "who", "what", "how", "why", "when",
    "where", "not", "no", "but", "if", "up", "out", "over", "after",
    "before", "new", "says", "said", "say", "could", "would", "will",
    "may", "can", "more", "most", "also", "about", "into", "than",
    "just", "now", "first", "last", "back", "all", "get", "do", "does",
    "did", "being", "between", "under", "through", "during", "here",
    "there", "then", "so", "very", "still", "amid", "among", "across",
    "against", "while", "off", "down", "set", "top", "big", "old",
    "high", "low", "long", "much", "many", "well", "even", "take",
    "make", "come", "go", "see", "look", "find", "give", "tell",
    "call", "try", "ask", "need", "want", "use", "should", "must",
    "live", "latest", "breaking", "update", "report", "reports",
    "news", "watch", "video", "exclusive", "opinion", "analysis",
}

def extract_entities(titles):
    """Extract likely proper nouns from headlines using capitalization patterns."""
    # Find capitalized multi-word phrases and single capitalized words
    entity_counts = {}

    for title in titles:
        # Find sequences of capitalized words (2+ words = likely proper noun phrase)
        phrases = re.findall(r"(?:[A-Z][a-zA-Z''-]+(?:\s+[A-Z][a-zA-Z''-]+)+)", title)
        for phrase in phrases:
            phrase = phrase.strip()
            if len(phrase) > 2:
                entity_counts[phrase] = entity_counts.get(phrase, 0) + 2  # Weight phrases higher

        # Find individual capitalized words (skip first word of title)
        words = title.split()
        for word in words[1:]:
            # Must start with uppercase, be 2+ chars, not all caps (skip "US", "UK" handled below)
            clean = re.sub(r"[^a-zA-Z''-]", "", word)
            if not clean or len(clean) < 2:
                continue
            if clean[0].isupper() and clean.lower() not in STOP_PROPER:
                entity_counts[clean] = entity_counts.get(clean, 0) + 1

        # Also capture common abbreviations/acronyms (2-5 uppercase letters)
        acronyms = re.findall(r"\b[A-Z]{2,5}\b", title)
        for acr in acronyms:
            if acr.lower() not in STOP_PROPER and acr not in ("CNN", "BBC", "NPR"):
                entity_counts[acr] = entity_counts.get(acr, 0) + 1

    # Sort by frequency and return top entries
    sorted_entities = sorted(entity_counts.items(), key=lambda x: -x[1])

    # Deduplicate: if "Ali Khamenei" and "Khamenei" both appear, keep the longer one
    final = []
    seen_lower = set()
    for name, count in sorted_entities:
        name_lower = name.lower()
        # Skip if this is a substring of something we already kept
        is_subset = any(name_lower in s for s in seen_lower)
        if not is_subset:
            # Remove any shorter versions we already added
            final = [(n, c) for n, c in final if n.lower() not in name_lower]
            seen_lower = {n.lower() for n, _ in final}
            final.append((name, count))
            seen_lower.add(name_lower)

    return [name for name, _ in final[:8]]

# ─── Build Feed List ───────────────────────────────────────────

def get_feeds_to_scan():
    """Return deduplicated feeds. For All Beats, merge across topics by URL."""
    if not is_all_beats:
        return TOPIC_FEEDS[selected_topic]

    # Deduplicate by URL — same URL under different outlet names
    # gets fetched once under a combined name
    seen_urls = {}
    merged = {}
    for topic, feeds in TOPIC_FEEDS.items():
        for name, url in feeds.items():
            if url not in seen_urls:
                seen_urls[url] = name
                merged[name] = url
            # If same outlet name exists with same URL, skip
    return merged

# ─── Main App ──────────────────────────────────────────────────

st.title("📡 Breaking News Signal Detector")
if is_all_beats:
    st.markdown("*All Beats — Cross-Topic Live RSS Analysis*")
else:
    st.markdown(f"*{selected_topic} — Live RSS Analysis*")

if st.button("🔍 Scan Now", type="primary", use_container_width=True):

    RSS_FEEDS = get_feeds_to_scan()

    # ── Phase 1: Fetch feeds ──────────────────────────────────

    st.markdown("---")
    scan_label = "all beats" if is_all_beats else selected_topic
    status = st.status(f"Scanning {scan_label}...", expanded=True)

    all_records = []
    for name, url in RSS_FEEDS.items():
        try:
            records = fetch_feed(name, url)
            status.write(f"✓ {name}: {len(records)} items")
            all_records.extend(records)
        except Exception as e:
            status.write(f"✗ {name}: {e}")

    headlines = pd.DataFrame(all_records)
    if headlines.empty:
        st.error("No headlines retrieved. Check your internet connection.")
        st.stop()

    # Deduplicate headlines with identical titles from shared feeds
    headlines = headlines.drop_duplicates(subset=["title", "link"]).reset_index(drop=True)
    headlines = headlines.sort_values("published", ascending=False).reset_index(drop=True)
    headlines["text"] = headlines["title"] + " " + headlines["summary"].fillna("")
    TOTAL_OUTLETS = headlines["outlet"].nunique()

    status.update(
        label=f"Fetched {len(headlines)} unique headlines from {TOTAL_OUTLETS} outlets",
        state="complete"
    )

    # ── Phase 2: Encode & compute similarity ──────────────────

    status2 = st.status("Encoding headlines with sentence transformer...", expanded=False)

    model = load_model()
    embeddings = model.encode(
        headlines["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)

    pairs = []
    n = len(headlines)
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i, j]
            if score < SIMILARITY_THRESHOLD:
                continue
            time_gap_hrs = abs(
                (headlines.iloc[i]["published"] - headlines.iloc[j]["published"])
                .total_seconds() / 3600
            )
            if time_gap_hrs > TIME_WINDOW_HRS:
                continue
            outlet_i = headlines.iloc[i]["outlet"]
            outlet_j = headlines.iloc[j]["outlet"]
            pairs.append({
                "idx_a": i, "idx_b": j,
                "similarity": round(score, 3),
                "cross_outlet": outlet_i != outlet_j,
            })

    pairs_df = pd.DataFrame(pairs)
    cross_count = len(pairs_df[pairs_df["cross_outlet"]]) if not pairs_df.empty else 0
    status2.update(label=f"Found {cross_count} cross-outlet pairs", state="complete")

    # ── Phase 3: Cluster detection ────────────────────────────

    if pairs_df.empty:
        st.warning("No similar headline pairs found. Try lowering the similarity threshold.")
        st.stop()

    G = nx.Graph()
    involved_indices = set(pairs_df["idx_a"]).union(set(pairs_df["idx_b"]))
    for idx in involved_indices:
        row = headlines.iloc[int(idx)]
        G.add_node(int(idx), outlet=row["outlet"], title=row["title"],
                   published=row["published"])
    for _, row in pairs_df.iterrows():
        G.add_edge(int(row["idx_a"]), int(row["idx_b"]), weight=row["similarity"])

    communities = nx.community.louvain_communities(
        G, weight="weight", resolution=RESOLUTION, seed=42
    )

    clusters = []
    for comm in communities:
        if len(comm) < 2:
            continue
        members = headlines.loc[list(comm)].copy()
        outlets = members["outlet"].unique()
        if len(outlets) < 2:
            continue
        earliest = members["published"].min()
        latest = members["published"].max()
        time_span_hrs = (latest - earliest).total_seconds() / 3600
        subgraph = G.subgraph(comm)
        edge_weights = [d["weight"] for _, _, d in subgraph.edges(data=True)]
        avg_sim = np.mean(edge_weights) if edge_weights else 0
        clusters.append({
            "headline_indices": list(comm),
            "num_headlines": len(comm),
            "num_outlets": len(outlets),
            "outlets": sorted(outlets),
            "avg_similarity": round(avg_sim, 3),
            "time_span_hrs": round(time_span_hrs, 1),
            "earliest": earliest,
            "latest": latest,
            "titles": members["title"].tolist(),
        })

    # ── Phase 4: Signal scoring ───────────────────────────────

    W_OUTLETS   = 0.45
    W_VELOCITY  = 0.30
    W_COHERENCE = 0.25

    for cl in clusters:
        outlet_score = cl["num_outlets"] / TOTAL_OUTLETS
        if cl["time_span_hrs"] <= 0:
            velocity_score = 1.0
        else:
            velocity_score = max(0, 1 - (cl["time_span_hrs"] / TIME_WINDOW_HRS))
        coherence_score = min(cl["avg_similarity"], 1.0)
        signal = (
            W_OUTLETS * outlet_score +
            W_VELOCITY * velocity_score +
            W_COHERENCE * coherence_score
        )
        cl["outlet_score"]    = round(outlet_score, 3)
        cl["velocity_score"]  = round(velocity_score, 3)
        cl["coherence_score"] = round(coherence_score, 3)
        cl["signal_score"]    = round(signal, 3)

        member_embeddings = embeddings[cl["headline_indices"]]
        centroid = member_embeddings.mean(axis=0)
        sims_to_centroid = cosine_similarity(
            member_embeddings, centroid.reshape(1, -1)
        ).flatten()
        best_idx = cl["headline_indices"][np.argmax(sims_to_centroid)]
        cl["representative"] = headlines.iloc[best_idx]["title"]

        # Determine topic beats this cluster spans
        cl["topics"] = get_cluster_topics(cl["outlets"])

    clusters.sort(key=lambda c: -c["signal_score"])

    if not clusters:
        st.warning("No multi-outlet clusters found. Try adjusting parameters.")
        st.stop()

    # ── Entity extraction ─────────────────────────────────────

    for cl in clusters:
        members = headlines.loc[cl["headline_indices"]]
        cl["entities"] = extract_entities(members["title"].tolist())

    # ── Phase 5: Display ──────────────────────────────────────

    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stories Detected", len(clusters))
    col2.metric("Headlines Scanned", len(headlines))
    col3.metric("Outlets Active", TOTAL_OUTLETS)
    breaking_count = sum(1 for c in clusters if c["signal_score"] >= 0.5)
    col4.metric("Developing+", breaking_count)

    # ── Top 5 banner (All Beats mode) ─────────────────────────

    if is_all_beats and len(clusters) > 0:
        st.markdown("---")
        st.subheader("🔥 Top Signals Across All Beats")

        for i, cl in enumerate(clusters[:5]):
            emoji = signal_emoji(cl["signal_score"])
            label = signal_label(cl["signal_score"])
            topic_tags = " ".join(
                f"{TOPIC_EMOJI.get(t, '📰')} {t}" for t in cl["topics"]
            )

            col_left, col_right = st.columns([4, 1])
            with col_left:
                st.markdown(
                    f"**{emoji} {cl['representative']}**"
                )
                st.caption(
                    f"{label} ({cl['signal_score']:.3f})  ·  "
                    f"{cl['num_outlets']} outlets  ·  "
                    f"{cl['time_span_hrs']}h span  ·  "
                    f"{topic_tags}"
                )
            with col_right:
                st.markdown(
                    f"<div style='text-align:right; font-size:24px; "
                    f"font-weight:bold; color:{signal_color(cl['signal_score'])}'>"
                    f"{cl['signal_score']:.2f}</div>",
                    unsafe_allow_html=True,
                )

            if i < 4:
                st.divider()

    # Signal cards
    st.markdown("---")
    st.subheader("All Detected Signals")

    for i, cl in enumerate(clusters):
        emoji = signal_emoji(cl["signal_score"])
        label = signal_label(cl["signal_score"])

        with st.container():
            # Topic tags for All Beats mode
            if is_all_beats:
                topic_tags = " · ".join(
                    f"{TOPIC_EMOJI.get(t, '📰')} {t}" for t in cl["topics"]
                )
                st.markdown(
                    f"### {emoji} Signal {i+1}: {label} — {cl['signal_score']:.3f}"
                )
                st.caption(topic_tags)
            else:
                st.markdown(
                    f"### {emoji} Signal {i+1}: {label} — {cl['signal_score']:.3f}"
                )

            st.markdown(f"**{cl['representative']}**")

            # Entity tags
            if cl.get("entities"):
                tags = " · ".join(f"`{e}`" for e in cl["entities"])
                st.markdown(f"🏷️ {tags}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Outlets", f"{cl['num_outlets']}/{TOTAL_OUTLETS}")
            m2.metric("Headlines", cl["num_headlines"])
            m3.metric("Time Span", f"{cl['time_span_hrs']}h")
            m4.metric("Coherence", f"{cl['avg_similarity']:.3f}")

            with st.expander("View all headlines"):
                members = headlines.loc[cl["headline_indices"]]
                for _, h in members.iterrows():
                    if h["link"]:
                        st.markdown(f"- [{h['title']}]({h['link']})  \n  *{h['outlet']}*")
                    else:
                        st.markdown(f"- {h['title']}  \n  *{h['outlet']}*")
                st.caption(f"Sources: {', '.join(cl['outlets'])}")

            st.markdown("---")

    # Charts
    st.subheader("Visualizations")
    tab1, tab2, tab3 = st.tabs([
        "📊 Signal Strength", "🗺️ Timeline", "📋 Outlet Coverage"
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, max(3, len(clusters) * 0.5)))
        sorted_cl = list(reversed(clusters))
        labels = [c["representative"][:45] + ("..." if len(c["representative"]) > 45 else "")
                  for c in sorted_cl]
        scores = [c["signal_score"] for c in sorted_cl]
        colors = [signal_color(s) for s in scores]

        bars = ax.barh(range(len(labels)), scores, color=colors,
                       edgecolor="white", height=0.7)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{score:.3f}  {signal_label(score)}",
                    va="center", fontsize=8, fontweight="bold",
                    color=signal_color(score))

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Signal Score")
        ax.set_title("Signal Strength", fontweight="bold")
        ax.axvline(x=0.7, color="#DC2626", linestyle="--", alpha=0.3)
        ax.axvline(x=0.5, color="#EA580C", linestyle="--", alpha=0.3)
        ax.axvline(x=0.3, color="#CA8A04", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(12, max(3, len(clusters) * 0.6)))
        for i, cl in enumerate(clusters):
            y = len(clusters) - 1 - i
            color = signal_color(cl["signal_score"])
            ax.plot([cl["earliest"], cl["latest"]], [y, y],
                    color=color, linewidth=3, solid_capstyle="round", alpha=0.7)
            member_times = headlines.loc[cl["headline_indices"]]["published"]
            ax.scatter(member_times, [y] * len(member_times),
                       s=40, color=color, edgecolors="white", linewidth=0.5,
                       zorder=5, alpha=0.9)
            short = cl["representative"][:40] + ("..." if len(cl["representative"]) > 40 else "")
            ax.text(cl["earliest"], y + 0.3,
                    f"{signal_label(cl['signal_score'])}  {short}",
                    fontsize=7, fontweight="bold", color=color, va="bottom")

        ax.set_yticks([])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        fig.autofmt_xdate()
        ax.set_title("Story Detection Timeline", fontweight="bold")
        ax.set_xlabel("Publication Time (UTC)")
        legend_patches = [
            mpatches.Patch(color="#DC2626", label="Breaking"),
            mpatches.Patch(color="#EA580C", label="Developing"),
            mpatches.Patch(color="#CA8A04", label="Emerging"),
            mpatches.Patch(color="#9CA3AF", label="Low"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        all_outlets = sorted(headlines["outlet"].unique())
        heatmap_data = []
        for cl in clusters:
            member_outlets = set(headlines.loc[cl["headline_indices"]]["outlet"])
            heatmap_data.append([1 if o in member_outlets else 0 for o in all_outlets])

        fig, ax = plt.subplots(figsize=(max(10, len(all_outlets) * 0.8),
                                        max(3, len(clusters) * 0.45)))
        cmap = plt.cm.colors.ListedColormap(["#F3F4F6", "#2563EB"])
        ax.imshow(heatmap_data, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(all_outlets)))
        ax.set_xticklabels(all_outlets, rotation=45, ha="right", fontsize=7)
        ylabels = [c["representative"][:35] + ("..." if len(c["representative"]) > 35 else "")
                   for c in clusters]
        ax.set_yticks(range(len(clusters)))
        ax.set_yticklabels(ylabels, fontsize=7)
        for i in range(len(all_outlets) + 1):
            ax.axvline(i - 0.5, color="white", linewidth=2)
        for i in range(len(clusters) + 1):
            ax.axhline(i - 0.5, color="white", linewidth=2)
        ax.set_title("Outlet Coverage Matrix", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Footer
    st.markdown("---")
    topic_label = "All Beats" if is_all_beats else selected_topic
    st.caption(
        f"Scan completed {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  "
        f"Topic: {topic_label}  |  "
        f"Model: all-MiniLM-L6-v2  |  "
        f"Threshold: {SIMILARITY_THRESHOLD}  |  "
        f"Window: {TIME_WINDOW_HRS}h  |  "
        f"Resolution: {RESOLUTION}"
    )

else:
    if is_all_beats:
        st.markdown(
            """
            <div style="text-align: center; padding: 60px 20px; color: #6B7280;">
                <p style="font-size: 48px; margin-bottom: 10px;">📡</p>
                <p style="font-size: 18px;">Click <strong>Scan Now</strong> to detect breaking stories across all beats</p>
                <p style="font-size: 14px;">Middle East · Business · US Politics · Religion · International · Sports</p>
                <p style="font-size: 13px; margin-top: 20px;">Or select a specific beat from the sidebar to drill down</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 60px 20px; color: #6B7280;">
                <p style="font-size: 48px; margin-bottom: 10px;">{TOPIC_EMOJI.get(selected_topic, '📡')}</p>
                <p style="font-size: 18px;">Click <strong>Scan Now</strong> to scan <strong>{selected_topic}</strong> feeds</p>
                <p style="font-size: 14px;">Adjust parameters in the sidebar before scanning</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
