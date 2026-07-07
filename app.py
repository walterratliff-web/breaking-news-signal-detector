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

# Suppress HuggingFace warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

# ─── Page Config ───────────────────────────────────────────────

st.set_page_config(
    page_title="Breaking News Signal Detector",
    page_icon="📡",
    layout="wide",
)

# ─── Sidebar Controls ─────────────────────────────────────────

st.sidebar.title("⚙️ Parameters")

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
    "**How it works:** Fetches live headlines from Middle East / geopolitics "
    "RSS feeds, encodes them with a sentence transformer, clusters similar "
    "headlines published within the time window, and scores each cluster "
    "on outlet breadth, velocity, and coherence."
)

# ─── RSS Feeds ─────────────────────────────────────────────────

RSS_FEEDS = {
    "BBC Middle East":      "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
    "Guardian Middle East":  "https://www.theguardian.com/world/middleeast/rss",
    "Al Jazeera":           "https://www.aljazeera.com/xml/rss/all.xml",
    "Al-Monitor":           "https://www.al-monitor.com/rss",
    "Middle East Eye":      "https://www.middleeasteye.net/rss",
    "NPR World":            "https://feeds.npr.org/1004/rss.xml",
    "Reuters World":        "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
}

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

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ─── Main App ──────────────────────────────────────────────────

st.title("📡 Breaking News Signal Detector")
st.markdown("*Middle East / Geopolitics — Live RSS Analysis*")

if st.button("🔍 Scan Now", type="primary", use_container_width=True):

    # ── Phase 1: Fetch feeds ──────────────────────────────────

    st.markdown("---")
    status = st.status("Scanning news feeds...", expanded=True)

    all_records = []
    for name, url in RSS_FEEDS.items():
        try:
            records = fetch_feed(name, url)
            status.write(f"✓ {name}: {len(records)} items")
            all_records.extend(records)
        except Exception as e:
            status.write(f"✗ {name}: {e}")

    headlines = pd.DataFrame(all_records)
    if not headlines.empty:
        headlines = headlines.sort_values("published", ascending=False).reset_index(drop=True)

    headlines["text"] = headlines["title"] + " " + headlines["summary"].fillna("")
    TOTAL_OUTLETS = headlines["outlet"].nunique()

    status.update(label=f"Fetched {len(headlines)} headlines from {TOTAL_OUTLETS} outlets", state="complete")

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

    clusters.sort(key=lambda c: -c["signal_score"])

    if not clusters:
        st.warning("No multi-outlet clusters found. Try adjusting parameters.")
        st.stop()

    # ── Phase 5: Display ──────────────────────────────────────

    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stories Detected", len(clusters))
    col2.metric("Headlines Scanned", len(headlines))
    col3.metric("Outlets Active", TOTAL_OUTLETS)
    breaking_count = sum(1 for c in clusters if c["signal_score"] >= 0.5)
    col4.metric("Developing+", breaking_count)

    # Signal cards
    st.markdown("---")
    st.subheader("Detected Signals")

    for i, cl in enumerate(clusters):
        emoji = signal_emoji(cl["signal_score"])
        label = signal_label(cl["signal_score"])
        color = signal_color(cl["signal_score"])

        with st.container():
            st.markdown(
                f"### {emoji} Signal {i+1}: {label} — {cl['signal_score']:.3f}"
            )
            st.markdown(f"**{cl['representative']}**")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Outlets", f"{cl['num_outlets']}/{TOTAL_OUTLETS}")
            m2.metric("Headlines", cl["num_headlines"])
            m3.metric("Time Span", f"{cl['time_span_hrs']}h")
            m4.metric("Coherence", f"{cl['avg_similarity']:.3f}")

            with st.expander("View all headlines"):
                for title in cl["titles"]:
                    st.markdown(f"- {title}")
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

        fig, ax = plt.subplots(figsize=(10, max(3, len(clusters) * 0.45)))
        cmap = plt.cm.colors.ListedColormap(["#F3F4F6", "#2563EB"])
        ax.imshow(heatmap_data, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(all_outlets)))
        ax.set_xticklabels(all_outlets, rotation=45, ha="right", fontsize=8)
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
    st.caption(
        f"Scan completed {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  "
        f"Model: all-MiniLM-L6-v2  |  "
        f"Threshold: {SIMILARITY_THRESHOLD}  |  "
        f"Window: {TIME_WINDOW_HRS}h  |  "
        f"Resolution: {RESOLUTION}"
    )

else:
    st.markdown(
        """
        <div style="text-align: center; padding: 60px 20px; color: #6B7280;">
            <p style="font-size: 48px; margin-bottom: 10px;">📡</p>
            <p style="font-size: 18px;">Click <strong>Scan Now</strong> to analyze live RSS feeds</p>
            <p style="font-size: 14px;">Adjust parameters in the sidebar before scanning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
