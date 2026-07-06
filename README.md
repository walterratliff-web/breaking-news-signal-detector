# Breaking News Signal Detector

A Python prototype that monitors RSS feeds from major news outlets and detects when multiple sources converge on the same emerging story within a short time window — the signature pattern of breaking news.

Built as a single Google Colab notebook. No API keys, no external services, no infrastructure required.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)

---

## What It Does

The detector ingests live headlines from seven news outlets covering the Middle East and geopolitics, measures textual similarity across all headline pairs, groups them into story clusters using community detection, and scores each cluster on a composite signal metric. The output is a ranked list of detected stories, from BREAKING to LOW, with four visualizations.

On a typical run it identifies 8–15 distinct story clusters from roughly 130 headlines, surfacing stories like a Syrian intelligence verdict (3 outlets, 1.4-hour convergence) or Iran's supreme leader funeral coverage (5 outlets, 9.5 hours) — and correctly ranking the faster-converging story higher.

---

## The Pipeline

### Phase 1: RSS Ingestion
Pulls live headlines and summaries from seven feeds — BBC Middle East, The Guardian Middle East, Al Jazeera, Al-Monitor, Middle East Eye, NPR World, and Reuters World. Each headline is normalized into a structured record with outlet name, title, summary, link, and publication timestamp.

### Phase 2: TF-IDF Similarity
Combines each headline's title and summary into a single text field, then converts them into numerical vectors using TF-IDF (Term Frequency–Inverse Document Frequency). Common words are filtered out so distinctive terms — place names, leader names, event-specific language — drive the similarity scores. Pairwise cosine similarity is computed across all headlines, but only pairs that are both textually similar (≥ 0.25) and published within a 6-hour window are retained.

### Phase 3: Cluster Detection
Headlines become nodes in a graph, with edges connecting similar-and-temporally-close pairs. Louvain community detection identifies densely connected subgroups — stories covered by multiple outlets — while avoiding the "mega-cluster" problem that simpler methods (connected components) produce when loosely related stories chain together transitively. Only clusters spanning two or more outlets are kept.

### Phase 4: Signal Scoring
Each cluster receives a composite score (0–1) from three weighted factors:
- **Outlet breadth (45%):** What fraction of monitored sources picked up the story.
- **Velocity (30%):** How quickly coverage converged. A 1-hour span scores higher than a 5-hour span.
- **Coherence (25%):** Average cosine similarity across the cluster's headlines.

The highest-scoring cluster's "representative headline" is selected as the one closest to the cluster centroid — the headline most typical of the group.

Scores map to urgency tiers:
| Score | Tier |
|-------|------|
| 0.70+ | 🔴 BREAKING |
| 0.50+ | 🟠 DEVELOPING |
| 0.30+ | 🟡 EMERGING |
| < 0.30 | ⚪ LOW |

### Phase 5: Visualization
Four matplotlib charts provide a presentation-ready view:
- **Signal Score Bar Chart** — ranked horizontal bars, color-coded by urgency tier.
- **Story Timeline** — plots each cluster along a time axis with individual headline dots.
- **Outlet Coverage Heatmap** — grid showing which outlets covered which stories.
- **Score Breakdown** — stacked bars decomposing each signal into its three components.

---

## Prior Work

This prototype draws on established approaches in news event detection. The EU's [Europe Media Monitor](https://emm.newsbrief.eu/overview.html) (EMM) has operated since 2002, clustering news every ten minutes using a four-hour window and selecting the cluster medoid as the representative headline — the same technique used here. EMM classifies rapidly rising clusters as breaking news based on article count and source diversity. Google News employs a similar architecture at scale, using publication velocity and cross-source confirmation as ranking signals.

In the research literature, TF-IDF with pairwise cosine similarity followed by unsupervised clustering is a standard approach to news story chain detection, generally preferred over supervised methods because new topics appear continuously. More recent work has explored LLM-enhanced clustering on the [GDELT](https://www.gdeltproject.org/) event database for improved event summarization and labeling.

This project implements the same core methodology — TF-IDF vectorization, cosine similarity with a time window, community detection, and centroid-based headline selection — as a lightweight, single-notebook prototype focused on a specific topic beat (Middle East/geopolitics), requiring no enterprise infrastructure or API keys.

---

## Limitations

- **Topic drift within clusters.** Louvain occasionally pulls in tangentially related headlines that share regional vocabulary but cover different events. Sentence transformer embeddings would improve semantic precision over TF-IDF.
- **RSS feed reliability.** Not all outlets maintain stable feeds. Some return zero results intermittently, and timestamp formats vary. The pipeline handles this gracefully, but coverage gaps are possible.
- **Static snapshot.** This prototype captures a single point-in-time scan. A production system would run continuously, tracking how clusters grow and merge — the evolution of a signal is often as informative as its initial detection.
- **Threshold sensitivity.** The similarity threshold (0.25), time window (6 hours), and Louvain resolution (1.5) were tuned empirically for this feed set. Different topic domains or outlet mixes would require recalibration.

---

## Quick Start

1. Open the notebook in Google Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/breaking-news-signal-detector/blob/main/breaking_news_detector.ipynb)

2. Run all cells: **Runtime → Run all**

3. Results appear in approximately 30 seconds. No API keys or configuration needed.

---

## Technical Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.10 |
| Text vectorization | scikit-learn (TfidfVectorizer) |
| Similarity | scikit-learn (cosine_similarity) |
| Graph clustering | NetworkX (Louvain community detection) |
| RSS parsing | feedparser |
| Visualization | matplotlib |
| Data handling | pandas, numpy |
| Environment | Google Colab |

---

## Configuration

Three parameters in the notebook control the detector's behavior:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `SIMILARITY_THRESHOLD` | 0.25 | Minimum cosine similarity to link two headlines. Raise for stricter matches. |
| `TIME_WINDOW_HRS` | 6 | Maximum hours between two headlines for them to be linked. Lower for faster-breaking stories. |
| `RESOLUTION` | 1.5 | Louvain granularity. Higher values produce more, smaller clusters. |

---

## License

MIT

---

## Author

Walter Ratliff — https://www.linkedin.com/in/walterratliff/
