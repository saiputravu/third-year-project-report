import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
from adjustText import adjust_text
from sentence_transformers import SentenceTransformer

# Define the sentences.
sentences = [
    "clustering: I hate dogs.",
    "clustering: i hate dogs",
    "clustering: I like dogs.",
    "clustering: My house was destroyed in an earthquake.",
    "clustering: I adore dogs.",
    "clustering: I despise dogs.",
    "clustering: Yesterday was a beautiful sunny day.",
    "clustering: The weather turned gloomy unexpectedly.",
    "clustering: I wish it would rain tomorrow.",
    "clustering: My car broke down in the middle of nowhere.",
    "clustering: I love my car.",
    "clustering: I really love my car.",
    "clustering: The concert was the best ever.",
    "clustering: The concert was a complete disaster.",
    "clustering: I enjoy reading mystery novels.",
    "clustering: I detest reading long boring textbooks.",
    "clustering: The city park is wonderful for a morning walk.",
    "clustering: The city park seems neglected and dirty.",
    "clustering: I am excited about the upcoming festival.",
    "clustering: I am not looking forward to the upcoming festival.",
    "clustering: The movie was both thrilling and inspiring.",
    "clustering: The movie was dull and uninspiring.",
    "clustering: I believe that kindness can change the world.",
    "clustering: I doubt that kindness makes a difference.",
    "clustering: The cake was delicious and moist.",
    "clustering: The cake was dry and tasteless.",
    "clustering: I plan to visit the new art museum this weekend.",
    "clustering: I do not plan to visit the new art museum this weekend.",
    "clustering: My computer runs smoothly after the update.",
    "clustering: My computer crashes frequently after the update.",
    "clustering: I enjoy working out at the gym.",
    "clustering: I dislike going to the gym.",
]

displayed = [
    "clustering: I hate dogs.",
    "clustering: i hate dogs",
    "clustering: I like dogs.",
    "clustering: My house was destroyed in an earthquake.",
    "clustering: I adore dogs.",
    "clustering: I despise dogs.",
    "clustering: Yesterday was a beautiful sunny day.",
    "clustering: My car broke down in the middle of nowhere.",
    "clustering: I love my car.",
    "clustering: The concert was the best ever.",
    "clustering: I really love my car.",
    "clustering: The concert was a complete disaster.",
    "clustering: The city park seems neglected and dirty.",
    "clustering: I enjoy working out at the gym.",
    "clustering: I dislike going to the gym.",
]

# Initialize the SentenceTransformer model.
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
embeddings = model.encode(sentences)
print("Embeddings:\n", model.encode(displayed[:2]))

# For small datasets (e.g., with 30 samples) use a lower n_neighbors if desired.
n_neighbors = min(15, len(embeddings) - 1)

# Initialize UMAP with random initialization to avoid spectral issues.
reducer = umap.UMAP(
    n_components=2, random_state=42, n_neighbors=n_neighbors, init="random"
)

# Transform embeddings to 2D.
embedding_2d = reducer.fit_transform(embeddings)

# Build a DataFrame for the 2D projections.
df = pd.DataFrame(
    {
        "UMAP_Component_1": embedding_2d[:, 0],
        "UMAP_Component_2": embedding_2d[:, 1],
        "Sentence": sentences,
    }
)

# Set Seaborn style for publication quality.
# sns.set_theme(style="whitegrid", font_scale=1.5)
#
# # Create the scatterplot.
# plt.figure(figsize=(12, 8))
# ax = sns.scatterplot(
#     data=df,
#     x="UMAP_Component_1",
#     y="UMAP_Component_2",
#     s=100,
#     color="skyblue",
#     edgecolor="coral",
# )
#
# # Annotate each point with its corresponding sentence; adjust offset for clarity.
# for _, row in df[df["Sentence"].isin(displayed)].iterrows():
#     plt.text(
#         row["UMAP_Component_1"] + 0.01,
#         row["UMAP_Component_2"] + 0.01,
#         row["Sentence"],
#         fontsize=15,
#     )
#
# # Update the axis labels and title for scientific clarity.
# plt.xlabel("UMAP Component 1 (arbitrary units)", fontsize=25)
# plt.ylabel("UMAP Component 2 (arbitrary units)", fontsize=25)
# # plt.title("2D Projection of Sentence Embeddings via UMAP", fontsize=16)
# plt.tight_layout()
#
# plt.show()

# 1.  Seaborn / Matplotlib defaults tuned for print
sns.set_theme(
    style="ticks",  # subtler grid
    font="DejaVu Sans",  # common cross-platform sans serif
    font_scale=1.8,  # globally bigger text
    rc={
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    },
)

# 2.  Figure: large, high-dpi, white background (reports love vector PDFs too)
fig, ax = plt.subplots(figsize=(15, 10), dpi=120, facecolor="white")

# 3.  Scatter: colour-blind friendly palette + thick outline for contrast
sns.scatterplot(
    data=df,
    x="UMAP_Component_1",
    y="UMAP_Component_2",
    s=160,
    linewidth=0.8,
    edgecolor="black",
    palette="colorblind",
    hue=None,  # or pass a categorical column if you want colour groups
    ax=ax,
)

# 4.  Text annotations – boxed, legible, non-overlapping
texts = []
for _, row in df[df["Sentence"].isin(displayed)].iterrows():
    txt = ax.text(
        row["UMAP_Component_1"],
        row["UMAP_Component_2"],
        row["Sentence"],
        ha="left",
        va="bottom",
        fontsize=16,
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="white",
            ec="skyblue",
            lw=0.5,
            alpha=0.8,
        ),
    )
    texts.append(txt)

# 5.  Let adjustText nudge labels until nothing overlaps
adjust_text(
    texts,
    arrowprops=dict(arrowstyle="-", lw=0.5, color="coral", alpha=0.8),
    ax=ax,
    expand_points=(1.5, 2),
)

# 6.  Final cosmetic touches
ax.set_xlabel("UMAP Component 1", labelpad=12)
ax.set_ylabel("UMAP Component 2", labelpad=12)
sns.despine(ax=ax, trim=True)
fig.tight_layout()

plt.show()
