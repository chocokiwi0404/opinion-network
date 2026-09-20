"""
This script builds ONE opinion-similarity network over all 87 respondents
using their combined responses across all 60 survey questions (Technology,
Education, Society & Ethics, Environment). The four domains are NOT built
as four separate networks — they are used afterwards to characterise the
communities that emerge from the single combined network.
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import community as community_louvain
from collections import Counter
from itertools import combinations

RNG_SEED = 42

# 1. Load data and define domains
df_raw = pd.read_csv("Survey_Results_UC_cleaned.csv")

opinion_cols = [c for c in df_raw.columns if c.startswith(("T", "E", "S", "V"))]

# T = Technology, E = Education
# S = Society & Ethics, V = Environment
domain_cols = {
    "Technology"       : [c for c in opinion_cols if c.startswith("T")],
    "Education"        : [c for c in opinion_cols if c.startswith("E")],
    "Society & Ethics" : [c for c in opinion_cols if c.startswith("S")],
    "Environment"      : [c for c in opinion_cols if c.startswith("V")],
}

print("\nDataset Summary")
print("-" * 40)
print(f"Total respondents : {len(df_raw)}")
print(f"Opinion questions : {len(opinion_cols)}")
for domain, cols in domain_cols.items():
    print(f"{domain:18s}: {len(cols)} questions")

# Check for missing values
missing = df_raw[opinion_cols].isnull().sum()
total_missing = missing.sum()
total_cells = len(df_raw) * len(opinion_cols)
print(f"\nMissing values : {total_missing}/{total_cells} "
      f"({100*total_missing/total_cells:.2f}%)")
print("Missing values are replaced using the median of each question.")

# 2. Clean data
df = df_raw[["respondent_id"] + opinion_cols].copy()
df = df.set_index("respondent_id")

assert df.apply(pd.api.types.is_numeric_dtype).all(), \
    "Non-numeric columns found — check opinion_cols"

zero_var = df.columns[df.nunique() <= 1].tolist()
if zero_var:
    print(f"\nRemoving zero-variance questions: {zero_var}")
    df = df.drop(columns=zero_var)
else:
    print("\nNo zero-variance questions found")

dupes = df.duplicated().sum()
print(f"Duplicate response patterns : {dupes}")

df = df.apply(lambda col: col.fillna(col.median()), axis=0)
print(f"Clean data : {df.shape[0]} respondents x {df.shape[1]} questions")

# 3. Check for possible reverse-coded questions
# Strong negative correlations may indicate a reverse-coded item.
print("\nReverse-coding check")
reverse_flags = []
for dname, cols in domain_cols.items():
    corr = df[cols].corr()
    for c1, c2 in combinations(cols, 2):
        r = corr.loc[c1, c2]
        if r < -0.25:
            reverse_flags.append((dname, c1, c2, round(r, 3)))

if reverse_flags:
    for dname, c1, c2, r in reverse_flags:
        print(f"  {dname}: {c1} - {c2} (corr={r})")
    print("No questions were reverse-coded.")
else:
    print("No candidate reverse-coded pairs found.")

# 4. Anonymize respondent IDs
respondents_raw = df.index.tolist()
n = len(respondents_raw)

rng = np.random.default_rng(RNG_SEED)
shuffled_ranks = rng.permutation(n)
id_map = {respondents_raw[i]: f"P{shuffled_ranks[i]+1:03d}" for i in range(n)}
respondents = [id_map[r] for r in respondents_raw]

df.index = respondents          # replace index with anonymous IDs
df.index.name = "anon_id"

# Keep the mapping locally only; do not submit it.
pd.DataFrame({"respondent_id": respondents_raw,
              "anon_id": respondents}).to_csv(
    "id_mapping_DO_NOT_SUBMIT.csv", index=False)
print(f"\nRespondent IDs changed to P001-P{n:03d}")

# 5. Standardize responses and calculate cosine similarity
# Standardization makes the similarity depend on response patterns.
scaler = StandardScaler()
opinion_matrix = scaler.fit_transform(df.values.astype(float))

sim_matrix = cosine_similarity(opinion_matrix)  # range [-1, 1]
upper_tri_vals = sim_matrix[np.triu_indices(n, k=1)]

print("\nSimilarity Matrix")
print(f"Shape : {sim_matrix.shape}")
print(f"Range : {sim_matrix.min():.4f} to {sim_matrix.max():.4f}")
print(f"Mean  : {sim_matrix.mean():.4f}")

# 6. Compare different similarity thresholds
print("\nThreshold Sensitivity")
print("-" * 65)
print(f"{'Pctl':>5} {'Thresh':>8} {'Edges':>7} {'Density':>8} "
      f"{'Comps':>6} {'Isolates':>9} {'Modularity':>11}")

sensitivity_rows = []
for pct in [60, 70, 75, 80, 90]:
    thr = np.percentile(upper_tri_vals, pct)
    Gt = nx.Graph()
    Gt.add_nodes_from(respondents)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= thr:
                Gt.add_edge(respondents[i], respondents[j],
                            weight=float(sim_matrix[i, j]))
    comps = nx.number_connected_components(Gt)
    isolates = len(list(nx.isolates(Gt)))
    dens = nx.density(Gt)
    edges = Gt.number_of_edges()

    if edges > 0:
        part_t = community_louvain.best_partition(Gt, weight="weight",
                                                    random_state=RNG_SEED)
        mod_t = community_louvain.modularity(part_t, Gt, weight="weight")
    else:
        mod_t = float("nan")

    print(f"{pct:>4}% {thr:>8.4f} {edges:>7} {dens:>8.4f} "
          f"{comps:>6} {isolates:>9} {mod_t:>11.4f}")
    sensitivity_rows.append({
        "percentile": pct, "threshold": round(thr, 4), "edges": edges,
        "density": round(dens, 4), "components": comps,
        "isolates": isolates, "modularity": round(mod_t, 4),
    })

pd.DataFrame(sensitivity_rows).to_csv("threshold_sensitivity.csv", index=False)
print("\n75th percentile selected for the final network.")

# 7. Build the final network
THRESHOLD = np.percentile(upper_tri_vals, 75)

G = nx.Graph()
for r in respondents:
    G.add_node(r)

for i in range(n):
    for j in range(i + 1, n):
        sim = sim_matrix[i, j]
        if sim >= THRESHOLD:
            G.add_edge(respondents[i], respondents[j],
                       weight=float(sim), distance=float(1 - sim))

avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
is_connected = nx.is_connected(G)
largest_cc_nodes = max(nx.connected_components(G), key=len)

# Calculate path length on the largest connected component.
G_lcc = G.subgraph(largest_cc_nodes)
avg_spl = nx.average_shortest_path_length(G_lcc, weight="distance")

print("\nNetwork Summary")
print("-" * 40)
print(f"Nodes                          : {G.number_of_nodes()}")
print(f"Edges                          : {G.number_of_edges()}")
print(f"Average degree                 : {avg_degree:.2f}")
print(f"Density                        : {nx.density(G):.4f}")
print(f"Is connected                   : {is_connected}")
print(f"Connected components           : {nx.number_connected_components(G)}")
print(f"Largest component size         : {len(largest_cc_nodes)} / {n}")
print(f"Avg weighted shortest path len : {avg_spl:.4f}  "
      f"(on largest component; weight = 1 - similarity)")
print(f"Threshold                      : {THRESHOLD:.4f} (75th percentile)")

# 8. Calculate centrality measures
# Path-based measures use distance = 1 - similarity.
degree_centrality = nx.degree_centrality(G)          # unweighted: # of connections
strength = dict(G.degree(weight="weight"))            # weighted: sum of similarity
betweenness_centrality = nx.betweenness_centrality(G, weight="distance")
closeness_centrality = nx.closeness_centrality(G, distance="distance")
clustering_coeff = nx.clustering(G, weight="weight")

print("\nCentrality Measures")
print("-" * 40)
print(f"Avg degree centrality : {np.mean(list(degree_centrality.values())):.4f}")
print(f"Avg weighted strength : {np.mean(list(strength.values())):.4f}")
print(f"Avg clustering coeff  : {np.mean(list(clustering_coeff.values())):.4f}")

top5 = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 respondents by degree centrality:")
for node, val in top5:
    print(f"  {node}: {val:.4f}  (strength={strength[node]:.3f})")

# 9. Detect communities using Louvain
partition = community_louvain.best_partition(G, weight="weight",
                                              random_state=RNG_SEED)
num_communities = len(set(partition.values()))
modularity = community_louvain.modularity(partition, G, weight="weight")
community_sizes = Counter(partition.values())

print("\nCommunity Detection")
print("-" * 40)
print(f"Communities detected : {num_communities}")
print(f"Modularity            : {modularity:.4f}")
for cid, size in sorted(community_sizes.items()):
    print(f"  Community {cid} -> {size} respondents")
small_comms = [c for c, s in community_sizes.items() if s <= 3]
if small_comms:
    print(f"\n  Note: communities {small_comms} have <=3 members. These are")
    print("  likely outlier respondents rather than substantive opinion")
    print("  clusters and should not be over-interpreted.")

# 10. Compare communities across the four domains
print("\nDomain Analysis")
print("-" * 40)

# Mean response for each domain
domain_scores = pd.DataFrame(index=df.index)
for domain, cols in domain_cols.items():
    valid = [c for c in cols if c in df.columns]
    domain_scores[domain] = df[valid].mean(axis=1)

# Correlation between domain scores
domain_corr = domain_scores.corr()
print("\nCorrelation between domain-average opinions across respondents:")
print(domain_corr.round(3).to_string())
domain_corr.to_csv("domain_correlations.csv")

# Compare how much each domain differs between communities.
domain_scores["community"] = [partition[r] for r in domain_scores.index]
print("\nCommunity separation by domain:")
eta_sq = {}
grand_mean = domain_scores[list(domain_cols.keys())].mean()
for domain in domain_cols.keys():
    ss_total = ((domain_scores[domain] - grand_mean[domain]) ** 2).sum()
    ss_between = 0.0
    for cid, group in domain_scores.groupby("community"):
        ss_between += len(group) * (group[domain].mean() - grand_mean[domain]) ** 2
    eta_sq[domain] = ss_between / ss_total if ss_total > 0 else 0.0
    print(f"  {domain:18s}: eta^2 = {eta_sq[domain]:.4f}")

pd.DataFrame([eta_sq]).to_csv("domain_eta_squared.csv", index=False)

# Average domain score for each community
print("\nCommunity profiles:")
domain_profiles = {}
for cid in sorted(community_sizes.keys()):
    members = domain_scores[domain_scores["community"] == cid]
    row = {"community": cid, "size": len(members)}
    for domain in domain_cols.keys():
        row[domain] = round(members[domain].mean(), 3)
    domain_profiles[cid] = row

profile_df = pd.DataFrame(domain_profiles.values())
print(profile_df.to_string(index=False))
profile_df.to_csv("community_profiles.csv", index=False)
print("\nSaved domain analysis files.")

# 11. Save node metrics
metrics_df = pd.DataFrame({
    "anon_id"               : list(degree_centrality.keys()),
    "community"             : [partition[k] for k in degree_centrality.keys()],
    "degree_centrality"     : [round(degree_centrality[k], 4) for k in degree_centrality],
    "weighted_strength"     : [round(strength[k], 4) for k in degree_centrality],
    "betweenness_centrality": [round(betweenness_centrality[k], 4) for k in degree_centrality],
    "closeness_centrality"  : [round(closeness_centrality[k], 4) for k in degree_centrality],
    "clustering_coeff"      : [round(clustering_coeff[k], 4) for k in degree_centrality],
})
metrics_df.to_csv("node_metrics.csv", index=False)
print("Saved node_metrics.csv")

# 12. Save network summary
summary = {
    "nodes"                       : G.number_of_nodes(),
    "edges"                       : G.number_of_edges(),
    "average_degree"              : round(avg_degree, 4),
    "density"                     : round(nx.density(G), 4),
    "threshold"                   : round(THRESHOLD, 4),
    "threshold_percentile"        : 75,
    "similarity_metric"           : "cosine_similarity (standardized inputs)",
    "is_connected"                : is_connected,
    "num_components"              : nx.number_connected_components(G),
    "largest_component_size"      : len(largest_cc_nodes),
    "avg_weighted_shortest_path"  : round(avg_spl, 4),
    "avg_clustering"              : round(np.mean(list(clustering_coeff.values())), 4),
    "num_communities"             : num_communities,
    "modularity"                  : round(modularity, 4),
    "avg_degree_centrality"       : round(np.mean(list(degree_centrality.values())), 4),
    "avg_weighted_strength"       : round(np.mean(list(strength.values())), 4),
}
pd.DataFrame([summary]).to_csv("network_summary.csv", index=False)
print("Saved network_summary.csv")

# 13. Save the graph
for node in G.nodes():
    G.nodes[node]["community"]         = partition[node]
    G.nodes[node]["degree_centrality"] = round(degree_centrality[node], 4)
    G.nodes[node]["strength"]          = round(strength[node], 4)

nx.write_graphml(G, "opinion_network.graphml")
print("Saved opinion_network.graphml")

# 14. Create visualizations
pos = nx.spring_layout(G, seed=RNG_SEED, k=0.5)

cmap_comm = matplotlib.colormaps["tab20"].resampled(num_communities)
node_colors = [cmap_comm(partition[node]) for node in G.nodes()]
COLORS = [cmap_comm(c) for c in range(num_communities)]
node_sizes = [200 + 1800 * degree_centrality[node] for node in G.nodes()]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Opinion Network — Survey Respondents (anonymized)",
             fontsize=15, fontweight="bold")

ax = axes[0]
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.12, edge_color="gray", width=0.5)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                       node_size=node_sizes, alpha=0.88)
handles = [plt.Line2D([0], [0], marker='o', color='w',
           markerfacecolor=COLORS[c], markersize=9,
           label=f"Community {c} (n={community_sizes[c]})")
           for c in sorted(community_sizes)]
ax.legend(handles=handles, fontsize=8, loc="upper left")
ax.set_title(f"Community structure | {num_communities} communities | "
             f"modularity={modularity:.3f}\nNode size = degree centrality "
             f"(structural position, not importance)", fontsize=10)
ax.axis("off")

ax2 = axes[1]
domains = list(domain_cols.keys())
cids = sorted(community_sizes.keys())
heat = np.array([[domain_profiles[c][d] for d in domains] for c in cids])
im = ax2.imshow(heat, cmap="YlOrRd", aspect="auto", vmin=1, vmax=5)
ax2.set_xticks(range(len(domains))); ax2.set_xticklabels(domains, fontsize=9)
ax2.set_yticks(range(len(cids)))
ax2.set_yticklabels([f"Community {c} (n={community_sizes[c]})" for c in cids],
                    fontsize=9)
for i in range(len(cids)):
    for j in range(len(domains)):
        ax2.text(j, i, f"{heat[i,j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax2, label="Mean Likert response (1-5)")
ax2.set_title("Community opinion profiles by domain\n"
              "(post-hoc characterization of the single combined network)",
              fontsize=10)

plt.tight_layout()
plt.savefig("opinion_network.png", dpi=150, bbox_inches="tight")
print("Saved opinion_network.png")

sens_df = pd.read_csv("threshold_sensitivity.csv")
fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4))
fig2.suptitle("Threshold Sensitivity Analysis", fontsize=13, fontweight="bold")
metrics_to_plot = [("edges", "#4C8BE0", "Edge count"),
                   ("density", "#50B87A", "Graph density"),
                   ("components", "#E05C4C", "Connected components"),
                   ("modularity", "#9B59B6", "Modularity")]
for ax, (col, color, title) in zip(axes2, metrics_to_plot):
    ax.plot(sens_df["percentile"], sens_df[col], marker="o", color=color)
    ax.axvline(75, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("Percentile"); ax.set_title(title)
plt.tight_layout()
plt.savefig("threshold_sensitivity.png", dpi=150, bbox_inches="tight")
print("Saved threshold_sensitivity.png")

print("\nAll outputs saved")
for f in ["opinion_network.graphml", "opinion_network.png",
          "threshold_sensitivity.png", "node_metrics.csv",
          "community_profiles.csv", "domain_correlations.csv",
          "domain_eta_squared.csv", "network_summary.csv",
          "threshold_sensitivity.csv"]:
    print(f"  {f}")
print("\nNote: id_mapping_DO_NOT_SUBMIT.csv is for local use only.")