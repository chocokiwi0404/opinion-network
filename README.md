# Opinion Network from Survey Responses

**Team:** NetFlux — Dynamic Processes and Complex Networks, Assignment 1

This repository builds and analyses an **opinion-similarity network** from an anonymised
class survey (60 Likert-scale items across Technology, Education, Society & Ethics, and
Environment). Each of the 87 retained respondents is a node; an edge connects two
respondents whose standardised response vectors are highly similar (cosine similarity,
75th-percentile threshold). See `report.pdf` for the full write-up, methodology, and
discussion of results.

## Repository structure

```
Assignment-1/
├── report.pdf                        # Final report (LaTeX-compiled PDF)
├── DPCN_Assignment_1.pdf             # Original assignment brief (for reference)
│
├── code/
│   ├── data_clean.py                 # Cleans, encodes, and anonymises the raw survey data
│   └── build_network.py              # Builds the network, runs all analysis, generates outputs
│
├── data/
│   ├── Survey_Results_UC_raw.csv     # Original, unfiltered survey export (96 respondents)
│   └── Survey_Results_UC_cleaned.csv # Cleaned, encoded survey data (87 respondents x 60 items)
│
└── outputs/
    ├── network_summary.csv           # Summary statistics of the final network
    ├── node_metrics.csv              # Per-respondent centrality metrics + community assignment
    ├── community_profiles.csv        # Mean domain-level opinion score per community
    ├── domain_correlations.csv       # Pearson correlation between domain-average scores
    ├── domain_eta_squared.csv        # eta^2 of each domain with respect to community membership
    ├── threshold_sensitivity.csv     # Edge count / density / components / modularity per percentile
    ├── opinion_network.png           # Network plot + community-domain heat map (Figure 2)
    ├── threshold_sensitivity.png     # Threshold sensitivity analysis (Figure 1)
    ├── opinion_network.graphml       # Final network in GraphML format (Gephi / Cytoscape)
    └── Network.pdf                   # Force-directed rendering of the network (Gephi export)
```

`data/` holds only the survey inputs (raw and cleaned); everything the pipeline
*generates* — tables and figures alike — lives in `outputs/`.

## How to run

1. **Install dependencies**
   ```bash
   pip install pandas numpy networkx scikit-learn python-louvain matplotlib
   ```

2. **Clean the raw survey data**
   ```bash
   python code/data_clean.py
   ```
   Reads `data/Survey_Results_UC_raw.csv`, encodes Likert responses to integers 1–5,
   treats blanks and "No Comments" as missing, drops completely empty or near-empty
   respondents, anonymises respondent IDs (fixed random seed = 42), and writes
   `data/Survey_Results_UC_cleaned.csv`.

3. **Build and analyse the network**
   ```bash
   python code/build_network.py
   ```
   Reads `data/Survey_Results_UC_cleaned.csv` and:
   - standardises each item and computes pairwise cosine similarity,
   - runs the threshold sensitivity sweep (60th–90th percentile),
   - builds the final network at the 75th-percentile threshold,
   - runs Louvain community detection (fixed random seed = 42),
   - computes centrality metrics, domain correlations, and eta^2 statistics,
   - writes all result tables and figures to `outputs/`.

All outputs in this repository were generated with the fixed random seed described above,
so re-running both scripts should reproduce every number and figure in `report.pdf`
exactly.

## Privacy note

The mapping between original and anonymised respondent IDs is intentionally **not**
included in this repository, to protect respondent privacy.
