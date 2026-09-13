# EvoScore: Zero-Shot Enzyme Variant Ranking

EvoScore is a machine learning pipeline that predicts the physical stability and fitness of enzyme mutations using zero-shot evolutionary scoring. By bypassing the need for expensive, time-consuming wet-lab initial screenings, EvoScore acts as a computational filter to triage highly destructive variants before synthesis.

## Validation & Performance
EvoScore was benchmarked against the TEM-1 beta-lactamase deep mutational scanning dataset (Firnberg et al., 2014):
* **Dataset Size:** 4,783 unique variants 
* **Spearman Rank Correlation (ρ):** 0.738
* **Compute Time:** ~15 minutes on a single NVIDIA T4 GPU