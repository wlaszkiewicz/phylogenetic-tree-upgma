# UPGMA Phylogenetic Tree Construction

Python implementation of the UPGMA (Unweighted Pair Group Method with Arithmetic Mean) algorithm for phylogenetic tree construction, with a PyQt5 GUI and ete3-based tree visualization.

## Features

- Two input modes: raw sequences (aligned via Center Star MSA) or pre-computed distance matrix
- Ultrametricity validation with user warning on violations
- Interactive graphical tree visualization with zoom and pan (ete3)
- Newick format export compatible with standard phylogenetic viewers
- Full results export: MSA (FASTA/txt), matrices (CSV/txt), tree (PNG, Newick)
- Configurable MSA scoring parameters

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Tech Stack

Python, PyQt5, NumPy, ete3

## Documentation

Full technical report including algorithm details, complexity analysis, and example analyses (related Cytochrome C sequences vs unrelated proteins) is available in [`docs/report.pdf`](docs/report.pdf).
