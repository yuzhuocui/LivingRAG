# LivingRAG

Official code for **LivingRAG: Augmenting Graph RAG with Experience**, accepted to Findings of EMNLP 2026.

[Project page](https://yuzhuocui.github.io/LivingRAG/) · [Paper](https://arxiv.org/abs/2608.25960)

## Setup

```bash
conda create -n living-rag --override-channels -c conda-forge python=3.9 pip -y
conda activate living-rag
bash scripts/setup.sh
python scripts/prepare.py --dataset all --models
```

## Run

Set your API key, then run a dataset:

```bash
export OPENAI_API_KEY="your-own-key"
bash scripts/run.sh 2wikimultihop
```

Dataset names: `2wikimultihop`, `hotpotqa`, `musique`, `musique_full`, `wixqa`.

## Acknowledgments and license

The retrieval backbone builds on [LinearRAG](https://github.com/DEEP-PolyU/LinearRAG).
Modifications for LivingRAG (2026-10-27) include experience-based retrieval and write-back, data preparation, and release scripts.
This project is licensed under [GNU GPL v3.0](LICENSE.txt). The upstream license and copyright notice are retained in `LICENSE.txt`.
