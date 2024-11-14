
# memo-canonical-novels 📚 

<a href="https://chc.au.dk"><img src="https://github.com/centre-for-humanities-computing/intra/raw/main/images/onboarding/CHC_logo-turquoise-full-name.png" width="25%" align="right"/></a>
[![cc](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)
[![cc](https://img.shields.io/badge/EMNLP-NLP4DH-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKAAAABwAgMAAADkn5ORAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAlQTFRFAAAA7R4k7RwkoncKaQAAAAF0Uk5TAEDm2GYAAAAJcEhZcwAACxMAAAsTAQCanBgAAAJVSURBVFjD7Zc7ksMwDEPTpMnptmGD023DhqdcApCT2NvYnJTxTH7K09CiSAi+3b7X23VHIQLxQbAiA/lREJFAzsCIyOQLUZVcQvCjYx9CnwZLYxwuENI0zUZNQUbtmw8mJHsePzt6z5qBDVX/lcxwpXZO6a4enoFKSFZTvRoCXIuwnIJgVKAa1JqgdTSHIahUdIq6CHQTTDTSuZ+B/ZuJ8ZqwtpOz+3ZmYJaKIXLVG/evv2pwBopy0MVoN8GJM1D5SFc/3AX9zowFZqAiKT+K6mrztCHIPoCaQcrEmGyJXmPmDOwKcBmk+iHUXKW5QzBiKwgo3SFFgMJPQQ7Gz/vYfaEzEJSRxO8O5NTCELTS1TF0/ivc06AkL2sX+hFWwiFIpas8gCGNmYFSThwTLiWoGahujUN6WgBZJ0NQ+o4d+IDldAaGy3aXHpZZK3XOQGvefgvvkqwYgrQEnfE9WBKCGIJs/8MWPiQEmIIUpWNRgCdoDUElgoUqRUke7kjLtIafAn0a1Flhped/9g2wQZLLyasgj9yKzXj55JT4ydvgJSznwfZuPVQRyxttJ6jekNdBtb+NUdhplY4O9jDLpa6DaQcn/1aAK8xHk/J1HWRM0CF5gg63lS2b2eugV2LpxMr+2td3m3QN1OazIGwNZbyWMrzq8SxYKzRrNL0OuTme9/I0cRVMLIte9gnhHijXbb0WcxpUPGCVlsL1Uio2h/N8WDkN2nLRc0HGCzov5Wwq483KnQZ1mtlZOtjm5NxwhQmY6zGiKFd6GGCr5noOyKvg9/peo+sPhLv+BGIWS+UAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTMtMDgtMDRUMjE6NTc6MjkrMDg6MDAj62PfAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDEzLTA4LTA0VDIxOjU3OjI5KzA4OjAwUrbbYwAAAABJRU5ErkJggg==)](https://aclanthology.org/2024.nlp4dh-1.14.pdf)


###

This repository contains code for making embeddings, plots and results for our paper: 

"Canonical Status and Literary Influence: A Comparative Study of Danish Novels from the Modern Breakthrough (1870–1900)" presented at NLP4DH at EMNLP 2024.

## Useful directions 📌

Some useful directions:
- `memo_canonical_novels/` the main folder contains the source code for the project, here you will find the makefile to create embeddings
- `notebooks/` contains the notebooks used for the analysis, `analysis.py` is the main notebook, `tfidf_comparison.py` is the notebook used to compare the embeddings with tf-idf. Other notebooks contain sanity checks.
- `figures/` contains the figures generated by the notebooks
- `data/` contains saved embeddings (.json) used for the analysis (and will contain generated embeddings if you generate them)

## Data & paper 📝

The dataset used is available at [huggingface](https://huggingface.co/datasets/MiMe-MeMo/Corpus-v1.1)

Please cite our [paper](https://aclanthology.org/2024.nlp4dh-1.14.pdf) if you use the code or the embeddings:

```
@inproceedings{feldkamp-etal-2024-canonical,
    title = "Canonical Status and Literary Influence: A Comparative Study of {D}anish Novels from the Modern Breakthrough (1870{--}1900)",
    author = "Feldkamp, Pascale  and
      Lassche, Alie  and
      Kostkan, Jan  and
      Kardos, M{\'a}rton  and
      Enevoldsen, Kenneth  and
      Baunvig, Katrine  and
      Nielbo, Kristoffer",
    editor = {H{\"a}m{\"a}l{\"a}inen, Mika  and
      {\"O}hman, Emily  and
      Miyagawa, So  and
      Alnajjar, Khalid  and
      Bizzoni, Yuri},
    booktitle = "Proceedings of the 4th International Conference on Natural Language Processing for Digital Humanities",
    month = nov,
    year = "2024",
    address = "Miami, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.nlp4dh-1.14",
    pages = "140--155"
}
```

## Project Organization 🏗️

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         memo_canonical_novels and configuration for tools like black
│
├── figures            <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src                <- Source code for use in this project, making embeddings.
    │
    ├── __init__.py             <- Makes memo_canonical_novels a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    └── pooling.py              <- Code to create average embeddings from raw embeddings
    │
    └── plots.py                <- Code to create visualizations
```

--------

