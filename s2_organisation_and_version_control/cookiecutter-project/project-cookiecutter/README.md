Project-Cookiecutter
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Execution scripts
------------
```make data <raw_data_dir> <processed_data_dir>``` <br>
Script for preparing the raw data for the training scripts. It finds all the .npz files in the raw data directory and
converts it into a Pytorch Tensor format. The tensors are stored in the processed data directory.

```make train <processed_data_dir>``` <br>
Train a neural network on the provided processed data. The trained network is store in the models directory.

```make predict <model_pt> <external_data_npy>``` <br>
Train a neural network on the provided external data. This should be a numpy representation of the MNIST dataset.

```make visualize <model_pt> <processed_data_dir> <image_ix_to_project> <projection_layer> <tsne_sample_limit>``` <br>
Makes three visualisations:
- Feature map: A single image (selected using `image_ix_to_project` as index on MNIST dataset) projected on a single
convolutional layer (on the specified `projection_layer`)
- 2d t-SNE representation of the data on one of the convolutional layers (~`projection_layer`). `tsne_sample_limit`
should be specified since it's an heavy operations. The limit sets the number of samples that should be used for the
calculation.
- Filters: a representation of the filters of the given convolutional layer (~`projection_layer`)


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
