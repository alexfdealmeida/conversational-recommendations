# movie-dialogue-dev (BERT Version)

This repository contains an adapted version of the code for NeurIPS 2018 paper "Towards Deep Conversational Recommendations" https://arxiv.org/abs/1812.07617.

Update: This version replaces the deprecated GenSen encoder with BERT (SBERT via sentence-transformers) for sentence embeddings.

## Requirements

- Python 3.x
- PyTorch
- sentence-tranformers
- tqdm
- nltk
- h5py
- numpy
- scikit-learn

## Usage

### 1. Get the data
Get ReDial data from https://github.com/ReDialData/website/tree/data and Movielens data https://grouplens.org/datasets/movielens/latest/. Note that for the paper we retrieved the Movielens
data set in December 2025. The Movielens latest dataset has been updated since then.
```
# Clone repository
git clone https://github.com/alexfdealmeida/conversational-recommendations.git
cd conversational-recommendations

# Install dependencies
pip install -r requirements.txt
python -m nltk.downloader punkt

# Create directories
mkdir -p redial movielens data

# Download ReDial
wget -O redial/redial_dataset.zip https://github.com/ReDialData/website/raw/data/redial_dataset.zip
# Download MovieLens
wget -O movielens/ml-latest.zip http://files.grouplens.org/datasets/movielens/ml-latest.zip

# Split ReDial data
python scripts/split-redial.py redial/
mv redial/test_data.jsonl redial/test_data

# Split Movielens data
python scripts/split-movielens.py movielens/
```

### 2. Match Movie Entities

Merge the movie lists by matching the movie names from ReDial and Movielens. Note that this will create an intermediate file `movies_matched.csv`, which is deleted at the end of the script.
```
python scripts/match_movies.py --redial_movies_path=redial/movies_with_mentions.csv --ml_movies_path=movielens/ml-latest/movies.csv --destination=redial/movies_merged.csv
```

### 3. Configuration

In `config.py`, ensure the paths point to your data folders. Note: The GenSen paths are no longer required as BERT is downloaded automatically.

- `MODELS_PATH`: Folder where trained models will be saved.
- `REDIAL_DATA_PATH`: Folder containing `train_data`, `valid_data`, and `test_data`.
- `ML_DATA_PATH`: Folder containing `train_ratings`, `valid_ratings`, and `test_ratings`.

### 4. Train models

Note on BERT: The first time you run these scripts, the BERT model (e.g., all-mpnet-base-v2) will be downloaded automatically by sentence-transformers.
- Train sentiment analysis: (Optional if using Recommender only) Trains a model to predict movie form labels from ReDial.
```
python train_sentiment_analysis.py
```
- Train autoencoder recommender system: Pre-trains an Autoencoder on Movielens, then fine-tunes it on ReDial.
```
python train_autorec.py
```
- Train conversational recommendation model: Trains the hierarchical RNN (HRNN) using BERT embeddings and the recommender module.
```
python train_recommender.py
```

### 5. Generate sentences
`generate_responses.py` loads a trained model and generates responses for the test set.
```
python generate_responses.py --model_path=/path/to/models/recommender/model_best --save_path=generations
```
