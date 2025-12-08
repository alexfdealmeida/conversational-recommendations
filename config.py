import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMP_DIR = os.path.join(BASE_DIR, 'temp')
REDIAL_DATA_PATH = os.path.join(TEMP_DIR, 'data')
MODELS_PATH = os.path.join(TEMP_DIR, 'models')

# if not os.path.exists(REDIAL_DATA_PATH):
#     os.makedirs(REDIAL_DATA_PATH, exist_ok=True)
# if not os.path.exists(MODELS_PATH):
#     os.makedirs(MODELS_PATH, exist_ok=True)

AUTOREC_MODEL = os.path.join(MODELS_PATH, "autorec")
SENTIMENT_ANALYSIS_MODEL = os.path.join(MODELS_PATH, 'sentiment_analysis')
RECOMMENDER_MODEL = os.path.join(MODELS_PATH, "recommender")

TRAIN_PATH = "train_data"
VALID_PATH = "valid_data"
TEST_PATH = "test_data"

MOVIE_PATH = os.path.join(REDIAL_DATA_PATH, "movies_merged.csv")
VOCAB_PATH = os.path.join(REDIAL_DATA_PATH, "vocabulary.p")

# Movielens Settings
ML_DATA_PATH = os.path.join(REDIAL_DATA_PATH, "movielens")
ML_SPLIT_PATHS = [os.path.join(ML_DATA_PATH, f"split{i}") for i in range(5)]
ML_TRAIN_PATH = "train_ratings"
ML_VALID_PATH = "valid_ratings"
ML_TEST_PATH = "test_ratings"

# Reddit (Opcional)
REDDIT_PATH = os.path.join(REDIAL_DATA_PATH, "reddit")
REDDIT_TRAIN_PATH = "task4_reddit_train.txt"
REDDIT_VALID_PATH = "task4_reddit_dev.txt"
REDDIT_TEST_PATH = "task4_reddit_test.txt"

# BERT Settings
BERT_MODEL_NAME = 'all-mpnet-base-v2' # or 'all-MiniLM-L6-v2' to be faster
BERT_EMBEDDING_SIZE = 768 # 768 for mpnet, 384 for MiniLM

CONVERSATION_LENGTH_LIMIT = 40
UTTERANCE_LENGTH_LIMIT = 80