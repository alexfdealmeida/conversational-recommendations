import os

# Ajuste para o Colab
MODELS_PATH = '/content/models'
REDIAL_DATA_PATH = '/content/data'

# Cria pastas se não existirem
if not os.path.exists(MODELS_PATH): os.makedirs(MODELS_PATH)
if not os.path.exists(REDIAL_DATA_PATH): os.makedirs(REDIAL_DATA_PATH)

AUTOREC_MODEL = os.path.join(MODELS_PATH, "autorec")
SENTIMENT_ANALYSIS_MODEL = os.path.join(MODELS_PATH, 'sentiment_analysis')
RECOMMENDER_MODEL = os.path.join(MODELS_PATH, "recommender")

TRAIN_PATH = "train_data"
VALID_PATH = "valid_data"
TEST_PATH = "test_data"

MOVIE_PATH = os.path.join(REDIAL_DATA_PATH, "movies_merged.csv")
VOCAB_PATH = os.path.join(REDIAL_DATA_PATH, "vocabulary.p")

# Reddit paths (mantidos, mas opcionais)
REDDIT_PATH = "/content/reddit"
REDDIT_TRAIN_PATH = "task4_reddit_train.txt"
REDDIT_VALID_PATH = "task4_reddit_dev.txt"
REDDIT_TEST_PATH = "task4_reddit_test.txt"

CONVERSATION_LENGTH_LIMIT = 40
UTTERANCE_LENGTH_LIMIT = 80

# Movielens
ML_DATA_PATH = "/content/movielens"
ML_SPLIT_PATHS = [os.path.join(ML_DATA_PATH, f"split{i}") for i in range(5)]
ML_TRAIN_PATH = "train_ratings"
ML_VALID_PATH = "valid_ratings"
ML_TEST_PATH = "test_ratings"

# --- NOVA CONFIGURAÇÃO DO BERT ---
BERT_MODEL_NAME = 'all-mpnet-base-v2' # ou 'all-MiniLM-L6-v2' para ser mais rápido
BERT_EMBEDDING_SIZE = 768 # 768 para mpnet, 384 para MiniLM