import config

# Model hyper parameters
decoder_params = {
    "hidden_size": config.BERT_EMBEDDING_SIZE,
    "num_layers": 1,
    "peephole": False
}

hrnn_params = {
    'use_movie_occurrences': False,
    'sentence_encoder_hidden_size': config.BERT_EMBEDDING_SIZE,
    'conversation_encoder_hidden_size': config.BERT_EMBEDDING_SIZE,
    'sentence_encoder_num_layers': 1,
    'conversation_encoder_num_layers': 1,
    'use_dropout': False,
}

hred_params = {
    'decoder_params': decoder_params,
    "hrnn_params": hrnn_params
}

sentiment_analysis_params = {
    'hrnn_params': {
        'use_movie_occurrences': 'word',
        'sentence_encoder_hidden_size': config.BERT_EMBEDDING_SIZE,
        'conversation_encoder_hidden_size': config.BERT_EMBEDDING_SIZE,
        'sentence_encoder_num_layers': 2,
        'conversation_encoder_num_layers': 2,
        'use_dropout': 0.4,
    }
}

autorec_params = {
    'layer_sizes': [1000],
    'f': "sigmoid",
    'g': "sigmoid",
}

recommend_from_dialogue_params = {
    "sentiment_analysis_params": sentiment_analysis_params,
    "autorec_params": autorec_params
}

recommender_params = {
    'decoder_params': decoder_params,
    'hrnn_params': hrnn_params,
    'recommend_from_dialogue_params': recommend_from_dialogue_params,
    'latent_layer_sizes': None,
    'language_aware_recommender': False,
}

# Training params
train_sa_params = {
    "learning_rate": 0.001,
    "batch_size": 32, # Default: 16
    "nb_epochs": 30, # Default: 50
    "patience": 5,
    "weight_decay": 0,
    "use_class_weights": True,
    "cut_dialogues": -1,
    "targets": "suggested seen liked"
}

train_autorec_params = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "nb_epochs": 30, # Default: 50
    "patience": 5,
    "batch_input": "random_noise",
    "max_num_inputs": 10000 
}

train_recommender_params = {
    "learning_rate": 0.001,
    "batch_size": 8, # Default: 4
    "nb_epochs": 30, # Default: 50
    "patience": 5,
}