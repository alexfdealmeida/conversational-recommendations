import torch
import torch.nn as nn
import os
import config
from models.hierarchical_rnn import HRNN
from models.autorec import AutoRec
# from sentiment_analysis import SentimentAnalysis # Se for usar, atualize similarmente
from models.decoders import SwitchingDecoder
from utils import sort_for_packed_sequence

class Recommender(nn.Module):
    def __init__(self, train_vocab, n_movies, params):
        super(Recommender, self).__init__()
        self.params = params
        self.n_movies = n_movies
        self.cuda_available = torch.cuda.is_available()

        # Instancia HRNN (que agora usa BERT internamente)
        self.encoder = HRNN(
            params=params['hrnn_params'],
            train_vocabulary=train_vocab,
            gensen=False, # Não usamos mais gensen externo
            train_gensen=False,
            conv_bidirectional=False
        )
        
        # Módulo de Recomendação (AutoRec)
        # Simplificação: removendo SentimentAnalysis complexo por enquanto para focar na migração do BERT
        # Se precisar do SentimentAnalysis, ele também precisará ser portado para usar o HRNN com BERT
        self.recommender_module = RecommendFromDialogue(
             n_movies=n_movies,
             params=params,
             cuda_available=self.cuda_available
        )

        # Camadas auxiliares
        if params['language_aware_recommender']:
             self.language_to_user = nn.Linear(
                 in_features=params['hrnn_params']['conversation_encoder_hidden_size'],
                 out_features=self.recommender_module.autorec.user_representation_size
             )

        # Decoder (Generation)
        context_size = params['hrnn_params']['conversation_encoder_hidden_size']
        self.decoder = SwitchingDecoder(
            context_size=context_size,
            vocab_size=len(train_vocab),
            **params['decoder_params']
        )
        
        if self.cuda_available:
            self.cuda()

    def forward(self, input_dict, return_latent=False):
        # Encoder (BERT + GRU)
        conversation_representations, sentence_representations = self.encoder(
            input_dict, return_all=True, return_sentence_representations=True)
            
        # Recommender
        user_rep = None
        if self.params['language_aware_recommender']:
            user_rep = self.language_to_user(conversation_representations)
            
        movie_recommendations = self.recommender_module(
            dialogue=input_dict["dialogue"], # Passamos, mas o BERT já processou no encoder
            movie_occurrences=input_dict["movie_occurrences"],
            recommend_new_movies=False,
            user_representation=user_rep
        )

        # Decoder Logic (Simplificada para brevidade, mantendo lógica original)
        # ... (O código do decoder original pode ser mantido quase igual, 
        #      apenas atente para shapes se mudou hidden_size)
        
        return movie_recommendations # Retornando apenas recs por enquanto para teste

# Classe auxiliar simplificada para recomendação
class RecommendFromDialogue(nn.Module):
    def __init__(self, n_movies, params, cuda_available):
        super(RecommendFromDialogue, self).__init__()
        self.n_movies = n_movies
        self.autorec = AutoRec(params=params['recommend_from_dialogue_params']['autorec_params'], n_movies=n_movies)
        self.cuda_available = cuda_available

    def forward(self, dialogue, movie_occurrences, recommend_new_movies, user_representation=None):
        # Lógica simplificada: AutoRec puro baseado nas ocorrências
        # No código original, havia SentimentAnalysis aqui. 
        # Para o "MVP" da migração, focamos em rodar o pipeline.
        
        batch_size, max_conv_length = dialogue.shape[:2]
        autorec_input = torch.zeros(batch_size, max_conv_length, self.n_movies)
        if self.cuda_available: autorec_input = autorec_input.cuda()
        
        # Preenchimento dummy baseado em ocorrências (placeholder para lógica complexa)
        # O ideal é migrar a classe SentimentAnalysis para usar o BERT também.
        
        output = self.autorec(autorec_input, additional_context=user_representation, range01=False)
        return output