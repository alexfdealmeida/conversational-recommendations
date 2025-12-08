import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import os
import config
from models.bert_wrapper import BertEncoderWrapper # Novo wrapper
from utils import sort_for_packed_sequence

class HRNN(nn.Module):
    def __init__(self,
                 params,
                 gensen=False, # Mantive o nome do argumento para compatibilidade
                 train_vocabulary=None,
                 train_gensen=True,
                 conv_bidirectional=False):
        super(HRNN, self).__init__()
        self.params = params
        self.use_bert = True # Forçamos BERT
        self.cuda_available = torch.cuda.is_available()
        
        # Vocab para reconstrução de texto (ID -> String)
        self.id2word = {idx: word for idx, word in enumerate(train_vocabulary)}
        self.word2id = {word: idx for idx, word in enumerate(train_vocabulary)}

        # Inicializa BERT Wrapper
        self.bert = BertEncoderWrapper(config.BERT_MODEL_NAME, device='cuda' if self.cuda_available else 'cpu')
        
        # Conversation Encoder (GRU)
        # Input size agora é o output do BERT
        input_size_conv = self.bert.output_dim
        
        if self.params['use_movie_occurrences'] == "sentence":
            input_size_conv += 1
        
        # Adiciona info do sender
        input_size_conv += 1 

        self.conversation_encoder = nn.GRU(
            input_size=input_size_conv,
            hidden_size=self.params['conversation_encoder_hidden_size'],
            num_layers=self.params['conversation_encoder_num_layers'],
            batch_first=True,
            bidirectional=conv_bidirectional
        )
        
        if self.params['use_dropout']:
            self.dropout = nn.Dropout(p=self.params['use_dropout'])

    def ids_to_sentences(self, dialogue_ids, lengths):
        """Reconstroi sentenças de texto a partir dos IDs para o BERT"""
        sentences = []
        dialogue_cpu = dialogue_ids.cpu().numpy()
        for i, seq in enumerate(dialogue_cpu):
            # Pega apenas os tokens válidos baseados no length
            valid_len = lengths[i]
            if valid_len == 0:
                sentences.append("")
                continue
            
            # Reconstrói a string
            words = [self.id2word.get(idx, '<unk>') for idx in seq[:valid_len]]
            # Remove tokens especiais se necessário
            words = [w for w in words if w not in ['<s>', '</s>', '<pad>']]
            sentences.append(" ".join(words))
        return sentences

    def get_sentence_representations(self, dialogue, senders, lengths, movie_occurrences=None):
        batch_size, max_conversation_length = dialogue.shape[:2]
        
        # Flatten para processar sentenças
        flat_dialogue = dialogue.view(-1, dialogue.shape[-1]) #(total_sentences, seq_len)
        flat_lengths = lengths.reshape(-1)
        
        # Otimização: processar apenas sentenças com tamanho > 0
        non_zero_mask = flat_lengths > 0
        active_indices = torch.nonzero(non_zero_mask).squeeze()
        
        if len(active_indices.shape) == 0: # Caso raro de batch vazio
             active_dialogue = flat_dialogue
             active_lengths = flat_lengths
        else:
            active_dialogue = flat_dialogue[active_indices]
            active_lengths = flat_lengths[active_indices]

        # 1. Converter IDs para Texto
        text_sentences = self.ids_to_sentences(active_dialogue, active_lengths)
        
        # 2. Passar pelo BERT (recebe lista de strings, retorna tensor)
        # O resultado já é (num_sentences, 768) - pooled embedding
        bert_embeddings = self.bert.get_sentence_embeddings(text_sentences)
        
        # 3. Restaurar formato (Preencher com zeros as sentenças vazias)
        if self.cuda_available:
            sentence_representations = torch.zeros(batch_size * max_conversation_length, self.bert.output_dim).cuda()
        else:
            sentence_representations = torch.zeros(batch_size * max_conversation_length, self.bert.output_dim)
            
        if len(active_indices.shape) > 0:
            sentence_representations[active_indices] = bert_embeddings

        # 4. Dropout
        if self.params['use_dropout']:
            sentence_representations = self.dropout(sentence_representations)

        # 5. Reshape para (batch, conv_len, hidden)
        sentence_representations = sentence_representations.view(batch_size, max_conversation_length, -1)
        
        # 6. Adicionar sender info (append na dimensão de features)
        sentence_representations = torch.cat([sentence_representations, senders.unsqueeze(2)], 2)

        # 7. Adicionar movie occurence se necessário
        if self.params['use_movie_occurrences'] == "sentence":
            if movie_occurrences is None:
                raise ValueError("Please specify movie occurrences")
            sentence_representations = torch.cat((sentence_representations, movie_occurrences.unsqueeze(2)), 2)

        return sentence_representations

    def forward(self, input_dict, return_all=True, return_sentence_representations=False):
        movie_occurrences = input_dict["movie_occurrences"] if self.params['use_movie_occurrences'] else None
        
        # BERT substitui o sentence encoder antigo
        sentence_representations = self.get_sentence_representations(
            input_dict["dialogue"], input_dict["senders"], lengths=input_dict["lengths"],
            movie_occurrences=movie_occurrences)

        # Passar pela GRU da conversa
        lengths = input_dict["conversation_lengths"]
        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(lengths, self.cuda_available)

        sorted_representations = sentence_representations.index_select(0, sorted_idx)
        packed_sequences = pack_padded_sequence(sorted_representations, sorted_lengths.cpu(), batch_first=True)
        
        conversation_representations, last_state = self.conversation_encoder(packed_sequences)

        conversation_representations, _ = pad_packed_sequence(conversation_representations, batch_first=True)
        conversation_representations = conversation_representations.index_select(0, rev)
        
        last_state = last_state.index_select(1, rev)
        
        if self.params['use_dropout']:
            conversation_representations = self.dropout(conversation_representations)
            last_state = self.dropout(last_state)
            
        if return_all:
            if not return_sentence_representations:
                return conversation_representations
            else:
                return conversation_representations, sentence_representations
        else:
            if self.conversation_encoder.bidirectional:
                last_state = torch.cat((last_state[-1], last_state[-2]), 1)
                return last_state
            else:
                return last_state[-1]