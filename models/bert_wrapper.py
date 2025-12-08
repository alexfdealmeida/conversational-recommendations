import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer

class BertEncoderWrapper(nn.Module):
    def __init__(self, model_name='all-mpnet-base-v2', device=None):
        super().__init__()
        print(f"Loading SBERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        if device:
            self.model.to(device)
        self.device = device
        
        # Dimensão do output do modelo (768 para mpnet, 384 para MiniLM)
        self.output_dim = self.model.get_sentence_embedding_dimension()
        
        # Atributos dummy para compatibilidade
        self.task_word2id = {} 

    def vocab_expansion(self, vocab):
        # BERT usa tokenizador próprio, não precisa de expansão de vocabulário manual
        pass

    def get_sentence_embeddings(self, sentences_list):
        """
        Recebe lista de strings e retorna tensor (batch, hidden_size)
        """
        # SBERT encode retorna numpy por padrão, convertemos para tensor
        embeddings = self.model.encode(sentences_list, convert_to_tensor=True, show_progress_bar=False)
        return embeddings