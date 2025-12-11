import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os

from models.decoders import SwitchingDecoder
from utils import sort_for_packed_sequence

import config
from models.hierarchical_rnn import HRNN
from models.autorec import AutoRec

class RecommendFromDialogue(nn.Module):
    def __init__(self,
                 n_movies,
                 params,
                 cuda_available,
                 autorec_path=os.path.join(config.AUTOREC_MODEL, "model_best")):
        super(RecommendFromDialogue, self).__init__()
        self.n_movies = n_movies
        self.cuda_available = cuda_available
        
        self.autorec = AutoRec(
            params=params['autorec_params'],
            n_movies=self.n_movies,
            resume=autorec_path if os.path.exists(autorec_path) else None
        )

    def forward(self, dialogue, movie_occurrences, recommend_new_movies, user_representation=None):
        batch_size, max_conv_length = dialogue.shape[:2]
        
        if self.cuda_available:
            autorec_input = torch.zeros(batch_size, max_conv_length, self.n_movies).cuda()
        else:
            autorec_input = torch.zeros(batch_size, max_conv_length, self.n_movies)

        output = self.autorec(autorec_input, additional_context=user_representation, range01=False)
        return output


class Recommender(nn.Module):
    def __init__(self,
                 train_vocab,
                 n_movies,
                 params,
                 ):
        super(Recommender, self).__init__()
        self.params = params
        self.train_vocab = train_vocab
        self.n_movies = n_movies
        self.cuda_available = torch.cuda.is_available()

        self.encoder = HRNN(params=params['hrnn_params'],
                            train_vocabulary=train_vocab,
                            gensen=False,
                            train_gensen=False,
                            conv_bidirectional=False)
                            
        self.recommender_module = RecommendFromDialogue(
            n_movies=n_movies,
            params=params['recommend_from_dialogue_params'],
            cuda_available=self.cuda_available
        )

        if params['language_aware_recommender']:
            self.language_to_user = nn.Linear(in_features=params['hrnn_params']['conversation_encoder_hidden_size'],
                                              out_features=self.recommender_module.autorec.user_representation_size)
        
        latent_layer_sizes = params['latent_layer_sizes']
        if latent_layer_sizes is not None:
            latent_variable_size = latent_layer_sizes[-1]
            self.prior_hidden_layers = nn.ModuleList(
                [nn.Linear(in_features=params['hrnn_params']['conversation_encoder_hidden_size'],
                           out_features=latent_layer_sizes[0]) if i == 0
                 else nn.Linear(in_features=latent_layer_sizes[i - 1], out_features=latent_layer_sizes[i])
                 for i in range(len(latent_layer_sizes) - 1)])
            penultimate_size = params['hrnn_params']['conversation_encoder_hidden_size'] \
                if len(latent_layer_sizes) == 1 else latent_layer_sizes[-2]
            self.mu_prior = nn.Linear(penultimate_size, latent_variable_size)
            self.sigma_prior = nn.Linear(penultimate_size, latent_variable_size)

            posterior_input_size = params['hrnn_params']['conversation_encoder_hidden_size'] +\
                                   2 * params['hrnn_params']['sentence_encoder_hidden_size'] + 1
            self.posterior_hidden_layers = nn.ModuleList(
                [nn.Linear(in_features=posterior_input_size,
                           out_features=latent_layer_sizes[0]) if i == 0
                 else nn.Linear(in_features=latent_layer_sizes[i - 1], out_features=latent_layer_sizes[i])
                 for i in range(len(latent_layer_sizes) - 1)])
            penultimate_size = posterior_input_size if len(latent_layer_sizes) == 1 else latent_layer_sizes[-2]
            self.mu_posterior = nn.Linear(penultimate_size, latent_variable_size)
            self.sigma_posterior = nn.Linear(penultimate_size, latent_variable_size)

        context_size = params['hrnn_params']['conversation_encoder_hidden_size']
        if latent_layer_sizes is not None:
            context_size += latent_layer_sizes[-1]
            
        self.decoder = SwitchingDecoder(
            context_size=context_size,
            vocab_size=len(train_vocab),
            **params['decoder_params']
        )

        if self.cuda_available:
            self.cuda()

    def reparametrize(self, mu, logvariance):
        std = torch.exp(0.5 * logvariance)
        if self.cuda_available:
            eps = torch.randn_like(std).cuda()
        else:
            eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_dict, return_latent=False):
        conversation_representations, sentence_representations = self.encoder(
            input_dict, return_all=True, return_sentence_representations=True)
            
        batch_size, max_conversation_length = input_dict["dialogue"].shape[:2]
        max_utterance_length = input_dict["dialogue"].shape[2]

        if self.params['language_aware_recommender']:
            user_rep_from_language = self.language_to_user(conversation_representations)
        else:
            user_rep_from_language = None
            
        movie_recommendations = self.recommender_module(
            dialogue=input_dict["dialogue"],
            movie_occurrences=input_dict["movie_occurrences"],
            recommend_new_movies=False,
            user_representation=user_rep_from_language
        )

        utterances = input_dict["dialogue"].view(batch_size * max_conversation_length, -1)
        lengths = input_dict["lengths"]
        
        lengths_flat = lengths.reshape((-1))
        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(lengths_flat, cuda=self.cuda_available)
        
        if isinstance(sorted_lengths, np.ndarray):
            sorted_lengths = torch.from_numpy(sorted_lengths)
        sorted_lengths = sorted_lengths.cpu()

        sorted_utterances = utterances.index_select(0, sorted_idx)

        if self.cuda_available:
            pad_tensor = torch.zeros(batch_size, 1, self.params['hrnn_params']['conversation_encoder_hidden_size']).cuda()
        else:
            pad_tensor = torch.zeros(batch_size, 1, self.params['hrnn_params']['conversation_encoder_hidden_size'])
            
        conversation_representations = torch.cat((pad_tensor, conversation_representations), 1).narrow(
            1, 0, max_conversation_length)
            
        conversation_representations = conversation_representations.contiguous().view(
            batch_size * max_conversation_length, self.params['hrnn_params']['conversation_encoder_hidden_size'])\
            .index_select(0, sorted_idx)

        if self.cuda_available:
            pad_rec = torch.zeros(batch_size, 1, self.n_movies).cuda()
        else:
            pad_rec = torch.zeros(batch_size, 1, self.n_movies)
            
        movie_recommendations = torch.cat((pad_rec, movie_recommendations), 1).narrow(
            1, 0, max_conversation_length)
        movie_recommendations = movie_recommendations.contiguous().view(
            batch_size * max_conversation_length, -1).index_select(0, sorted_idx)

        num_positive_lengths = int((sorted_lengths > 0).sum())
        sorted_utterances = sorted_utterances[:num_positive_lengths]
        sorted_lengths = sorted_lengths[:num_positive_lengths]
        conversation_representations = conversation_representations[:num_positive_lengths]
        movie_recommendations = movie_recommendations[:num_positive_lengths]

        if self.params['latent_layer_sizes'] is not None:
            h_prior = conversation_representations
            for layer in self.prior_hidden_layers:
                h_prior = F.relu(layer(h_prior))
            mu_prior = self.mu_prior(h_prior)
            logvar_prior = self.sigma_prior(h_prior)
            
            sentence_representations = sentence_representations.view(
                batch_size * max_conversation_length, -1).index_select(0, sorted_idx)
            sentence_representations = sentence_representations[:num_positive_lengths]
            
            h_posterior = torch.cat((conversation_representations, sentence_representations), 1)
            for layer in self.posterior_hidden_layers:
                h_posterior = F.relu(layer(h_posterior))
            mu_posterior = self.mu_posterior(h_posterior)
            logvar_posterior = self.sigma_posterior(h_posterior)

            mu, logvar = (mu_posterior, logvar_posterior) if self.training else (mu_prior, logvar_prior)
            z = self.reparametrize(mu, logvar)
            context = torch.cat((conversation_representations, z), 1)
        else:
            context = conversation_representations
            mu_prior, logvar_prior, mu_posterior, logvar_posterior = None, None, None, None

        outputs = self.decoder(
            sorted_utterances,
            sorted_lengths,
            context,
            movie_recommendations,
            log_probabilities=True,
            sample_movies=False
        )

        if num_positive_lengths < batch_size * max_conversation_length:
            missing_count = batch_size * max_conversation_length - num_positive_lengths
            vocab_dim = len(self.train_vocab) + self.n_movies
            if self.cuda_available:
                pad_out = torch.zeros(missing_count, max_utterance_length, vocab_dim).cuda()
            else:
                pad_out = torch.zeros(missing_count, max_utterance_length, vocab_dim)
            outputs = torch.cat((outputs, pad_out), 0)

        outputs = outputs.index_select(0, rev).view(batch_size, max_conversation_length, max_utterance_length, -1)
        
        if return_latent:
            return outputs, mu_prior, logvar_prior, mu_posterior, logvar_posterior
        return outputs

    def train_iter(self, batch, criterion, kl_coefficient=1):
        self.train()
        if self.params['latent_layer_sizes'] is not None:
            outputs, mu_prior, logvar_prior, mu_posterior, logvar_posterior = self.forward(batch, return_latent=True)
        else:
            outputs = self.forward(batch, return_latent=False)

        batch_size, max_conv_length, max_seq_length, vocab_size = outputs.data.shape
        
        mask = (batch["senders"].view(-1) == -1)
        idx = torch.nonzero(mask).squeeze()
        
        if idx.numel() == 0:
            return 0.0

        outputs = outputs.view(-1, max_seq_length, vocab_size).index_select(0, idx)
        target = batch["target"].view(-1, max_seq_length).index_select(0, idx)

        loss = criterion(outputs.view(-1, vocab_size), target.view(-1))

        if self.params['latent_layer_sizes'] is not None:
            kld = .5 * (-1 + logvar_prior - logvar_posterior +
                        (torch.exp(logvar_posterior) + (mu_posterior - mu_prior).pow(2)) / torch.exp(logvar_prior))
            kld = torch.mean(torch.sum(kld, -1))
            loss += kl_coefficient + kld
            
        loss.backward()
        return loss.item()

    def evaluate(self, batch_loader, criterion, subset="valid"):
        self.eval()
        batch_loader.batch_index[subset] = 0
        n_batches = batch_loader.n_batches[subset]

        losses = []
        for _ in tqdm(range(n_batches)):
            batch = batch_loader.load_batch(subset=subset)
            if self.cuda_available:
                batch["dialogue"] = batch["dialogue"].cuda()
                batch["target"] = batch["target"].cuda()
                batch["senders"] = batch["senders"].cuda()
            
            with torch.no_grad():
                outputs = self.forward(batch)

            batch_size, max_conv_length, max_seq_length, vocab_size = outputs.data.shape
            
            mask = (batch["senders"].view(-1) == -1)
            idx = torch.nonzero(mask).squeeze()
            
            if idx.numel() == 0: continue

            outputs = outputs.view(-1, max_seq_length, vocab_size).index_select(0, idx)
            target = batch["target"].view(-1, max_seq_length).index_select(0, idx)

            loss = criterion(outputs.view(-1, vocab_size), target.view(-1))
            losses.append(loss.item())
            
        print("{} loss : {}".format(subset, np.mean(losses) if losses else 0))
        self.train()
        return np.mean(losses) if losses else 0