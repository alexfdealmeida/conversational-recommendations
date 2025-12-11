from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse
import os

from models.recommender_model import Recommender
from sequence_generator import SequenceGenerator
from batch_loaders.batch_loader import DialogueBatchLoader
from utils import load_model
from beam_search import get_best_beam
import test_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--save_path")
    parser.add_argument("--beam_size", default=10, type=int)
    parser.add_argument("--n_examples", default=10, type=int)
    parser.add_argument("--only_best", default="True",
                        help="whether to display all the beam results, or only the best")
    parser.add_argument("--full_dialogue", default="True",
                        help="whether to display the full dialogue or only the answers from the model")
    parser.add_argument("--subset", default="test",
                        help="subset on which to condition the model")
    args = parser.parse_args()

    # Converter strings para booleanos
    args.only_best = str(args.only_best).lower() == "true"
    args.full_dialogue = str(args.full_dialogue).lower() == "true"

    temperatures = [1]
    batch_loader = DialogueBatchLoader(
        sources="dialogue movie_occurrences movieIds_in_target",
        batch_size=1
    )
    rec = Recommender(
        batch_loader.train_vocabulary,
        batch_loader.n_movies,
        params=test_params.recommender_params
    )
    
    # Verifica se o modelo existe antes de carregar
    if os.path.isfile(args.model_path):
        load_model(rec, args.model_path)
    else:
        print(f"AVISO: Checkpoint não encontrado em '{args.model_path}'.")
        print("Certifique-se de ter executado 'python train_recommender.py' com sucesso antes de gerar respostas.")
        exit(1)

    batch_loader.set_word2id(rec.encoder.word2id)
    generator = SequenceGenerator(
        rec.decoder,
        beam_size=args.beam_size,
        word2id=batch_loader.word2id,
        movie_id2name=batch_loader.id2name,
        max_sequence_length=40
    )
    batch_loader.batch_index[args.subset] = 0

    # START
    with open(args.save_path, "w", encoding='utf-8') as f:
        f.write("")
        
    for _ in tqdm(range(args.n_examples)):
        # Load batch
        batch_index = batch_loader.batch_index[args.subset]
        batch = batch_loader.load_batch(subset=args.subset)
        if rec.cuda_available:
            batch["dialogue"] = batch["dialogue"].cuda()
            batch["target"] = batch["target"].cuda()
            batch["senders"] = batch["senders"].cuda()

        # 1) Compute the contexts and recommendation vectors
        # encoder result: (conv_length, hidden_size)
        conversation_representations = rec.encoder(batch, return_all=True).squeeze(0)
        
        # get movie_recommendations
        movie_recommendations = rec.recommender_module(
            dialogue=batch["dialogue"],
            movie_occurrences=batch["movie_occurrences"],
            recommend_new_movies=True,
            user_representation=None 
        ).squeeze(0)  # (conv_length, n_movies)
        
        conv_length = movie_recommendations.shape[0]

        # select contexts after seeker's utterances
        # indices of seeker's utterances(< conv_len)
        # Fix: nonzero com as_tuple=False e squeeze correto
        mask = (batch["senders"].view(-1) == 1)
        idx = torch.nonzero(mask, as_tuple=False).squeeze()
        
        if idx.numel() == 0:
            continue
            
        if rec.cuda_available:
            idx = idx.cuda()
            
        conversation_representations = conversation_representations.index_select(0, idx)
        movie_recommendations = movie_recommendations.index_select(0, idx)
        
        # if first utterance is recommender, add a 0-context at the beginning
        if batch["senders"].cpu().flatten()[0] == -1:
            if rec.cuda_available:
                pad_conv = torch.zeros((1, rec.params["hrnn_params"]["conversation_encoder_hidden_size"])).cuda()
                pad_rec = torch.zeros((1, rec.n_movies)).cuda()
            else:
                pad_conv = torch.zeros((1, rec.params["hrnn_params"]["conversation_encoder_hidden_size"]))
                pad_rec = torch.zeros((1, rec.n_movies))
                
            conversation_representations = torch.cat((pad_conv, conversation_representations), 0)
            movie_recommendations = torch.cat((pad_rec, movie_recommendations), 0)

        # Latent variable
        if rec.params['latent_layer_sizes'] is not None:
            h_prior = conversation_representations
            for layer in rec.prior_hidden_layers:
                h_prior = F.relu(layer(h_prior))
            mu_prior = rec.mu_prior(h_prior)
            logvar_prior = rec.sigma_prior(h_prior)
            
            mu, logvar = (mu_prior, logvar_prior)
            z = rec.reparametrize(mu, logvar)

            context = torch.cat((conversation_representations, z), 1)
        else:
            context = conversation_representations

        # 2) generate sentences conditioned on the contexts and recommendation vectors
        index = 0
        if args.full_dialogue:
            output_str = "CONVERSATION {} \n".format(batch_index)
        else:
            output_str = ""
            
        # Fix: .tolist() direto
        messages = [[batch_loader.id2word[w] for w in sentence[:length]]
                    for (sentence, length) in zip(batch["dialogue"][0].cpu().tolist(), batch["lengths"][0])]
                    
        # keep track of movies mentioned by the model
        mentioned_movies = set()
        
        senders_flat = batch["senders"][0].cpu().numpy()
        
        for (i, msg) in enumerate(messages):
            if senders_flat[i] == -1:  # sent by recommender: generate response
                if args.full_dialogue:
                    output_str += "GROUND TRUTH: " + " ".join(msg) + "\n"
                for temperature in temperatures:
                    # BEAM SEARCH
                    # Verifica se o indice não ultrapassa o tamanho do contexto
                    if index >= len(context):
                        break
                        
                    beams = generator.beam_search(
                        [batch_loader.word2id["<s>"]],
                        forbid_movies=mentioned_movies,
                        # add batch dimension
                        context=context[index].unsqueeze(0),
                        movie_recommendations=movie_recommendations[index].unsqueeze(0),
                        sample_movies=True,
                        temperature=temperature
                    )
                    if args.only_best:
                        # add best beam
                        best_beam = get_best_beam(beams)
                        if args.full_dialogue:
                            output_str += "GENERATED T={}: ".format(temperature)
                        output_str += best_beam.get_string(batch_loader.id2word) + "\n"
                        # update set of mentioned movies
                        mentioned_movies.update(best_beam.mentioned_movies)
                        print("mentioned movies", mentioned_movies)
                    else:
                        # show all beams sorted by likelihood
                        sorted_beams = sorted(beams, key=lambda b: -b.likelihood)
                        for (beam_rank, beam) in enumerate(sorted_beams):
                            if args.full_dialogue:
                                output_str += "GENERATED T={}, nb {}: ".format(temperature, beam_rank)
                            output_str += beam.get_string(batch_loader.id2word) + "\n"
                index += 1
            else:  # sent by seeker
                if args.full_dialogue:
                    output_str += "SEEKER: " + " ".join(msg) + "\n"
        output_str += "\n"
        with open(args.save_path, "a", encoding='utf-8') as f:
            f.write(output_str)

