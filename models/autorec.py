import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import ndcg_score

class UserEncoder(nn.Module):
    def __init__(self, layer_sizes, n_movies, f):
        super(UserEncoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=n_movies, out_features=layer_sizes[0]) if i == 0
                                     else nn.Linear(in_features=layer_sizes[i - 1], out_features=layer_sizes[i])
                                     for i in range(len(layer_sizes))])

        if f == 'identity':
            self.f = lambda x: x
        elif f == 'sigmoid':
            self.f = nn.Sigmoid()
        elif f == 'relu':
            self.f = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for f : {}".format(f))

    def forward(self, input, raw_last_layer=False):
        for (i, layer) in enumerate(self.layers):
            if raw_last_layer and i == len(self.layers) - 1:
                input = layer(input)
            else:
                input = self.f(layer(input))
        return input


class AutoRec(nn.Module):
    def __init__(self,
                 n_movies,
                 params,
                 resume=None):
        super(AutoRec, self).__init__()
        self.params = params
        self.cuda_available = torch.cuda.is_available()
        self.n_movies = n_movies
        self.layer_sizes = params['layer_sizes']
        if params['g'] == 'identity':
            self.g = lambda x: x
        elif params['g'] == 'sigmoid':
            self.g = nn.Sigmoid()
        elif params['g'] == 'relu':
            self.g = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for g : {}".format(params['g']))

        if resume is not None:
            if self.cuda_available:
                checkpoint = torch.load(resume, weights_only=False)
            else:
                checkpoint = torch.load(resume, map_location=lambda storage, loc: storage, weights_only=False)
            
            if "layer_sizes" in checkpoint:
                self.layer_sizes = checkpoint["layer_sizes"]
            self.encoder = UserEncoder(layer_sizes=self.layer_sizes, n_movies=n_movies, f=params['f'])
            self.user_representation_size = self.layer_sizes[-1]
            self.decoder = nn.Linear(in_features=self.user_representation_size, out_features=n_movies)
            model_dict = self.state_dict()
            
            model_dict.update({k: v for k, v in checkpoint['state_dict'].items()
                               if k != "encoder.layers.0.weight" and "decoder" not in k})
            
            encoder0weight = checkpoint["state_dict"]["encoder.layers.0.weight"][:, :self.n_movies]
            decoderweight = checkpoint["state_dict"]["decoder.weight"][:self.n_movies, :]
            decoderbias = checkpoint["state_dict"]["decoder.bias"][:self.n_movies]
            
            if encoder0weight.shape[1] < self.n_movies:
                tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
                encoder0weight = torch.cat((
                    encoder0weight,
                    torch.zeros(encoder0weight.shape[0], self.n_movies - encoder0weight.shape[1], out=tt())), dim=1)
                decoderweight = torch.cat((
                    decoderweight,
                    torch.zeros(self.n_movies - decoderweight.shape[0], decoderweight.shape[1], out=tt())), dim=0)
                decoderbias = torch.cat((
                    decoderbias, torch.zeros(self.n_movies - decoderbias.shape[0], out=tt())), dim=0)
            model_dict.update({
                "encoder.layers.0.weight": encoder0weight,
                "decoder.weight": decoderweight,
                "decoder.bias": decoderbias,
            })
            self.load_state_dict(model_dict)
        else:
            self.encoder = UserEncoder(layer_sizes=self.layer_sizes, n_movies=n_movies, f=params['f'])
            self.user_representation_size = self.layer_sizes[-1]
            self.decoder = nn.Linear(in_features=self.user_representation_size, out_features=n_movies)

        if self.cuda_available:
            self.cuda()

    def forward(self, input, additional_context=None, range01=True):
        encoded = self.encoder(input, raw_last_layer=True)
        if additional_context is not None:
            encoded = self.encoder.f(encoded + additional_context)
        else:
            encoded = self.encoder.f(encoded)
        
        if range01:
            return self.g(self.decoder(encoded))
        else:
            return self.decoder(encoded)

    def evaluate(self, batch_loader, criterion, subset, batch_input):
        self.eval()
        batch_loader.batch_index[subset] = 0
        n_batches = batch_loader.n_batches[subset]

        losses = []
        ndcg_scores = []

        for _ in tqdm(range(n_batches)):
            batch = batch_loader.load_batch(subset=subset, batch_input=batch_input)
            if self.cuda_available:
                batch["input"] = batch["input"].cuda()
                batch["target"] = batch["target"].cuda()
            
            with torch.no_grad():
                output = self.forward(batch["input"])
                loss = criterion(output, batch["target"])
                losses.append(loss.item())

                predictions = output.cpu().detach().numpy()
                targets = batch["target"].cpu().detach().numpy()

                targets_relevance = np.copy(targets)
                targets_relevance[targets_relevance == -1] = 0
                
                if np.sum(targets_relevance) > 0:
                    try:
                        score = ndcg_score(targets_relevance, predictions, k=10)
                        ndcg_scores.append(score)
                    except ValueError:
                        pass

        final_loss = criterion.normalize_loss_reset(np.sum(losses))
        final_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        print("{} loss with input={} : {:.4f} | nDCG@10 : {:.4f}".format(subset, batch_input, final_loss, final_ndcg))
        
        self.train()
        return final_loss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.nb_observed_targets = 0

    def forward(self, input, target):
        mask = (target != -1)
        observed_input = torch.masked_select(input, mask)
        observed_target = torch.masked_select(target, mask)
        self.nb_observed_targets += len(observed_target)
        loss = self.mse_loss(observed_input, observed_target)
        return loss

    def normalize_loss_reset(self, loss):
        if self.nb_observed_targets == 0:
            return 0
        n_loss = loss / self.nb_observed_targets
        self.nb_observed_targets = 0
        return n_loss
