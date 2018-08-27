import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ELMoForManyLangs.src.modules.token_embedder import ConvTokenEmbedder
from ELMoForManyLangs.src.modules.elmo import ElmobiLm

class Model(nn.Module):
  def __init__(self, config, word_embedding, char_embedding, use_cuda=False):
    super(Model, self).__init__()
    self.use_cuda = use_cuda
    self.config = config

    self.token_embedder = ConvTokenEmbedder(config, word_embedding, char_embedding, use_cuda)
    self.encoder = ElmobiLm(config, use_cuda)
    self.output_dim = config['encoder']['projection_dim'] # of the encoder
    # in the input to the first layer we use 3 to collapse all ELMo layers into one
    self.decoder = nn.Sequential(nn.Linear(config['encoder']['projection_dim']*2*3,
                                           config['encoder']['projection_dim']*2),
                                 nn.ReLU(),
                                 nn.Linear(config['encoder']['projection_dim']*2,
                                           (config['token_embedder']['max_characters_per_token']
                                            * char_embedding.embedding.num_embeddings)),
                                 nn.ReLU(),
                                 nn.LogSoftmax(dim=0))

  def forward(self, word_inp, chars_package, mask):
####    token_embeddings = []
####    for word_n in range(word_inp.size(0)):
####      # this is linear transformation embedding_dim -> projection_dim
####      # the mask is read as batch_size, seq_len
####      token_embedding = self.token_embedder(word_inp[word_n], chars_package,
####                                            (word_inp[word_n].size(0), word_inp[word_n].size(1)))
####      token_embeddings.append(token_embedding)
####    token_embeddings = torch.stack(tuple(token_embeddings))
    token_embedding = self.token_embedder(word_inp, chars_package,
                                          (word_inp.size(0), word_inp.size(1)))
    if self.use_cuda:
      mask = mask.cuda()
    # the encoder reads mask as "batch_size, total_sequence_length"
    encoder_output = self.encoder(token_embedding, mask)

    # the output has shape: (num_layers, batch_size, sequence_length, hidden_size - which is also the projection size)
    sz = encoder_output.size()
    # this reshapes duplicated token_embedding to (1, batch_size, sequence_length, hidden_size)
    token_embedding = torch.cat([token_embedding, token_embedding], dim=2).view(1, sz[1], sz[2], sz[3])
    encoder_output = torch.cat([token_embedding, encoder_output], dim=0)

    # Use the encoder output to produce a correction.
    # (collapse ELMo layers, taking only the middle token (throw away markers):)
    # the second index goes over batch members
    decoder_input = torch.cat([encoder_output[0, :, 1, :], encoder_output[1, :, 1, :], encoder_output[2, :, 1, :]], dim=1)
    char_predictions = self.decoder(decoder_input)

    return char_predictions
