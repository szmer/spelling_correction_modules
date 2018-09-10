import torch, os
import torch.nn as nn
from ELMoForManyLangs.src.modules.token_embedder import ConvTokenEmbedder
from ELMoForManyLangs.src.modules.elmo import ElmobiLm

class Model(nn.Module):
  def __init__(self, config, word_embedding, char_embedding, use_cuda=False):
    super(Model, self).__init__()
    self.use_cuda = use_cuda
    self.config = config

    self.token_embedder = ConvTokenEmbedder(config, word_embedding, char_embedding, use_cuda)
    self.char_embedding = char_embedding # for the decoder
    self.encoder = ElmobiLm(config, use_cuda)
    self.output_dim = config['encoder']['projection_dim'] # of the encoder
    # in the input to the first layer we use 3 to collapse all ELMo layers into one
    self.lstm_hidden_size = 512
    self.lstm_layers_n = 2
    self.decoder_feeder = nn.Sequential(nn.Linear(config['encoder']['projection_dim']*2*3,
                                                     self.lstm_hidden_size*self.lstm_layers_n*2),
                                           nn.ReLU())
    self.decoder_lstm = nn.LSTM(input_size=char_embedding.n_d, # the input char
                                hidden_size=self.lstm_hidden_size,
                                num_layers=self.lstm_layers_n,
                                bidirectional=True,
                                batch_first=True)
    self.decoder_decision = nn.Sequential(nn.Linear(self.lstm_hidden_size*2,
                                                    char_embedding.embedding.num_embeddings),
                                          nn.LogSoftmax(dim=0))

  def forward(self, word_inp, chars_package, mask):
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

    # note that we do no masking of padded stuff, so LSTM can output seqs that are as long as it wants
    if self.use_cuda:
      chars_package = chars_package.cuda()
    embedded_chars = self.char_embedding(chars_package[:, 1, :]) # here we want the token w/o markers also
    # Use the encoder output to produce a correction.
    # (collapse ELMo layers, taking only the middle token (throw away markers):)
    # the second index goes over batch members
    decoder_input = torch.cat([encoder_output[0, :, 1, :], encoder_output[1, :, 1, :], encoder_output[2, :, 1, :]], dim=1)
    decoder_init_state = self.decoder_feeder(decoder_input)
    decoder_init_state = decoder_init_state.view(self.lstm_layers_n*2, # bidirectional
                                                 chars_package.size(0), # batch size
                                                 self.lstm_hidden_size)
    decoder_output, hidden = self.decoder_lstm(embedded_chars, (decoder_init_state, decoder_init_state))
    # + maybe add an intermediate linear transformation?
    decoder_decisions = self.decoder_decision(decoder_output)
    return decoder_decisions

  def load_model(self, path):
    self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl'), map_location=lambda storage, loc: storage))
    self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl'), map_location=lambda storage, loc: storage))
