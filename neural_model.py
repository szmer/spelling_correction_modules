import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self, char_embedding, use_cuda=False, bidirectional=False):
    super(Model, self).__init__()
    self.use_cuda = use_cuda

    self.char_embedding = char_embedding # for the decoder
    self.lstm_hidden_size = 512
    self.lstm_layers_n = 2
    self.bidirectional = bidirectional
    self.decoder_lstm = nn.LSTM(input_size=char_embedding.embedding_dim, # the input char
                                hidden_size=self.lstm_hidden_size,
                                num_layers=self.lstm_layers_n,
                                bidirectional=self.bidirectional,
                                batch_first=True)
    self.decoder_decision = nn.Sequential(nn.Linear(self.lstm_hidden_size*(2 if self.bidirectional else 1),
                                                    char_embedding.num_embeddings),
                                          nn.LogSoftmax(dim=0))

  def forward(self, chars_package):
    if self.use_cuda:
      chars_package = chars_package.cuda()
    embedded_chars = self.char_embedding(chars_package)
    # The initial state of LSTM is some noise.
    decoder_init_state = torch.randn(self.lstm_layers_n*(2 if self.bidirectional else 1),
                                     chars_package.size(0), # batch size
                                     self.lstm_hidden_size)
    if self.use_cuda:
      decoder_init_state = decoder_init_state.cuda()
    decoder_output, hidden = self.decoder_lstm(embedded_chars, (decoder_init_state, decoder_init_state))
    # + maybe add an intermediate linear transformation?
    decoder_decisions = self.decoder_decision(decoder_output)
    return decoder_decisions
