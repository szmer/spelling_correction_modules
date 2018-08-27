import pickle, sys, datetime, json, csv
from random import shuffle
import numpy as np
import torch

THREADS_NUM = int(sys.argv[1])
EXPERIM_ID = sys.argv[2]
EXPERIM_FILE = sys.argv[3]
EPOCHS_COUNT = int(sys.argv[4])
CHARS_PATH = sys.argv[5]

# Load and setup the ready Elmo solution.
import ELMoForManyLangs.src as elmolangs
import ELMoForManyLangs.src.modules.embedding_layer
import elmo_model

use_cuda = True
with open('./ELMoForManyLangs/configs/cnn_50_100_512_4096_sample.json') as elmo_config_fl:
    elmo_config = json.load(elmo_config_fl)
elmo_config['token_embedder']['max_characters_per_token'] = 17
# Load char and word indices.
with open('./pl.model/char.dic') as char_lexicon_fl:
    # (can't use csv package because the driver trips up on missing fields)
    char_lexicon = dict([tuple(row.split('\t')) if len(row.split('\t')) == 2 else ('\u3000', row.split('\t')[0])
                         for row in char_lexicon_fl.read().strip().split('\n')])
    for token in char_lexicon:
        char_lexicon[token] = int(char_lexicon[token])
with open('./pl.model/word.dic') as word_lexicon_fl:
    word_lexicon = dict([tuple(row.split('\t')) if len(row.split('\t')) == 2 else ('\u3000', row.split('\t')[0])
                         for row in word_lexicon_fl.read().strip().split('\n')])
    for token in word_lexicon:
        word_lexicon[token] = int(word_lexicon[token])
char_embedding = elmolangs.modules.embedding_layer.EmbeddingLayer(elmo_config['token_embedder']['char_dim'], char_lexicon, fix_emb=False, embs=None)
word_embedding = elmolangs.modules.embedding_layer.EmbeddingLayer(elmo_config['token_embedder']['word_dim'], word_lexicon, fix_emb=False, embs=None)
# Create the model.
model = elmo_model.Model(elmo_config, word_embedding, char_embedding, use_cuda)
if use_cuda:
    torch.cuda.set_device(0)
    model.cuda()

# Load the train and test corpora.
####train_err_objs = None
####with open('train_set_{}.pkl'.format(EXPERIM_ID), 'rb') as pkl:
####    train_err_objs = pickle.load(pkl)
####test_err_objs = None
####with open('test_set_{}.pkl'.format(EXPERIM_ID), 'rb') as pkl:
####    test_err_objs = pickle.load(pkl)
dev_err_objs = None
with open('dev_set_{}.pkl'.format(EXPERIM_ID), 'rb') as pkl:
    dev_err_objs = pickle.load(pkl)

# Vectorize the test&train corpora.
####train_x = []
####train_y = []
####test_x = []
####test_y = []
dev_x = []
dev_y = []
def chars_ids(chars):
    chars = chars[:elmo_config['token_embedder']['max_characters_per_token']-2]
    chars = ([ '<eow>' ] # yes, those are swapped in the original code
             + list(chars)
             + [ '<bow>' ]
             + ((elmo_config['token_embedder']['max_characters_per_token']-len(chars)-2) * [ char_lexicon['<pad>'] ]))
    char_ids = [ char_lexicon[char] if char in char_lexicon else char_lexicon['<oov>']
                 for char in chars ]
    return char_ids

# NOTE not used
def chars_onehots(chars):
    ids = chars_ids(chars)
    return [np.array([(i == char_id) for i in range(len(char_lexicon))], dtype=np.float32)
              for char_id in ids]

def preprocess(err_obj):
    x_token_ids = torch.tensor([[word_lexicon['<bos>'], # mock sentence markers, include one sentence
                                 word_lexicon['<oov>'], # these are non-word errors by definition
                                 word_lexicon['<eos>']]])
                #######err_obj['error'][:elmo_config['token_embedder']['max_characters_per_token']+2],
    x_text = [ err_obj['error'] ]
    x_chars_ids = torch.LongTensor( [[ chars_ids(['<bos>']), chars_ids(err_obj['error']), chars_ids(['<eos>']) ]] )

    y_chars_onehots = torch.LongTensor(chars_ids(err_obj['correction']))
    if use_cuda:
        y_chars_onehots = y_chars_onehots.cuda()
    x = ((x_token_ids, x_text, x_chars_ids))
    y = (y_chars_onehots)
    mask = torch.LongTensor([ [ 1 ] * (len(x_token_ids)+2) ])
    return x, y, mask

####preprocess(dev_err_objs, dev_x, dev_y)
dev_samples_count = len(dev_err_objs)
####preprocess(train_err_objs, train_x, train_y)
####preprocess(test_err_objs, test_x, test_y)
####train_samples_count = len(train_x)
####test_samples_count = len(test_x)

# Train the model.
corp_indices = list(range(dev_samples_count))
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
for epoch_n in range(EPOCHS_COUNT):
    print('Epoch {}/{} of testing'.format(epoch_n+1, EPOCHS_COUNT))
    history = None
    shuffle(corp_indices)
    counter = 0
    for sample_n in corp_indices:
        x, y, mask = preprocess(dev_err_objs[sample_n])
        predicted_chars = model.forward(x[0], x[2], mask)
        predicted_chars = predicted_chars.view(elmo_config['token_embedder']['max_characters_per_token'], char_embedding.embedding.num_embeddings)
        optimizer.zero_grad()
        #for char_n in range(elmo_config['token_embedder']['max_characters_per_token']):
        loss_val = loss(predicted_chars, y)
        loss_val.backward()
        optimizer.step()

        print('{}/{}'.format(counter, dev_samples_count), end='\r') # overwrite the number
        sys.stdout.flush()
        counter += 1
    print('\nLoss metric: {}'.format(loss.item()))

# Test the model.
good = 0
print('Evaluating neural prediction with ELMo.')
with open('Elmo_corrections_{}.tab'.format(EXPERIM_ID), 'w+') as corrs_file:
    for sample_n in range(len(dev_err_objs)):
        print('{}/{}'.format(sample_n, dev_samples_count), end='\r') # overwrite the number
        sys.stdout.flush()

        predicted_chars = model.forward(dev_x[0], dev_x[2], 1, 1)
        predicted_chars = predicted_chars.view(3, 1, elmo_config['token_embedder']['max_characters_per_token'], char_embedding.embedding.num_embeddings)
        prediction = torch.argmax(predicted_chars, axis=1)
        predicted_chars = [idx_to_char[prediction[char_n]] for char_n in range(prediction.shape[0])]
        correction = ''.join([char for char in predicted_chars
                              if len(char) == 1]) # eliminate markers

        if dev_err_objs[sample_n]['correction'] == correction:
            good += 1
        print('{}\t{}'.format(dev_err_objs[sample_n]['error'], correction), file=corrs_file)
print() # line feed

# Write the results.
####with open(EXPERIM_FILE, 'a') as res_file:
timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
print('ELMo ({})'.format(timestamp))##, file=res_file)
print('Accuracy: {}'.format(good/len(test_err_objs)))##, file=res_file)
