import pickle, sys, datetime, json, csv
from random import shuffle
import numpy as np
import torch

THREADS_NUM = int(sys.argv[1])
EXPERIM_ID = sys.argv[2]
EXPERIM_FILE = sys.argv[3]
EPOCHS_COUNT = int(sys.argv[4])
MODEL_PATH = sys.argv[5]
BATCH_SIZE = int(sys.argv[6])
USE_CUDA = bool(sys.argv[7])

# Load and setup the ready Elmo solution.
import ELMoForManyLangs.src as elmolangs
import ELMoForManyLangs.src.modules.embedding_layer
import elmo_model

with open('./ELMoForManyLangs/configs/cnn_50_100_512_4096_sample.json') as elmo_config_fl:
    elmo_config = json.load(elmo_config_fl)
elmo_config['token_embedder']['max_characters_per_token'] = 17
max_chars = elmo_config['token_embedder']['max_characters_per_token']
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
idx_to_char = { num : char for (num, char) in enumerate(char_lexicon) }
char_embedding = elmolangs.modules.embedding_layer.EmbeddingLayer(elmo_config['token_embedder']['char_dim'], char_lexicon, fix_emb=False, embs=None)
word_embedding = elmolangs.modules.embedding_layer.EmbeddingLayer(elmo_config['token_embedder']['word_dim'], word_lexicon, fix_emb=False, embs=None)
# Create the model.
model = elmo_model.Model(elmo_config, word_embedding, char_embedding, USE_CUDA)
if USE_CUDA:
    torch.cuda.set_device(0)
    model.cuda()
model.load_model(MODEL_PATH)

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
    chars = chars[:max_chars-2]
    chars = ([ '<eow>' ] # yes, those are swapped in the original code
             + list(chars)
             + [ '<bow>' ]
             + ((max_chars-len(chars)-2) * [ char_lexicon['<pad>'] ]))
    char_ids = [ char_lexicon[char] if char in char_lexicon else char_lexicon['<oov>']
                 for char in chars ]
    return char_ids

# NOTE not used
def chars_onehots(chars):
    ids = chars_ids(chars)
    return [np.array([(i == char_id) for i in range(len(char_lexicon))], dtype=np.float32)
              for char_id in ids]

def preprocess(err_obj):
    x_token_ids = [word_lexicon['<bos>'], # mock sentence markers, include one sentence
                    word_lexicon['<oov>'], # these are non-word errors by definition
                    word_lexicon['<eos>']]
                #######err_obj['error'][:max_chars+2],
    ####x_text = [ err_obj['error'] ]
    x_chars_ids = [ chars_ids(['<bos>']), chars_ids(err_obj['error']), chars_ids(['<eos>']) ]

    y_char_ids = chars_ids(err_obj['correction'])

    x = ((x_token_ids, x_chars_ids))
    y = (y_char_ids)
    mask = [ 1 ] * (len(x_token_ids))
           ## this messes up the encodder:
           #### + [ 0 ] * (max_chars-len(x_token_ids)))
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
    print('Epoch {}/{} of training'.format(epoch_n+1, EPOCHS_COUNT))
    shuffle(corp_indices)
    counter = 0
    while counter < len(corp_indices):
        batch_xs, batch_ys, batch_masks = [], [], []
        for i in range(BATCH_SIZE):
            sample_n = corp_indices[counter]+i
            if sample_n >= len(corp_indices):
                break
            x, y, mask = preprocess(dev_err_objs[sample_n])
            batch_xs.append(x)
            batch_ys.append(y)
            batch_masks.append(mask)

        predicted_chars = model.forward(torch.tensor([x[0] for x in batch_xs]),
                                        torch.LongTensor([x[1] for x in batch_xs]),
                                        torch.LongTensor([mask for mask in batch_masks]))
        predicted_chars = predicted_chars.view(max_chars*len(batch_xs),
                                               char_embedding.embedding.num_embeddings)
        optimizer.zero_grad()
        #for char_n in range(max_chars):
        y = torch.LongTensor(sum(batch_ys, []))
        if USE_CUDA:
            y = y.cuda()
        loss_val = loss(predicted_chars, y)
        loss_val.backward()
        optimizer.step()

        print('{}/{}'.format(counter, dev_samples_count), end='\r') # overwrite the number
        sys.stdout.flush()
        counter += BATCH_SIZE
    print('\nLoss metric: {}'.format(loss_val))
    # TODO: zbierać historię funkcji, testować po drodze?

# Test the model.
good = 0
print('Evaluating neural prediction with ELMo.')
with open('Elmo_corrections_{}.tab'.format(EXPERIM_ID), 'w+') as corrs_file:
    counter = 0
    while counter < len(corp_indices):
        print('{}/{}'.format(sample_n, dev_samples_count), end='\r') # overwrite the number
        sys.stdout.flush()

        batch_xs, batch_ys, batch_masks = [], [], []
        for i in range(BATCH_SIZE):
            sample_n = counter + i
            if sample_n >= len(corp_indices):
                break
            x, y, mask = preprocess(dev_err_objs[sample_n])
            batch_xs.append(x)
            batch_ys.append(y)
            batch_masks.append(mask)

        predicted_chars = model.forward(torch.tensor([x[0] for x in batch_xs]),
                                        torch.LongTensor([x[1] for x in batch_xs]),
                                        torch.LongTensor([mask for mask in batch_masks]))
        predicted_chars = predicted_chars.view(max_chars*len(batch_xs),
                                               char_embedding.embedding.num_embeddings)

        predictions = torch.argmax(predicted_chars, dim=1)
        for i in range(BATCH_SIZE):
            sample_n = counter + i
            if sample_n >= len(corp_indices):
                break
            predicted_chars = [idx_to_char[predictions[i*max_chars:(i+1)*max_chars][char_n].item()] # the .item() part converts tensor to number
                               for char_n in range(max_chars)]
            correction = ''.join([char for char in predicted_chars
                                if len(char) == 1]) # eliminate markers
            if dev_err_objs[sample_n]['correction'] == correction:
                good += 1
            print('{}\t{}'.format(dev_err_objs[sample_n]['error'], correction), file=corrs_file)
        counter += BATCH_SIZE
print() # line feed

# Write the results.
####with open(EXPERIM_FILE, 'a') as res_file:
timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
print('ELMo ({})'.format(timestamp))##, file=res_file)
print('Accuracy: {}'.format(good/len(dev_err_objs)))##, file=res_file)
