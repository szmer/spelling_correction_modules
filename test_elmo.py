import os, pickle, sys, datetime, json, csv
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

#
# Load and setup the ready Elmo solution.
#
import ELMoForManyLangs.src as elmolangs
import ELMoForManyLangs.src.modules.embedding_layer
import elmo_model

# Load and setup embeddings.
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
train_err_objs = None
with open('train_set_{}.pkl'.format(EXPERIM_ID), 'rb') as pkl:
    train_err_objs = pickle.load(pkl)
test_err_objs = None
with open('test_set_{}.pkl'.format(EXPERIM_ID), 'rb') as pkl:
    test_err_objs = pickle.load(pkl)

train_samples_count = len(train_err_objs)
test_samples_count = len(test_err_objs)

# Preprocessing error samples.
def chars_ids(chars):
    "Get a list of char ids, trimmed to max length, with added markers and padding"
    chars = chars[:max_chars-2]
    chars = ([ '<eow>' ] # yes, those are swapped in the original code
             + list(chars)
             + [ '<bow>' ]
             + ((max_chars-len(chars)-2) * [ char_lexicon['<pad>'] ]))
    char_ids = [ char_lexicon[char] if char in char_lexicon else char_lexicon['<oov>']
                 for char in chars ]
    return char_ids

def preprocess(err_obj):
    x_token_ids = [word_lexicon['<bos>'], # mock sentence markers, include one sentence
                    word_lexicon['<oov>'], # these are non-word errors by definition
                    word_lexicon['<eos>']]
    ####x_text = [ err_obj['error'] ]
    x_chars_ids = [ chars_ids(['<bos>']), chars_ids(err_obj['error']), chars_ids(['<eos>']) ]

    y_char_ids = chars_ids(err_obj['correction'])

    x = ((x_token_ids, x_chars_ids))
    y = (y_char_ids)
    mask = [ 1 ] * (len(x_token_ids))
           ## this messes up the encodder:
           #### + [ 0 ] * (max_chars-len(x_token_ids)))
    return x, y, mask

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Try to load a saved model, or train the model.
if os.path.isfile('elmo_model_{}.torch'.format(EXPERIM_ID)):
    model.load_state_dict(torch.load('elmo_model_{}.torch'.format(EXPERIM_ID)))
else: # saved model file not found
    corp_indices = list(range(train_samples_count))
    loss_history = []
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
                x, y, mask = preprocess(train_err_objs[sample_n])
                batch_xs.append(x)
                batch_ys.append(y)
                batch_masks.append(mask)

            predicted_distribution = model.forward(torch.tensor([x[0] for x in batch_xs]),
                                             torch.LongTensor([x[1] for x in batch_xs]),
                                             torch.LongTensor(batch_masks))
            predicted_distribution = predicted_distribution.view(max_chars*len(batch_xs),
                                                                 char_embedding.embedding.num_embeddings)
            optimizer.zero_grad()
            y = torch.LongTensor(sum(batch_ys, []))
            if USE_CUDA:
                y = y.cuda()
                loss_val = loss(predicted_distribution, y)
                loss_val.backward()
                optimizer.step()

            print('{}/{}'.format(counter, train_samples_count), end='\r') # overwrite the number
            sys.stdout.flush()
            counter += BATCH_SIZE
        print('\nLoss metric: {}'.format(loss_val))
        loss_history.append(loss_val)

    # Save the model.
    torch.save(model.state_dict(), 'elmo_model_{}.torch'.format(EXPERIM_ID))

    # Write the loss history to a text file.
    with open('Elmo_history_{}'.format(datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')), 'w+') as his_fl:
        for item in loss_history:
            print(item.item(), file=his_fl, end=' ')

# Test the model.
good = 0
perplexity = 0.0
batches_count = 0
test_loss = 0.0
print('Evaluating neural prediction with ELMo.')
with open('Elmo_corrections_{}.tab'.format(EXPERIM_ID), 'w+') as corrs_file:
    with torch.no_grad(): # avoid out of memory errors
        counter = 0
        while counter < test_samples_count:
            batch_xs, batch_ys, batch_masks = [], [], []
            for i in range(BATCH_SIZE):
                sample_n = counter + i
                if sample_n >= test_samples_count:
                    break
                x, y, mask = preprocess(test_err_objs[sample_n])
                batch_xs.append(x)
                batch_ys.append(y)
                batch_masks.append(mask)

            predicted_distribution = model.forward(torch.tensor([x[0] for x in batch_xs]),
                                            torch.LongTensor([x[1] for x in batch_xs]),
                                            torch.LongTensor(batch_masks))
            predicted_distribution = predicted_distribution.view(max_chars*len(batch_xs),
                                                char_embedding.embedding.num_embeddings)

            predictions = torch.argmax(predicted_distribution, dim=1)

             # Computing perplexity.
            true_y = sum(batch_ys, [])
            local_perplexity = 0
            for i, j in enumerate(true_y):
                local_perplexity += predicted_distribution[i, j].item() # they're already logs
            perplexity += local_perplexity / predicted_distribution.size(0)
            batches_count += 1

            # Computing loss value.
            y = torch.LongTensor(sum(batch_ys, []))
            test_loss += loss(predicted_distribution.cpu(), y)

            for i in range(BATCH_SIZE):
                sample_n = counter + i
                if sample_n >= test_samples_count:
                    break
                predicted_chars = [idx_to_char[predictions[i*max_chars:(i+1)*max_chars][char_n].item()] # the .item() part converts tensor to number
                                for char_n in range(max_chars)]
                correction = ''.join([char for char in predicted_chars
                                    if len(char) == 1]) # eliminate markers
                if test_err_objs[sample_n]['correction'] == correction:
                    good += 1
                print('{}\t{}'.format(test_err_objs[sample_n]['error'], correction), file=corrs_file)

            print('{}/{}'.format(counter, test_samples_count), end='\r') # overwrite the number
            sys.stdout.flush()
            counter += BATCH_SIZE
print() # line feed

# Write the results.
with open(EXPERIM_FILE, 'a') as res_file:
    timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    print('ELMo ({})'.format(timestamp), file=res_file)
    print('Accuracy: {}'.format(good/len(test_err_objs)), file=res_file)
    print('Perplexity: {}'.format(2**-(perplexity/batches_count)), file=res_file)
    print('Test loss: {}'.format(test_loss/batches_count), file=res_file)
