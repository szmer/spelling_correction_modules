import os, pickle, sys, datetime, json, csv
from random import shuffle
import numpy as np
import torch

THREADS_NUM = int(sys.argv[1])
EXPERIM_ID = sys.argv[2]
EXPERIM_FILE = sys.argv[3]
EPOCHS_COUNT = int(sys.argv[4])
BATCH_SIZE = int(sys.argv[5])
USE_CUDA = bool(sys.argv[6])

DIRECTIONS = sys.argv[7]
is_bidirectional = True if DIRECTIONS == 'bidirectional' else False

import neural_model

# Setup embeddings.
with open('./pl.model/char.dic') as char_lexicon_fl:
    # (can't use csv package because the driver trips up on missing fields)
    char_lexicon = dict([tuple(row.split('\t')) if len(row.split('\t')) == 2 else ('\u3000', row.split('\t')[0])
                         for row in char_lexicon_fl.read().strip().split('\n')])
    for token in char_lexicon:
        char_lexicon[token] = int(char_lexicon[token])
idx_to_char = { num : char for (num, char) in enumerate(char_lexicon) }
char_embedding = torch.nn.Embedding(len(char_lexicon), 50, padding_idx=char_lexicon['<pad>'])
max_chars = 17

# Create the model.
model = neural_model.Model(char_embedding, USE_CUDA, is_bidirectional)
if USE_CUDA:
    torch.cuda.set_device(0)
    model.cuda()

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
    return chars_ids(err_obj['error']), chars_ids(err_obj['correction'])

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Try to load a saved model, or train the model.
if os.path.isfile('neural_{}_model_{}.torch'.format(DIRECTIONS, EXPERIM_ID)):
    model.load_state_dict(torch.load('neural_{}_model_{}.torch'.format(DIRECTIONS, EXPERIM_ID)))
else: # saved model file not found
    corp_indices = list(range(train_samples_count))
    loss_history = []
    for epoch_n in range(EPOCHS_COUNT):
        print('Epoch {}/{} of training'.format(epoch_n+1, EPOCHS_COUNT))
        shuffle(corp_indices)
        counter = 0
        while counter < len(corp_indices):
            batch_xs, batch_ys = [], []
            for i in range(BATCH_SIZE):
                sample_n = corp_indices[counter]+i
                if sample_n >= len(corp_indices):
                    break
                x, y = preprocess(train_err_objs[sample_n])
                batch_xs.append(x)
                batch_ys.append(y)

            predicted_distribution = model.forward(torch.LongTensor(batch_xs))
            predicted_distribution = predicted_distribution.view(max_chars*len(batch_xs),
                                                char_embedding.num_embeddings)
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
    torch.save(model.state_dict(), 'neural_{}_model_{}.torch'.format(DIRECTIONS, EXPERIM_ID))

    # Write the loss history to a text file.
    with open('Neural_{}_history_{}'.format(DIRECTIONS,
                                            datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')), 'w+') as his_fl:
        for item in loss_history:
            print(item.item(), file=his_fl, end=' ')

# Test the model.
good = 0
perplexity = 0.0
test_loss = 0.0
batches_count = 0
print('Evaluating neural prediction of {} LSTM.'.format(DIRECTIONS))
with open('Neural_{}_corrections_{}.tab'.format(EXPERIM_ID, DIRECTIONS), 'w+') as corrs_file:
    with torch.no_grad(): # avoid out of memory errors
        counter = 0
        while counter < test_samples_count:
            batch_xs, batch_ys = [], []
            for i in range(BATCH_SIZE):
                sample_n = counter + i
                if sample_n >= test_samples_count:
                    break
                x, y = preprocess(test_err_objs[sample_n])
                batch_xs.append(x)
                batch_ys.append(y)

            predicted_distribution = model.forward(torch.LongTensor(batch_xs))
            predicted_distribution = predicted_distribution.view(max_chars*len(batch_xs),
                                                char_embedding.num_embeddings)

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

                # Computing accuracy.
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
    print('{} neural ({})'.format(DIRECTIONS, timestamp), file=res_file)
    print('Accuracy: {}'.format(good/len(test_err_objs)), file=res_file)
    print('Perplexity: {}'.format(2**-(perplexity/batches_count)), file=res_file)
    print('Test loss: {}'.format(test_loss/batches_count), file=res_file)
