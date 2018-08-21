import pickle, sys, datetime
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Embedding
from keras.optimizers import RMSprop

THREADS_NUM = int(sys.argv[1])
EXPERIM_ID = sys.argv[2]
EXPERIM_FILE = sys.argv[3]
EPOCHS_COUNT = int(sys.argv[4])
CHARS_PATH = sys.argv[5]

# Load the train and test corpora.
train_err_objs = None
with open('train_set_{}.pkl'.format(EXPERIM_ID), 'rb') as pkl:
    train_err_objs = pickle.load(pkl)
test_err_objs = None
with open('test_set_{}.pkl'.format(EXPERIM_ID), 'rb') as pkl:
    test_err_objs = pickle.load(pkl)

# Load the list of chars.
chars = [ '<BLANK>', '<AOV>' ]
with open(CHARS_PATH) as char_fl:
    chars += char_fl.read().strip().split('\n')
char_to_idx = { char : num for (num, char) in enumerate(chars) }
idx_to_char = { num : char for (num, char) in enumerate(chars) }

# Vectorize the test&train corpora.
train_x = [] # these are lists of arrays; each array in a list corresponds to step-by-step feeding chars to the net
train_y = []
test_x = []
test_y = []
def vectorize(err_objs, x, y):
    for (sample_n, err_obj) in enumerate(err_objs):
        x_indices = [ char_to_idx[char] if char in char_to_idx
                                        else char_to_idx['<AOV>']
                      for char in err_obj['error']]
        y_indices = [ char_to_idx[char] if char in char_to_idx
                                        else char_to_idx['<AOV>']
                      for char in err_obj['correction'] ]
        # 1 here and further below forces the delay between receiving an
        # emission (in x) and predicting its true form in y.
        sample_len = max(len(x_indices), len(y_indices)+1)
        x_indices = x_indices + [ char_to_idx['<BLANK>'] ] * (sample_len-len(x_indices)) # align to the number of columns
        y_indices = [ char_to_idx['<BLANK>'] ] + y_indices

        sample_x_arr = np.zeros((sample_len, sample_len)) # indices of characters for each step of feeding
                                                          # the sample into the net
        sample_y_arr = np.zeros((sample_len, len(chars))) # one hot vector
        for char_n in range(sample_len):
            sample_x_arr[char_n, :char_n] = x_indices[:char_n]

            if char_n < len(y_indices):
                sample_y_arr[char_n, y_indices[char_n]] = 1.0
            else:
                sample_y_arr[char_n, char_to_idx['<BLANK>']] = 1.0

        x.append(sample_x_arr)
        y.append(sample_y_arr)
vectorize(train_err_objs, train_x, train_y)
vectorize(test_err_objs, test_x, test_y)
train_samples_count = len(train_x)
test_samples_count = len(test_x)

# Define the model.
try:
    model = load_model('neural_model_{}.hdf5'.format(EXPERIM_ID))
    print('Loaded the previously saved model.')
except OSError: # saved model file not found
    model = Sequential()
    model.add(Embedding(len(chars), 80))
    model.add(LSTM(128))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    opt = RMSprop()

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    for epoch_n in range(EPOCHS_COUNT):
        print('Epoch {}/{} of training'.format(epoch_n+1, EPOCHS_COUNT))
        history = None
        for sample_n in range(len((train_x))):
            history = model.fit(train_x[sample_n], train_y[sample_n], verbose=0)
            print('{}/{}'.format(sample_n, train_samples_count), end='\r') # overwrite the number
            sys.stdout.flush()
        print('\nMetrics: {}'.format(history.history))

    # Save the model.
    model.save('neural_model_{}.hdf5'.format(EXPERIM_ID))

# Test the model.
good = 0
print('Evaluating neural prediction.')
with open('Neural_corrections_{}.tab'.format(EXPERIM_ID), 'w+') as corrs_file:
    for sample_n in range(len(test_err_objs)):
        print('{}/{}'.format(sample_n, test_samples_count), end='\r') # overwrite the number
        sys.stdout.flush()

        sample_x_arr = test_x[sample_n]
        prediction = np.argmax(model.predict(sample_x_arr), axis=1)
        predicted_chars = [idx_to_char[prediction[char_n]] for char_n in range(prediction.shape[0])]
        correction = ''.join([char for char in predicted_chars
                              if char != '<BLANK>'])

        if test_err_objs[sample_n]['correction'] == correction:
            good += 1
        print('{}\t{}'.format(test_err_objs[sample_n]['error'], correction), file=corrs_file)
print() # line feed

# Write the results.
with open(EXPERIM_FILE, 'a') as res_file:
    timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    print('Neural net ({})'.format(timestamp), file=res_file)
    print('Accuracy: {}'.format(good/len(test_err_objs)), file=res_file)
