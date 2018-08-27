import lucene
import pickle, sys, datetime
import numpy as np
from scipy.spatial import distance

THREADS_NUM = int(sys.argv[1]) # NOTE not used
EXPERIM_ID = sys.argv[2]
EXPERIM_FILE = sys.argv[3]
DICTIONARY_PATH = sys.argv[4]
VECTORS_PATH = sys.argv[5]
NONVEC_SURROGATE_DISTANCE = float(sys.argv[6])

# Load the reference dictionary.
dict_str = ''
with open(DICTIONARY_PATH) as inp:
    dict_str = inp.read().strip()

# Load word vectors.
word_to_idx = {} # ie. indices of vectors
idx_to_word = {}
first_line = True
word_n, vecs_dim = 0, 0
with open(VECTORS_PATH) as vecs_file:
    for line in vecs_file:
        if first_line:
            vecs_dim = int(line.split(' ')[1])
            first_line = False
            continue
        # Read word forms.
        word = line.split(' ')[0].lower()
        word_to_idx[word] = word_n
        idx_to_word[word_n] = word
        word_n += 1
word_vecs = np.loadtxt(VECTORS_PATH, encoding="utf-8",
                       dtype=np.float32, comments=None,
                       skiprows=1, usecols=tuple(range(1, vecs_dim+1)))

# Load the test corpus.
test_err_objs = None
with open('test_set_{}.pkl'.format(EXPERIM_ID), 'rb') as pkl:
    test_err_objs = pickle.load(pkl)
test_samples_count = len(test_err_objs)

# Java imports:
from org.apache.lucene.search.spell import PlainTextDictionary, SpellChecker
# boilerplate for setting up spellchecking:
from java.io import StringReader
from org.apache.lucene.store import RAMDirectory
from org.apache.lucene.index import IndexWriterConfig
from org.apache.lucene.analysis.core import KeywordAnalyzer
from org.apache.lucene.search.spell import LevensteinDistance # corrected to Levenshtein in later versions

# Start JVM for Lucene.
lucene.initVM()

# Set up Lucene spellchecking.
dict_reader = StringReader(dict_str)
dictionary = PlainTextDictionary(dict_reader)

ramdir = RAMDirectory()
spellchecker = SpellChecker(ramdir)
spellchecker.indexDictionary(dictionary, IndexWriterConfig(KeywordAnalyzer()), True)

# Set up Lucene distance computation.
distance_check = LevensteinDistance()

# Run the word correction test.
def word_vec(word):
    return word_vecs[word_to_idx[word]]

def correct_word(word):
    candidates = spellchecker.suggestSimilar(word, 10)
    # Add edit distance information (we get this as a Lucene similarity measure in [0, 1]).
    candidates = [(cand, 1.0-distance_check.getDistance(word, cand)) for cand in candidates]
    # Add vector distance information.
    candidates = [(cand, ed_dist, distance.cosine(word_vec(cand), word_vec(word))
                                  if (cand in word_to_idx and word in word_to_idx)
                                  else NONVEC_SURROGATE_DISTANCE)
                  for (cand, ed_dist) in candidates]
    candidates.sort(key=lambda x: x[1]+x[2])
    if len(candidates) > 0:
        return candidates[0][0]
    else:
        return ''

good, bad = [], [] # append True's here to avoid threading issuses
counter = 0
with open('Vector_distance_corrections_{}.tab'.format(EXPERIM_ID), 'w+') as corrs_file:
    for (sample_n, err_obj) in enumerate(test_err_objs):
        print('{}/{}'.format(sample_n, test_samples_count), end='\r') # overwrite the number
        sys.stdout.flush()

        error = err_obj['error']
        true_correction = err_obj['correction']
        correction = correct_word(error)
        print('{}\t{}'.format(error, correction), file=corrs_file)
        if correction == true_correction:
            good.append(True)
        else:
            bad.append(True)
print() # line feed

with open(EXPERIM_FILE, 'a') as res_file:
    timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    print('Vector distance ({})'.format(timestamp), file=res_file)
    print('Accuracy: {}'.format(len(good)/len(test_err_objs)), file=res_file)
