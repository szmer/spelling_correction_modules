import lucene
import pickle, sys, datetime

THREADS_NUM = int(sys.argv[1]) # NOTE not used
EXPERIM_ID = sys.argv[2]
EXPERIM_FILE = sys.argv[3]
DICTIONARY_PATH = sys.argv[4]

# Load the reference dictionary.
dict_str = ''
copyright_end = False
with open(DICTIONARY_PATH) as inp:
    for line in inp:
        line = line.strip()
        if '</COPYRIGHT>' in line:
            copyright_end = True
            continue
        if not copyright_end:
            continue
        dict_str += ' ' + line.split('\t')[0]

# Load the test corpus.
test_err_objs = None
with open('test_set_{}.pkl'.format(EXPERIM_ID), 'rb') as pkl:
    test_err_objs = pickle.load(pkl)

## Java imports:
from org.apache.lucene.search.spell import PlainTextDictionary, SpellChecker
# boilerplate for setting up spellchecking:
from java.io import StringReader
from org.apache.lucene.store import RAMDirectory
from org.apache.lucene.index import IndexWriterConfig
from org.apache.lucene.analysis.core import KeywordAnalyzer

# Start JVM for Lucene.
lucene.initVM()

# Set up Lucene spellchecking.
dict_reader = StringReader(dict_str)
dictionary = PlainTextDictionary(dict_reader)

ramdir = RAMDirectory()
spellchecker = SpellChecker(ramdir)
spellchecker.indexDictionary(dictionary, IndexWriterConfig(KeywordAnalyzer()), True)

# Run the word correction test.
def correct_word(word):
    candidates = spellchecker.suggestSimilar(word, 10)
    if len(candidates) > 0:
        return candidates[0]
    else:
        return ''

good, bad = [], []
with open('Edit_distance_corrections_{}.tab'.format(EXPERIM_ID), 'w+') as corrs_file:
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
    print('Edit distance ({})'.format(timestamp), file=res_file)
    print('Accuracy: {}'.format(len(good)/len(test_err_objs)), file=res_file)

#####nie, 19 sie 2018, 12:56:12 CEST
#####Testing edit distance...
#####13768700
#####nie, 19 sie 2018, 13:42:14 CEST
