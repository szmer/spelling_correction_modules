import lucene
import pickle, sys, datetime

THREADS_NUM = int(sys.argv[1]) # NOTE not used
EXPERIM_ID = sys.argv[2]
EXPERIM_FILE = sys.argv[3]
DICTIONARY_PATH = sys.argv[4]

# Define diacritical equivalences.
equivalences = [ ('a', 'ą'), ('c', 'ć'), ('e', 'ę'), ('l', 'ł'), ('n', 'ń'),
                 ('o', 'ó'), ('s', 'ś'), ('z', 'ż'), ('z', 'ź'), ('ź', 'ż')]
# (add reversed variants:)
equivalences += [(eq2, eq1) for (eq1, eq2) in equivalences]

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

def correct_word(word):
    if len(word) > 17:
        return ''
    equivalence_points = []
    candidates = set( [ word ] )
    for (char_n, char) in enumerate(word):
        equivalence_points += [(char_n, eq1, eq2) for (eq1, eq2) in equivalences
                              if eq1 == char]
    for (char_n, eq1, eq2) in equivalence_points:
        new_candidates = set()
        for candidate in candidates:
            new_candidates.add(candidate[:char_n] + eq1 + candidate[char_n+1:])
            new_candidates.add(candidate[:char_n] + eq2 + candidate[char_n+1:])
        candidates = candidates.union(new_candidates)
    # Prune non-words and add edit distance information (we get this as a
    # Lucene similarity measure in [0, 1]).
    candidates = [(cand, 1.0-distance_check.getDistance(word, cand)) for cand in candidates
                  if spellchecker.exist(cand)]
    candidates.sort(key=lambda x: x[1])
    if len(candidates) > 0:
        return candidates[0][0]
    else:
        return ''

good, bad = [], []
counter = 0
with open('Diacritical_swapping_corrections_{}.tab'.format(EXPERIM_ID), 'w+') as corrs_file:
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
    print('Diacritical swapping ({})'.format(timestamp), file=res_file)
    print('Accuracy: {}'.format(len(good)/len(test_err_objs)), file=res_file)
