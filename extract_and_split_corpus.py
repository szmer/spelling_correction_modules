import yaml, pickle, random, re, os, sys
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

THREADS_NUM = int(sys.argv[1])
EXPERIM_ID = sys.argv[2]
PLEWI_PATH = sys.argv[3]

dev_part = 0.05
test_part = 0.25
# (the rest will be automatically assigned to the training set)

def yaml_constructor(loader, node): # it's for comments in YAML, just so they load and can be discarded
    return str(node)
yaml.add_constructor('tag:yaml.org,2002:yaml', yaml_constructor)
yaml.add_constructor('tag:yaml.org,2002:value', yaml_constructor)

# Collect error objects from YAML files.
error_objs = []

def load_errors_from_yaml(filename):
    with open(PLEWI_PATH + filename) as yamlfile:
        docs = yaml.load_all(yamlfile)
        try:
            for doc in docs:
                for fragm in doc:
                    if not 'errors' in fragm:
                        continue
                    for err in fragm['errors']:
                        if not ':type' in err['attributes']:
                            continue
                        elif err['attributes'][':type'] == ':nonword':
                            error_objs.append(err)
                        elif not err['attributes'][':type'] in {':unknown', ':realword', ':nonword',
                                                                ':multiword', ':deletion', ':insertion'}:
                            raise RuntimeError('unknown text error type {} in file {}'.format(
                                err['attributes'][':type'], filename))
        except yaml.constructor.ConstructorError as e:
            print(e)
            print('a yaml error occured, did you run pythonize_plewi.py?')
            sys.exit(-1)

# Collect corpus TAML filenames.
good_filenames = []
for _, __, filenames in os.walk(PLEWI_PATH):
    for filename in filenames:
        if not re.match('.*\.yaml', filename):
            continue
        good_filenames.append(filename)

# Collect errors from files.
pool = ThreadPool(THREADS_NUM)
pool.map(load_errors_from_yaml, good_filenames)
pool.close()
pool.join() # wait for finish

print('{} errors extracted.'.format(len(error_objs)))
fail()

# Split the corpus and dump pickle files.
shuffled_indices = list(range(len(error_objs)))
random.shuffle(shuffled_indices)
dev_len = int(len(error_objs) * dev_part)
test_len = int(len(error_objs) * test_part)

with open('dev_set_{}.pkl'.format(EXPERIM_ID), 'wb+') as pkl:
    err_subset = []
    for ind in shuffled_indices[:dev_len]:
        err_subset.append(error_objs[ind])
    print('{} errors in the dev set.'.format(len(err_subset)))
    pickle.dump(err_subset, pkl)
with open('test_set_{}.pkl'.format(EXPERIM_ID), 'wb+') as pkl:
    err_subset = []
    for ind in shuffled_indices[dev_len:dev_len+test_len]:
        err_subset.append(error_objs[ind])
    print('{} errors in the test set.'.format(len(err_subset)))
    pickle.dump(err_subset, pkl)
with open('train_set_{}.pkl'.format(EXPERIM_ID), 'wb+') as pkl:
    err_subset = []
    for ind in shuffled_indices[dev_len+test_len:]:
        err_subset.append(error_objs[ind])
    print('{} errors in the train set.'.format(len(err_subset)))
    pickle.dump(err_subset, pkl)
