set -o errexit

export EXPERIM_ID='0'
export THREADS_NUM=8 # this is now only used for loading PlEWi, which takes quite some time by itself
export EXPERIM_FILE="results_$EXPERIM_ID" # all results will be written to this file
export PLEWI_PATH='../plewic-yaml-1.0/' # the main error corpus
export DICTIONARY_PATH='../../sgjp/data' # reference dictionary
export VECTORS_PATH='../../nkjp/wektory/nkjp+wiki-forms-all-100-skipg-ns.txt/data'
export NONVEC_SURROGATE_DISTANCE=1.0 # arbitrary cosine distance measure when some vectors are missing
# for ELMo:
export EPOCHS_COUNT=35
export MODEL_PATH='pl.model'
export BATCH_SIZE=512
export USE_CUDA=1

date
echo 'Splitting the corpus... '
#python3 extract_and_split_corpus.py $THREADS_NUM $EXPERIM_ID $PLEWI_PATH

date
echo 'Testing edit distance...'
python3 -i test_edit_distance.py $THREADS_NUM $EXPERIM_ID $EXPERIM_FILE $DICTIONARY_PATH

date
echo 'Testing diacritical swapping...'
#python3 test_diacritical_swapping.py $THREADS_NUM $EXPERIM_ID $EXPERIM_FILE $DICTIONARY_PATH

date
echo 'Testing vector distance...'
#python3 test_vector_distance.py $THREADS_NUM $EXPERIM_ID $EXPERIM_FILE $DICTIONARY_PATH $VECTORS_PATH $NONVEC_SURROGATE_DISTANCE

date
echo 'Testing a unidirectional neural net...'
#python3 test_neural.py $THREADS_NUM $EXPERIM_ID $EXPERIM_FILE $EPOCHS_COUNT $BATCH_SIZE $USE_CUDA 'unidirectional'

date
echo 'Testing a bidirectional neural net...'
#python3 test_neural.py $THREADS_NUM $EXPERIM_ID $EXPERIM_FILE $EPOCHS_COUNT $BATCH_SIZE $USE_CUDA 'bidirectional'

date
echo 'Testing an ELMo net...'
#python3 test_elmo.py $THREADS_NUM $EXPERIM_ID $EXPERIM_FILE $EPOCHS_COUNT $MODEL_PATH $BATCH_SIZE $USE_CUDA

date
