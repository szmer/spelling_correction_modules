set -o errexit

# execute pythonize_plewi.sh in the corpus file if it wasn't done before!
# to get a SGJP vocab file, run sgjp_vocab.sh and remove the copyright information in the beginning (recommended)
# both of these have to be executed in directories where the resources live

export EXPERIM_ID='0'
export THREADS_NUM=8
export EXPERIM_FILE="results_$EXPERIM_ID" # all results will be written to this file
export PLEWI_PATH='../plewic-yaml-1.0/' # the main error corpus
export DICTIONARY_PATH='../../sgjp/vocab' # reference dictionary

#export MAX_EDIT_DISTANCE=4

date
echo 'Splitting the corpus... '
#python3 extract_and_split_corpus.py $THREADS_NUM $EXPERIM_ID $PLEWI_PATH

date
echo 'Testing edit distance...'
python3 test_edit_distance.py $THREADS_NUM $EXPERIM_ID $EXPERIM_FILE $DICTIONARY_PATH

date
