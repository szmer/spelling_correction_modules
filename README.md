# Dependencies

## Packages

* PyTorch (and its dependencies like Scipy, Numpy etc.)
* PyLucene (needs Java)
* https://github.com/HIT-SCIR/ELMoForManyLangs needs to be cloned into this catalog

## Resources

* a set of semantic vectors -- can be obtained from http://dsmodels.nlp.ipipan.waw.pl/
* SGJP (Słownik Gramatyczny Języka Polskiego) dictionary in text version -- download from http://sgjp.pl/morfeusz/download/ a file `sgjp-xxx.tab.gz`, where `xxx` is a version date (I used `20180708`)
* PlEWi corpus -- can be obtained from http://romang.home.amu.edu.pl/plewic/

In `scripts/` you can find a bash script meant for making PlEWi usable by this code. Execute `pythonize_plewi.sh` in the PlEWi's catalog (just copy it there before running). This replaces Ruby annotations in YAML with Python's.

# Running

In `PROCEDURE.sh` you can point the script to your resources and tweak some variables, like whether you want to `USE_CUDA` in training neural nets.

Then run `bash PROCEDURE.sh`. All accuracies will be written to `results_x`, where `x` is your specified `EXPERIM_ID`. Note the entire test may run for around 20 hours depending on your specific specs.
