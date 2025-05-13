This repository contains our implementation of bounded fitting for the description logic ALC as well as instruction how to reproduce the data reported in the paper. 

## Requirements
- SParCEL jar-file and DL-Learner 1.5
- Packages from `requirements.txt`
- Installation of Python 3.10 (as EvoLearner requires this version)

## Setup
- As CELOE and SParCEL are not written in python, they are called using the python subprocess library.  For this to work, in the file `alc_benchmark.py` the variable `CELOE_PATH` has to be set to a path to the file `bin/cli` provided with DL-Learner 1.5 and the variable `SPARCEL_PATH` has to be set to the SParCEL jar file. The required jar-file that restricts SParCEL to ALC concepts rather than ALCQD concepts can be found under `learningsystems/spacel/SParCEL/parcel-cli-alc.jar` in the second repository that we submitted as supplemental materials to reproduce the results on the SML-Benchmarks. 

## Data sets
The data sets generated from the YAGO fragments that were used to obtain the graphs regarding exact learning are contained in the folder `alc_benchmarks`. There is a subfolder `family` with data sets generated from the family YAGO fragment and a subfolder `language` with data sets generated from the language fragments. 

## Reproduce Data
To reproduce the data reported in the paper for exact learning on the family benchmark, run 
`python alc_benchmark.py <path_to_data set folder>`
 CELOE, SparCEL, EvoLearner and ALC-SAT will then be run on this data and a file `result_reproduced.csv` will be created within the folder. The file contains values for time and accuracy for each tool and data set contained in the folder.  To reproduce our results on the language fragment, only ALCSAT+ has to be run. This can be achieved with 
 `python alc_benchmark_language.py <path_to_data set folder>`. Switching between the optimized and the unoptimized version can be done by changing the flags 
 `TYPE_ENCODING`and `TREE_TEMPLATES` in the file `fitting_alc.py`.