
## Setup

## Requirements
- SParCEL and CELOE jar files
- Packages from `requirements.txt`

## Data sets
The data sets used to obtain the graphs regarding exact learning are contained in the folder `alc_benchmarks`. There is a subfolder `family` with data sets generated from the family YAGO fragments and a subfolder `language` with data sets generated from the language fragments. 

## Run tools
To run ALC-SAT on one of the provided data sets, run 
```python alc_benchmark.py <path_to_data set folder> ```
 CELOE, SparCEL, EvoLearner and ALC-SAT will then be run on this data and a file `result_reproduced.csv` will be created within the folder. The file contains values for time, accuracy and concepts for each data set contained in the folder. 