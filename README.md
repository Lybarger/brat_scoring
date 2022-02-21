# BRAT Scoring

## Introduction
This repository is a Python package for comparing and scoring two sets of [BRAT](https://brat.nlplab.org/) annotations. The current version focuses on event-based scoring.

## Evaluation criteria
The evaluation criteria are defined in [sdoh_scoring.pdf](docs/sdoh_scoring.pdf).

The primary criteria for the SDOH challenge will be:
- trigger: "overlap"
- span-only arguments: "exact"
- labeld arguments: "label"    

The other criteria are included in the scoring routine to assist with troubleshooting.

## Requirements
The scoring routine is implemented in Python 3. NO testing was performed using Python 2.

The following packages are needed:
- spacy with languange model "en_core_web_sm"
- tqdm
- pandas
- wheel
 
 The package install will fail, if `wheel` is not already installed. 

## Evaluation script
The scoring routine can be pip installed or called from command line. The scoring routine implements the aforementioned evaluation by comparing two directories with BRAT-style annotations (*.txt and *.ann files). The scoring routine identifies all the *.ann files in both directories, finds matching filenames in the directories, and then compares the annotations defined in the *.ann files.

### Pythnon function import
 The `brat_scoring` package can be installed using `pip` as follows:

```
pip3 install git+https://github.com/Lybarger/brat_scoring.git
```
NOTE: the wheel package is needed to pip install the package.

The required arguments that define the input and output paths include:
- gold_dir: *str*, path to the input directory with gold annotations in BRAT format, e.g. "/home/gold/"
- predict_dir: *str*, path to the input directory with predicted annotations in BRAT format, e.g. "/home/predict/"
- output: *str*, path for the output CSV file that will contain the evaluation results, e.g. "/home/scoring.csv"

The optional arguments define the evaluation criteria:
- labeled_args: *list*, list of labeled argument names as str, default is ['StatusTime', 'StatusEmploy', 'TypeLiving']
- score_trig: *str*, trigger scoring criterion, options include {"exact", "overlap", "min\_dist"}, default is "overlap".
- score_span: *str*, span-only argument scoring criterion, options include {"exact", "overlap", "partial"}, default is "exact"
- score_labeled: *str*, labeled argument (span-with-value argument) scoring criterion, options include {"exact", "overlap", "label"}, default is "label"    
- include_detailed: *bool*, if True, the scoring routine will generate document-level scores, in addition to the corpus-level scores
- loglevel: *str*, logging level can be set, default value is "info"

Below is an example usage:

```python
from brat_scoring.scoring import score_brat_sdoh
from brat_scoring.constants import EXACT, LABEL, OVERLAP, PARTIAL, MIN_DIST

df = score_brat_sdoh( \
                gold_dir = "/home/gold/",
                predict_dir = "/home/predict/", 
                output_path = "/home/scoring.csv",
                score_trig = OVERLAP,
                score_span = EXACT,
                score_labeled = LABEL,
                )
```


### Command line
The command line script, `run_sdoh_scoring.py`, is a simple wrapper for the function, `score_brat_sdoh`.

The arguments for the command line script,`run_sdoh_scoring.py`, are similar to that of the function `score_brat_sdoh` above. The arguments can be view using:
```
python3 run_sdoh_scoring.py -h
```

Below is an example usage:
```
python3 run_sdoh_scoring.py /home/gold/ /home/predict/ /home/scoring.csv
--score_trig min_dist --score_span exact --score_labeled label
```
