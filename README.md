# BRAT Scoring

## Evaluation criteria


## Evaluation script
The scoring routine can be called from command line or imported as a Python package. The scoring routine implements the aforementioned evaluation by comparing two directories with BRAT-style annotations (*.txt and *.ann files). The scoring routine identifies all the *.ann files in both directories, finds matching filenames in the directories, and then compares the annotations defined in the *.ann files.

### Command line
The scoring script, `score_sdoh.py`, can be called from command-line. 

The required arguments that define the input and output paths include:
- gold_dir: *str*, path to the input directory with gold annotations in BRAT format, e.g. "/home/gold/"
- predict_dir: *str*, path to the input directory with predicted annotations in BRAT format, e.g. "/home/predict/"
- output: *str*, path for the output CSV file that will contain the evaluation results, e.g. "/home/scoring.csv"

The optional arguments define the evaluation criteria:
- score_trig: *str*, trigger scoring criterion, options include {"exact", "overlap", "min\_dist"}, default is "overlap".
- score_span: *str*, span-only argument scoring criterion, options include {"exact", "overlap", "partial"}, default is "exact"
- score_labeled: *str*, labeled argument (span-with-value argument) scoring criterion, options include {"exact", "overlap", "label"}, default is "label"    
- include_detailed: if passed, the scoring routine will generate document-level scores, in addition to the corpus-level scores
- loglevel: *str*, logging level can be set, default value is "info"

Below is an example usage:
```
python3 score_brat.py /home/gold/ /home/predict/ /home/scoring.csv
--score_trig min_dist --score_span exact --score_labeled label
```

### Pythnon function import
The command line script, `score_sdoh.py`, is a simple wrapper for the function, `score_brat`. Below is an example.

```python
import sys
sys.path.insert(1, package_path

from brat_scoring.scoring import score_brat
from brat_scoring.constants import EXACT, LABEL, OVERLAP, PARTIAL, MIN_DIST

df = score_brat( \
                gold_dir = "/home/gold/", 
                predict_dir = "/home/predict/", \
                score_trig = OVERLAP,
                score_span = EXACT,
                score_labeled = LABEL,
                path = "/home/scoring.csv")
```



