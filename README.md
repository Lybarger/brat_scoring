# BRAT Scoring



## Evaluation script
The scoring routine can be called from command line or imported as a Python package. The scoring routine implements the aforementioned evaluation by comparing two directories with BRAT-style annotations (*.txt and *.ann files). The scoring routine identifies all the *.ann files in both directories, finds matching filenames in the directories, and then compares the annotations defined in the *.ann files.

### Command line
The scoring script `score_sdoh.py`, can be called from command-line. The required arguments that define the input and output paths include:

- `gold_dir`: *str*, path to the input directory with gold annotations in BRAT format, e.g. \mbox{`/home/gold/'}
- `predict_dir`: *str*, path to the input directory with predicted annotations in BRAT format, e.g. \mbox{`/home/predict/'}
- `output`: *str*, path for the output CSV file that will contain the evaluation results, e.g. , \mbox{`/home/scoring.csv'}
\end{itemize}
The optional arguments define the evaluation criteria:
\begin{itemize}
    \item `score_trig`: *str}, trigger scoring criterion, options include \{`exact', `overlap', `min\_dist'\}
    \item `score_span`: *str}, span-only argument scoring criterion, options include \{`exact', `overlap', `partial'\}
    \item `score_labeled`: *str}, span-with-value argument scoring criterion, options include \{`exact', `overlap', `label'\}    
\end{itemize}

Below is an example usage:
\begin{lstlisting}
python3 score_brat.py /home/gold/ /home/predict/ /home/scoring.csv
--score_trig min_dist --score_span exact --score_labeled label
\end{lstlisting}
