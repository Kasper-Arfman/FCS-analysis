# FCS analysis
Fluorescence Correlation Spectroscopy analysis code belonging to the PhD thesis: Combine and Conquer: Auxin Reponse Factor Multimers Anchoring Nucleotides.

Install the dependencies using 
```Python
pip install -r requirements.txt
```

## Usage
1. Modify the contents of `_script_params.py`. In this file, you specify
 - Where your data is located
 - How the analyses should be performed

2. Modify config.yaml to match your dataset. Specify a calibrant by providing its diffusion coefficient.

3. Run `compute_trace.py`. This constructs intensity time traces and saves them as .pkl files (Python-readable format)

4. Run `compute_acf.py`. This computes Autocorrelation curves - a slow process - and stores the results in .csv format

5. Run `fit_calibrant.py`. Obtain shape parameters of the detection volume, used for subsequent fitting.

6. Run `fit_maxent.py`. Construct the size distribution of the analytes.