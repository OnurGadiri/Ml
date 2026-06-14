# iris-species-classifier

small ml project i built to practice training and saving models. uses sklearn iris dataset — 4 flower measurements, 3 species.

## what it does

- trains a random forest on the iris dataset
- saves model + scaler to `models/`
- you can run predictions from the command line after training

## setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## train

```bash
python train.py
```

you should see accuracy printed and files inside `models/`:
- `classifier.pkl` — trained model
- `scaler.pkl` — feature scaler (fit on train split only)
- `metrics.json` — accuracy + classification report

## predict

pass 4 numbers in this order: sepal length, sepal width, petal length, petal width

```bash
python predict.py 5.1 3.5 1.4 0.2
```

## variable names in code

i kept names short on purpose. here's what they mean:

### train.py

| name | what it is |
|------|------------|
| `a` | full iris dataset object from sklearn |
| `b` | feature matrix (all samples, 4 columns) |
| `c` | labels / target array |
| `d` | class names (setosa, versicolor, virginica) |
| `e` | training features |
| `f` | test features |
| `g` | training labels |
| `h` | test labels |
| `i` | standard scaler instance |
| `j` | scaled training data |
| `k` | scaled test data |
| `l` | random forest classifier |
| `m` | predictions on test set |
| `n` | accuracy score |
| `o` | classification report as dict |
| `p` | path to models folder |
| `q` | metrics dict written to json |
| `r` | file handle for metrics.json |

### predict.py

| name | what it is |
|------|------------|
| `a` | list of 4 input features from argv |
| `b` | reshaped numpy array (1 sample) |
| `c` | models directory path |
| `d` | loaded scaler |
| `e` | loaded classifier |
| `f` | scaled input |
| `g` | predicted class index |
| `h` | probability for each class |
| `i` | file handle for metrics.json |
| `j` | parsed metrics from json |
| `k` | class names from metrics |
| `l` | final predicted label |
| `m` | loop index |
| `n` | probability value in loop |

## stack

python, scikit-learn, numpy, joblib

## notes

dataset is built into sklearn so no manual download. test split is 25%, stratified. i might add a notebook or try other models later.
