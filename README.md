# iris-species-classifier

small ml project i built to practice training, comparing models and saving the best one. uses sklearn iris dataset — 4 flower measurements, 3 species.

## what it does

- compares 3 models with 5-fold cross validation
- picks the best model and trains it on the train split
- saves model + scaler + metrics to `models/`
- single prediction from command line or batch prediction from csv
- confusion matrix plot + simple flask api
- web ui to test predictions in the browser

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

output example:
- best model name
- test accuracy
- cv scores for each model

files in `models/`:
- `classifier.pkl` — best trained model
- `scaler.pkl` — feature scaler
- `metrics.json` — comparison results + classification report + confusion matrix

## evaluate

```bash
python evaluate.py
```

## plot

saves confusion matrix image to `output/confusion_matrix.png`:

```bash
python plot.py
```

## predict (single sample)

```bash
python predict.py 5.1 3.5 1.4 0.2
```

order: sepal length, sepal width, petal length, petal width

## predict (batch)

```bash
python batch_predict.py
python batch_predict.py data/samples.csv
```

csv must have the same column names as the iris dataset features.

## web ui

train first, then start the server and open the browser:

```bash
python train.py
python app.py
```

go to: http://127.0.0.1:5000

the page has input fields, live prediction, model stats and confusion matrix chart.

## api

start server:

```bash
python app.py
```

health check:

```bash
curl http://127.0.0.1:5000/health
```

predict:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}"
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
| `e` | feature name list |
| `f` | training features |
| `g` | test features |
| `h` | training labels |
| `i` | test labels |
| `j` | standard scaler instance |
| `k` | scaled training data |
| `l` | scaled test data |
| `m` | dict of 3 model candidates |
| `n` | cv scores dict for each model |
| `o` | name of best model |
| `p` | best cv mean score so far |
| `q` | model name in comparison loop |
| `r` | model instance in loop |
| `s` | cross validation score array |
| `t` | mean cv score |
| `u` | std cv score |
| `v` | selected best model instance |
| `w` | predictions on test set |
| `x` | test accuracy score |
| `y` | classification report as dict |
| `ae` | confusion matrix as nested list |
| `z` | path to models folder |
| `aa` | metrics dict written to json |
| `ab` | file handle for metrics.json |
| `ac` | model name in print loop |
| `ad` | model stats in print loop |

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

### evaluate.py

| name | what it is |
|------|------------|
| `a` | path to metrics.json |
| `b` | file handle for metrics.json |
| `c` | parsed metrics dict |
| `d` | model name in comparison loop |
| `e` | model stats in comparison loop |
| `f` | class name in report loop |
| `g` | per-class metrics dict |
| `h` | class names for matrix header |
| `i` | confusion matrix data |
| `j` | header line string |
| `k` | row index in matrix loop |
| `l` | one row of matrix |
| `m` | formatted row values string |

### batch_predict.py

| name | what it is |
|------|------------|
| `a` | csv file path from argv or default |
| `b` | path object for csv |
| `c` | models directory path |
| `d` | loaded scaler |
| `e` | loaded classifier |
| `f` | file handle for metrics.json |
| `g` | parsed metrics from json |
| `h` | class names from metrics |
| `i` | feature names from metrics |
| `j` | list of feature rows from csv |
| `k` | file handle for csv |
| `l` | csv dict reader |
| `m` | one csv row dict |
| `n` | feature values from one row |
| `o` | numpy array of all rows |
| `p` | scaled input |
| `q` | predicted class indices |
| `r` | probability matrix |
| `s` | original feature row in output loop |
| `t` | predicted class index in loop |
| `u` | probability row in loop |
| `v` | predicted class label string |
| `w` | confidence of predicted class |
| `x` | formatted feature string for print |

### plot.py

| name | what it is |
|------|------------|
| `a` | path to metrics.json |
| `b` | file handle for metrics.json |
| `c` | parsed metrics dict |
| `d` | confusion matrix as numpy array |
| `e` | class names list |
| `f` | matplotlib figure |
| `g` | matplotlib axes |
| `h` | image object from imshow |
| `k` | output folder path |
| `l` | output png path |
| `m` | row index in text loop |
| `n` | column index in text loop |

### app.py

| name | what it is |
|------|------------|
| `a` | flask app instance |
| `b` | models directory path |
| `c` | loaded scaler |
| `d` | loaded classifier |
| `e` | file handle for metrics.json |
| `f` | parsed metrics from json |
| `g` | class names list |
| `h` | health check route function |
| `i` | predict route function |
| `j` | json body from request |
| `k` | required field names list |
| `l` | collected input values |
| `m` | one field name in loop |
| `n` | numpy array from input |
| `o` | scaled input |
| `p` | predicted class index |
| `q` | probability array |
| `r` | predicted species label |
| `s` | loop index for probabilities dict |
| `t` | index route — serves index.html |
| `u` | chart route — serves confusion matrix png |
| `v` | metrics route function |

### static/app.js

| name | what it is |
|------|------------|
| `a` | form element |
| `b` | result output element |
| `c` | probability bars container |
| `d` | model info container |
| `e` | error message element |
| `f` | chart image element |
| `g` | load metrics function |
| `h` | render probability bars function |
| `i` | event or probabilities arg |
| `j` | fetch response or sorted entries |
| `k` | request body or species key |
| `l` | fetch response or probability value |
| `m` | json data or bar row element |

## stack

python, scikit-learn, numpy, joblib, matplotlib, flask, html/css/js

## notes

dataset is built into sklearn. test split is 25%, stratified. web ui runs on the same flask server as the api.
