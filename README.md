# iris-species-classifier

ml platform i built around the iris dataset. started small but grew into a full pipeline — data expansion, model comparison, auto tuning, charts and a web ui.

## what it does

- expands 150 original samples to 1500 with noise augmentation
- compares 6 models with 5-fold cross validation
- runs grid search on the best model type
- saves metrics, charts and trained model
- web dashboard with live predictions and history

## quick start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
python app.py
```

open http://127.0.0.1:5000

## pipeline

`run.py` runs everything in order:

```bash
python run.py
```

or step by step:

```bash
python expand_data.py
python train.py
python plot.py
python evaluate.py
```

## scripts

| script | what it does |
|--------|--------------|
| `expand_data.py` | builds original + expanded csv, saves stats |
| `train.py` | trains 6 models, tunes best one, saves to models/ |
| `plot.py` | confusion matrix, model comparison, feature importance charts |
| `evaluate.py` | prints all metrics in terminal |
| `predict.py` | single prediction from cli |
| `batch_predict.py` | batch prediction from csv |
| `app.py` | flask api + web dashboard |
| `run.py` | full pipeline runner |

## data

- `data/iris_original.csv` — 150 real samples
- `data/iris_expanded.csv` — 1500 samples (generated, not in git)
- `data/stats.json` — counts and metadata
- `data/predictions.json` — last 50 web predictions
- `data/samples.csv` — batch predict examples

## api endpoints

| endpoint | method | description |
|----------|--------|-------------|
| `/` | GET | web dashboard |
| `/health` | GET | server status |
| `/dataset` | GET | data expansion stats |
| `/metrics` | GET | full model metrics |
| `/history` | GET | prediction history |
| `/samples` | GET | preset input values |
| `/predict` | POST | make prediction |
| `/chart/<name>` | GET | png charts |

predict body:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

## variable names in code

i kept names short on purpose. here's what they mean:

### expand_data.py

| name | what it is |
|------|------------|
| `a` | iris dataset object |
| `b` | feature matrix |
| `c` | target labels |
| `d` | class names |
| `e` | feature name list |
| `f` | data folder path |
| `g` | original csv path |
| `h` | expanded csv path |
| `i` | stats json path |
| `j` | csv header columns |
| `k` | file handle for original csv |
| `l` | csv writer for original |
| `m` | row index in loop |
| `n` | one original row |
| `o` | noise scale per feature |
| `p` | augmented copies per sample |
| `q` | list tracking expanded rows |
| `r` | file handle for expanded csv |
| `s` | csv writer for expanded |
| `t` | original row in expanded file |
| `u` | augment loop index |
| `v` | noisy feature array |
| `w` | one augmented row |
| `x` | stats dict |
| `y` | file handle for stats json |

### train.py

| name | what it is |
|------|------------|
| `a` | expanded csv path |
| `b` | feature rows list |
| `c` | species labels list |
| `m` | feature column names |
| `g` | csv dict reader |
| `h` | one csv row |
| `i` | feature values from row |
| `j` | feature numpy array |
| `k` | label array |
| `n` | label encoder |
| `o` | encoded labels |
| `p` | train features (60%) |
| `q` | test features (20%) |
| `r` | train labels |
| `s` | test labels |
| `t` | train split after val cut |
| `u` | val features |
| `v` | train labels after val cut |
| `w` | val labels |
| `x` | standard scaler |
| `y` | scaled train data |
| `z` | scaled val data |
| `aa` | scaled test data |
| `ab` | dict of 6 models |
| `ac` | cv scores per model |
| `ad` | best model name |
| `ae` | best cv score tracker |
| `af` | model name in cv loop |
| `ag` | model instance in loop |
| `ah` | cv score array |
| `ai` | mean cv score |
| `aj` | std cv score |
| `ak` | grid search param grids |
| `al` | grid search object |
| `am` | tuned best estimator |
| `an` | val predictions |
| `ao` | test predictions |
| `ap` | val accuracy |
| `aq` | test accuracy |
| `ar` | classification report dict |
| `as_` | confusion matrix list |
| `at` | feature importance dict |
| `au` | feature name in importance loop |
| `av` | importance value |
| `aw` | mean abs coefficients |
| `ax` | models folder path |
| `ay` | stats json path |
| `az` | stats file handle |
| `ba` | loaded stats dict |
| `bb` | full metrics dict |
| `bc` | metrics file handle |

### plot.py

| name | what it is |
|------|------------|
| `a` | metrics json path |
| `b` | metrics file handle |
| `c` | parsed metrics |
| `d` | output folder |
| `e` | confusion matrix array |
| `f` | class names |
| `g` | figure for confusion matrix |
| `h` | axes for confusion matrix |
| `i` | imshow object |
| `j` | row index in text loop |
| `k` | col index in text loop |
| `l` | confusion matrix png path |
| `m` | model names for bar chart |
| `n` | cv scores for bar chart |
| `p` | figure for model comparison |
| `q` | axes for model comparison |
| `r` | bar chart object |
| `s` | bar in label loop |
| `t` | score value in label loop |
| `u` | model comparison png path |
| `v` | feature names for importance |
| `w` | importance values |
| `x` | figure for feature importance |
| `y` | axes for feature importance |
| `z` | bar chart for importance |
| `aa` | feature importance png path |

### app.py

| name | what it is |
|------|------------|
| `a` | flask app |
| `b` | models folder |
| `c` | loaded scaler |
| `d` | loaded classifier |
| `e` | metrics file handle |
| `f` | parsed metrics |
| `g` | class names |
| `h` | feature key names for api |
| `i` | predictions json path |
| `j` | stats json path |
| `k` | load history function |
| `l` | history file handle |
| `m` | save history function |
| `n` | one history entry |
| `o` | history list before save |
| `p` | history file handle for write |
| `t` | index route |
| `u` | chart route |
| `q` | chart filename param |
| `r` | chart file path |
| `v` | health route |
| `w` | dataset route |
| `x` | stats file handle |
| `y` | metrics route |
| `z` | history route |
| `aa` | samples route |
| `ab` | predict route |
| `ac` | request json body |
| `ad` | collected feature values |
| `ae` | one feature key in loop |
| `af` | input numpy array |
| `ag` | scaled input |
| `ah` | predicted class index |
| `ai` | probability array |
| `aj` | predicted species name |
| `ak` | response dict |
| `al` | loop index for probabilities |

### static/app.js

| name | what it is |
|------|------------|
| `a` | form element |
| `b` | result output |
| `c` | probability bars |
| `d` | model info container |
| `e` | error element |
| `f` | stats container |
| `g` | history container |
| `h` | presets container |
| `i` | load dashboard function |
| `j` | load history function |
| `k` | load presets function |
| `l` | render probability bars |
| `m` | probabilities arg or event |

## stack

python, scikit-learn, numpy, joblib, matplotlib, flask, html/css/js

## notes

expanded data uses small gaussian noise (6% of feature std). not real new data but good for practicing larger datasets and pipeline workflows.
