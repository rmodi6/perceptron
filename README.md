# Perceptron
# Adaptive Boosting
The path to dataset can be provided using the `dataset` parameter and `mode` parameter can be used to specify the mode in which to execute perceptron. There are two modes available: `erm` for Empirical Risk Minimization and `cv` for 10 fold Cross Validation. For example:
```bash
python perceptron.py --dataset 'path/to/dataset' --mode erm
```