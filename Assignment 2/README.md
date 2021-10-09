# PHYS449 Assignment 2

## Dependencies

- json
- numpy
- argparse
- python 3.8.10
- pytorch
- pandas
- matplotlib

## Running `main.py`

To run `main.py`, use

```sh
python main.py --input_csv <csv input> --param <json hparams input>
```
For help run
```
python main.py --help
```

File structure looks like

```
Assignment 2
│
└─── README.md
│
│
└───inputs
│    └ even_mnist.csv #input data
│    └ hyper_param.json # hyper params
│
│
│
└───notebooks  #notebooks
│    └ test.ipynb
│
└───src
│   └ data_gen.py  # Creates dataset
│   └ nn_gen.py  #neural net
│   └ train.py  #train neural net
│
│
└─── results #place where you store results
│
└──main.py #run this file
```
