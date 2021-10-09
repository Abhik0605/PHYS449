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
python main.py <csv input> <json hparams input>
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
│    └ even_mnist.csv
│    └ hyper_param.json
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
└──main.py #run this file
```
