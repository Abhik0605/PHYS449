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
python main.py --x-field "-y/np.sqrt(x**2 + y**2)" --y-field "x/np.sqrt(x**2 + y**2)" --ub 1.0 --lb -1.0 --param inputs/param.json --n-tests 4
```
For help run
```
python main.py --help
```