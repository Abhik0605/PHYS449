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
python main.py --x-field "-y/np.sqrt(x**2 + y**2)" --y-field "x/np.sqrt(x**2 + y**2)" --ub 1.0 --lb -1.0 --param inputs/param.json --n-tests 4 --res-path results
```
and you can use
```sh
python main.py --param inputs/param.json --res-path results --x-field "np.sin(np.pi*x)+ np.sin(np.pi*y)" --y-field "np.cos(np.pi*y)" --lb -1.0 --ub 1.0 --n-tests 3
```
For help run
```
python main.py --help
```