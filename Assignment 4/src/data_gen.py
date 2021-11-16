import numpy as np

def extract_data(lines):
	data = np.zeros((len(lines), len(lines[0])-1))
	for idx1, value in enumerate(lines):
		temp = -np.ones(len(lines[0]) - 1)
		for idx2, i in enumerate(value):
			if i == '+':
				temp[idx2] = 1
			else:
				continue
		data[idx1] = temp

	return data
