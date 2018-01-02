# Result Recycler

A little tool to predict results of calculation based on historical results.

Not a lot of magic here, but useful to get good start values for simulations and sometimes nice in sequential optimization.

If you use it for your own research, please cite [this](http://dx.doi.org/10.14279/depositonce-5915).

A short demo:

```python
#!/usr/bin/python

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'resultrecycler')))

from resultrecycler import ResultRecycler, SampleData

def func(a, b, c):
    return [a * b * b * b - c, a*b + b*c*c]

def jacobian(a, b, c):
    return [[b*b*b, 3*a*b*b, -1], [b, a+c*c, 2*b*c]]

def hessian(a, b, c):
    return [[[0, 3*b*b, 0], [3*b*b, 6*a*b, 0], [0, 0, 0]],
            [[0, 1, 0],     [1, 0, 2*c],       [0, 2*c, 2*b]]]

samples = [[1, 2, 3], [6, 5, 7], [4, 3, 8], [7, 5, 2], [4, 7, 3], [6, 7, 4]]
guess = [3, 4, 5]

print('Correct result:', func(*guess))  # [187 112]

rr = ResultRecycler()
for sample in samples:
    rr.add_data(sample, func(*sample))

print('Guess without derivatives:', rr.calculate(guess))  # [471.75 106.75]

rr = ResultRecycler()
for sample in samples:
    rr.add_data(sample, func(*sample), jacobian(*sample))

print ('Guess with first derivative:', rr.calculate(guess))  # [184 125]

rr = ResultRecycler()
for sample in samples:
    rr.add_data(sample, func(*sample), jacobian(*sample), hessian(*sample))

print ('Guess with second derivative:', rr.calculate(guess))  # [193 103]
```