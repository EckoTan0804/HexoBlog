---
title: Numpy Basics
tags:
- Python
- Numpy
- Basics
- Learn

---

[NumPy](http://www.numpy.org/)  is the fundamental package for scientific computing with Python, adding support for large, multi-dimensional [arrays](https://en.wikipedia.org/wiki/Array_data_structure) and [matrices](https://en.wikipedia.org/wiki/Matrix_(math)), along with a large collection of [high-level](https://en.wikipedia.org/wiki/High-level_programming_language) [mathematical](https://en.wikipedia.org/wiki/Mathematics) [functions](https://en.wikipedia.org/wiki/Function_(mathematics)) to operate on these arrays.

<!--more-->

## prerequisite

+ [Python](https://www.python.org/) installed

+ `NumPy` installed

  ~~~shell
  $ python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
  ~~~



## Using `NumPy`

~~~python
import numpy as np
~~~



## `Array`Basics

### Arrays Attributes

#### Creating Arrays (randomly)

~~~python
np.random.seed(0)

x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array
~~~

Each array has attributes

+  `ndim` (the number of dimensions),
+  `shape` (the size of each dimension), 
+  `size` (the total size of the array)

For example:

```python
print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)
```

Result:

```
x3 ndim:  3
x3 shape: (3, 4, 5)
x3 size:  60
```

### Array Slicing: Accessing Subarrays

```python
x[start:stop:step]
```

If any of these are unspecified, they default to the values `start=0`, `stop=size of dimension`, `step=1`. We'll take a look at accessing sub-arrays in one dimension and in multiple dimensions.

~~~python
x = np.arange(10) # x = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
x[:5]  # first five elements = array([0, 1, 2, 3, 4])
x[5:]  # elements after index 5 = array([5, 6, 7, 8, 9])
x[4:7]  # middle sub-array = array([4, 5, 6])
x[::2]  # every other element = array([0, 2, 4, 6, 8])
x[1::2]  # every other element, starting at index 1 = array([1, 3, 5, 7, 9])
~~~

When the `step` value is negative, the defaults for `start` and `stop` are swapped. This becomes a convenient way to reverse an array.

~~~python
x[::-1]  # all elements, reversed = array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
x[5::-2]  # reversed every other from index 5 = array([5, 3, 1])
~~~

#### Multi-dimensional subarrays

Works in the same way, with multiple slices separated by commas. 

In my case, the array `x2` I have created before is

~~~python
array([[5, 0, 3, 3],
       [7, 9, 3, 5],
       [2, 4, 7, 6]])
~~~

~~~python
x2[:2,1::2]
~~~

~~~
array([[0, 3],
       [9, 5]])
~~~

+ For the first dimension, also the row in 2 dimension array, `:2` (= `:2:`) gives us the first 2 elements starting at index 0, also the first row and the second row
+ For the second dimension, also the column in 2 dimension array, `1::2` gives us every 2 element starting at index 1, also the second and the fourth column in this case

#### Create copy of arrays instead of directly using the subarrays

One important–and extremely useful–thing to know about array slices is that they return *views* rather than *copies* of the array data. Which means if we directly modify the subarray, we'll see that the original array is changed! 

Using the last example, if we change the element's value in x2_sub, for instance:

~~~python
x2_sub[0, 0] = 3
x2_sub
~~~

Result:

~~~
array([[3, 3],
       [9, 5]])
~~~

And let's see what happens in `x2`

~~~python
x2
~~~

Result:

~~~
array([[5, 3, 3, 3],
       [7, 9, 3, 5],
       [2, 4, 7, 6]])
~~~

`x2[0,1]`changes from 0 to 3!!!

Therefore, in order not to affecting the original array when we're handling the subarray, we should copy it explicitly with the `copy()` method.

~~~python
x2_sub = x2[:2,1::2].copy()
~~~

### Rashaping of Arrays

#### Reshape an array into $m*n$ grid

**For this to work, the size of the initial array must match the size of the reshaped array.**

Example:

~~~python
grid = np.arange(1, 10).reshape((3,3))
grid
~~~

Result:

~~~
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
~~~

#### Conversion of a one-dimensional array into a two-dimensional row or column matrix

This can be done with the `reshape` method:

```python
x = np.array([1, 2, 3])

# row vector via reshape
x.reshape((1, 3)) # 1 * 3 matrix
```

```
array([[1, 2, 3]])
```

```python
# column vector via reshape
x.reshape((3, 1))
```

```
array([[1],
       [2],
       [3]])
```

or more easily done by making use of the `newaxis` keyword within a slice operation:

```python
# row vector via newaxis
x[np.newaxis, :]
```

```
array([[1, 2, 3]])
```

```
# column vector via newaxis
x[:, np.newaxis]
```

```
array([[1],
       [2],
       [3]])
```

### Concatenation and splitting

#### Concatenation

+ `np.concatenate`
+ `np.vstack`
+ `np.hstack`

#### Splitting

+ `np.split`
+ `np.hsplit`
+ `np.vsplit`







