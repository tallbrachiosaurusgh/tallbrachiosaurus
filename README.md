# tallbrachiosaurus

A small Python package for computing discrete derivatives of timeseries data.

Written for CPE 486/586, Machine Learning for Computer Engineering
Applications, at the University of Alabama in Huntsville.

## Installation

    pip install tallbrachiosaurus

## Usage

    from tallbrachiosaurus import diff

    t = [0, 0.1, 0.3, 0.4]
    x = [23.1, 22.5, 23.5, 21.88]
    v = diff(t, x)

The diff function takes two equal-length Python lists, time values and
signal values, and returns the discrete derivative as a list of length
n minus 1.
