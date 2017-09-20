# [WIP] d-ssvm example: multi classification

This example shows digits (0 to 9) classification benchmarks comparing linear subgradient SVM between pystruct, libsvm and d-ssvm.

```
$ python main.py
...
Score with pystruct subgradient ssvm: 0.893333 (took 158.390666 seconds)
Score with sklearn and libsvm: 0.980000 (took 0.067348 seconds)
...
$ dub run --compiler=ldc2 --build=release-nobounds
Score with d-ssvm subgradient ssvm: 0.873333 (took 11.094986 seconds)
```
