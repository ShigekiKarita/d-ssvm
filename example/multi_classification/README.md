# d-ssvm example: multi classification

This example shows digits (0 to 9) classification benchmarks comparing linear subgradient SVM between pystruct, libsvm and d-ssvm.

```
$ python main.py
...
Score with pystruct subgradient ssvm: 0.868889 (took 98.038185 seconds)
Score with sklearn and libsvm: 0.971111 (took 0.078522 seconds)

$ dub run --compiler=ldc2 --build=release-nobounds
Score with d-ssvm subgradient ssvm: 0.873333 (took 1.854730 seconds)
```

Intel(R) Core(TM) i7-4770K CPU @ 3.50GHz
