# d-ssvm example: binary classification

This example shows binary (odd or even digit) classification benchmarks comparing linear subgradient SVM between d-ssvm and pystruct.

```
$ python main.py
Score with pystruct subgradient ssvm: 0.857778 (took 6.552562 seconds)
Score with sklearn and libsvm: 0.920000 (took 0.063966 seconds)

$ dub run --compiler=ldc2 --build=release-nobounds
Score with d-ssvm subgradient ssvm: 0.908889 (took 0.035989 seconds)
```

Intel(R) Core(TM) i7-4770K CPU @ 3.50GHz
