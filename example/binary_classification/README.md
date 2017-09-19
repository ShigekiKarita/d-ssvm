# d-ssvm example: binary classification

This example shows binary (odd or even digit) classification benchmarks comparing linear subgradient SVM between d-ssvm and pystruct.

```
$ python main.py
Score with pystruct subgradient ssvm: 0.857778 (took 11.426986 seconds)
Score with sklearn and libsvm: 0.920000 (took 0.069885 seconds)
$ dub run --compiler=ldc2 --build=release-nobounds
Score with d-ssvm subgradient ssvm: 1.000000 (took 0.131741 seconds)
```
