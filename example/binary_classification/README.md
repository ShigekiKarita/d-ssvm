# d-ssvm example: binary classification

This example shows binary (odd or even digit) classification benchmarks comparing linear subgradient SVM between d-ssvm and pystruct.

```
$ python main.py
Score with pystruct subgradient ssvm: 0.857778 (took 17.017000 seconds)
Score with sklearn and libsvm: 0.920000 (took 0.149421 seconds)
$ dub run --compiler=ldc2 --build=release-nobounds
Score with d-ssvm subgradient ssvm: 0.924444 (took 0.152297 seconds)
```

@Intel(R) Core(TM) i5-4210U CPU @ 1.70GHz
