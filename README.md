# d-ssvm

WIP

dlang (and mir) implementation of structural SVM (SSVM)

```
$ cd res; python digits.py; cd ..
Score with pystruct subgradient ssvm: 0.857778 (took 11.426986 seconds)
Score with sklearn and libsvm: 0.920000 (took 0.069885 seconds)
$ dub 
Score with d-ssvm subgradient ssvm: 1.000000 (took 0.131741 seconds)
```


### DONE

+ digit binary classification (in app.d)
+ minibatch range
+ subgradient optimization

### TODO

+ use mir-algorithm functions instead of `foreach`
+ benchmarks
+ more models
  + multi class SSVM
  + latent SSVM
+ find or implement convex optimization library
  + cutting plane (1-slack)
  + concave-convex procedure
+ SMO


### supported compilers

+ LDC v1.4.0- (recommended)
+ DMD v2.074-
