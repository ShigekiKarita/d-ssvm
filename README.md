# d-ssvm

WIP

dlang (and mir) implementation of structural SVM (SSVM)


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
