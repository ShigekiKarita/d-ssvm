# d-ssvm

WIP

dlang (and mir) implementation of structural SVM (SSVM)

see [examples](/example) for information.

### DONE

+ subgradient optimization
+ examples

### TODO

+ more models
  + multi class SSVM
  + latent SSVM
+ find or implement convex optimization library
  + cutting plane (1-slack)
  + concave-convex procedure
+ documentation
+ use mir-algorithm functions instead of `foreach`


### supported compilers

+ LDC v1.4.0- (recommended)
+ DMD v2.074-
