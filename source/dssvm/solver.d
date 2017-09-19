module dssvm.cqps;

/++

Note:

Considering a minimization part of the 1-slack cutting plane algorithm for SSVM,

min_{w, e} 0.5 || w ||^2 + c e

s.t.

w^T \sum_{i=1}^{N} d_o(\bar{y}_i) \geq \sum_{i=1}^{N} I(\bar{y}_i \neq y_i) - e,

\{ \bar{y}_i \}_{i \in [1,2, \dots, N]} \in V


where
- w is a weight matrix (numClass x numFeature) to be optimized
- c is a constant value (a.k.a. penalty)
- d is a function of a feature difference: d_i(y) = f(x_i, y_i) - f(x_i, y)
- N is a number of samples
- V is a set of constraints

Referrence:
- http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
- http://cvxopt.org/userguide/coneprog.html#quadratic-programming
- http://cvxopt.org/userguide/coneprog.html#quadratic-cone-programs

In this paper p.11 "5  Path-following algorithm for cone QPs",

 +/


import std.stdio : writeln;

import mir.ndslice;

import numir;



/++
 LinearProgramSolver

 minimize:     c^T x
 subject to:   A x = b, x >= 0

 +/
struct LinearProgramSolver(Float=double) {
    
}


/++
 minimize:     0.5 x^T P x + q^T x
 subject to:   G x <= h, A x = b
 +/
struct ConeQuadProgramSolver(Float=double) {
    alias S(size_t n) = Slice!(Universal, [n], Float*);
    alias Vec = S!1;
    alias Mat = S!2;

    Mat P, G, A;
    Vec q, h, b;

    private Vec _s, _x, _y, _z; // iter values
    private Mat _W; // see Sec 4. Nesterov-Todd scaling
    private size_t nvars = 0;
    private bool isInitialized = false;

    ref auto initialize() {
        assert(!isInitialized);

        // validate shapes
        foreach (i, ref s; this.tupleof) {
            static if (isSlice!(typeof(s))) {
                if (s.empty) {
                    continue;
                }
                else if (nvars == 0) {
                    nvars = s.shape[0];
                } else {
                    assert(s.shape.sliced.all!(m => m == nvars));
                }
            }
        }
        assert(nvars != 0);

        // initialize undefined slices
        foreach (i, ref s; this.tupleof) {
            alias S = typeof(s);
            static if (is(S == Vec)) {
                if (s.empty) { s = zeros!Float(nvars).universal; }
            }
            static if (is(S == Mat)) {
                if (s.empty) { s = zeros!Float(nvars, nvars).universal; }
            }
        }
        isInitialized = true;
        return this;
    }

    auto isOptimal() {
        
    }

    auto residuals() {
        // NOTE: stack is renamed to concatenation at eq. 17
        auto At_Gt = concatenation!1(A.transposed!(0, 1), G.transposed!(0, 1)); // shape: [2, 4]
        auto At00_Gt00 = At_Gt.pad!"post"(0, [nvars, nvars]); // shape: [6, 4]
        auto PAG = concatenation!0(P, A, G); // shape: [6, 2]
        auto PAG_At00_Gt00 = concatenation!1(PAG, At00_Gt00); // shape: [6, 6];
        return null;
    }

    auto loop() {
        // TODO: init _s, ..., _z
        auto rs = this.residuals();
        // LinearProgramSolver();
    }

    auto toString() {
        import std.format : format;
        auto result =  typeof(this).stringof ~ " {\n";
        foreach (i, s; this.tupleof) {
            if (isSlice!(typeof(s))) {
                result ~= "  %s: %s\n".format(this.tupleof[i].stringof, s);
            }
        }
        result ~= "}";
        return result;
    }
}


unittest {
    import numir;
    ConeQuadProgramSolver!double cqps = {
    P: iota(2, 2).as!double.slice.universal,
    b: iota(2).as!double.slice.universal,
    };
    cqps.initialize();
    assert(cqps.nvars == 2);
    cqps.writeln;
    cqps.residuals();
}



/++
 minimize:     0.5 x^T H x + f^T x
 subject to:
     sum_{i in I_k} x[i] == b[k]  for all k such that S[k] == 0 
     sum_{i in I_k} x[i] <= b[k]  for all k such that S[k] == 1

 where:
   + H is a diagonal matrix (nvars x nvars)
   + a, x are vectors (nvars)
   + l, u are vectors (nvars) that can consists +-inf
   + b is a scalar

http://cmp.felk.cvut.cz/~xfrancv/libqp/html/
 +/
struct DiagonalSimplexQuadProgramSolver(Float=double) {
    alias S(size_t n) = Slice!(Universal, [n], Float*);
    alias Vec = S!1;
    alias Mat = S!2;

    Vec diagH, a, x, l, u, b;
}
