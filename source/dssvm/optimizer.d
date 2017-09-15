module dssvm.optimizer;

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
 (Cone) QuadProgramOptimizer solve this

 minimize 0.5 x^T P x + q^T x

 s.t. G x <= h, A x = b

 +/
struct ConeQuadProgramOptimizer(Float=double) {
    alias S(size_t n) = Slice!(Contiguous, [n], Float*);
    alias Vec = S!1;
    alias Mat = S!2;

    Mat P, G, A;
    Vec q, h, b;

    private Vec _s, _x, _y, _z; // iter values
    private Mat _W; // see Sec 4. Nesterov-Todd scaling
    private size_t nvars = 0;

    auto init() {
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
                if (s.empty) { s = zeros!Float(nvars); }
            }
            static if (is(S == Mat)) {
                if (s.empty) { s = zeros!Float(nvars, nvars); }
            }
        }
    }

    auto residual() {
        // TODO: use stack at eq. 17

        // TODO: use gemm at eq. 17

    }

    auto iter() {
        // TODO: init _s, ..., _z
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
    ConeQuadProgramOptimizer!double optimizer = {
    P: iota(2, 2).as!double.slice,
    b: iota(2).as!double.slice,
    };
    optimizer.init();
    assert(optimizer.nvars == 2);
    optimizer.writeln;
}
