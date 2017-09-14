import std.algorithm : sum;
import std.stdio;

import mir.ndslice : each, map, Slice, Universal;

import numir : view, RNG;
import numir.core : maxIndex;
import numir.io : loadNpy;
import numir.random : uniform;

import ranges : MiniBatchRange;


mixin template StructuralSVM(Float=double) {
    alias FloatT = Float;

    version(LDC) { import ldc.intrinsics : fmax = llvm_maxnum; }
    else { import std.math : fmax; }
    import mir.ndslice;
    import std.numeric : dotProduct;
    import std.algorithm : sum;

    const size_t numFeature, numClass;
    Slice!(Contiguous, [2LU], Float*) weight;
    Float penalty;

    this (size_t numFeature, size_t numClass, Float penalty = 1.0) {
        this.numFeature = numFeature;
        this.numClass = numClass;
        this.weight = uniform!Float(numClass, numFeature);
        this.weight.each!((ref a) => a = a * 0.01);
        this.penalty = penalty;
    }

    auto prior() {
        // Spherical Gaussian prior: Normal(0.0, 1.0) a.k.a. L2 regularizer
        return reduce!("a + b ^^ 2")(0.0, this.weight) / 2.0;
    }

    auto weightedFeature(X, Y)(X x, Y y) {
        // TODO: use matrix-vector product http://docs.mir.dlang.io/latest/mir_glas_l2.html#gemv
        auto f = this.feature(x, y);
        return this.weight.pack!1.map!(
            w => dotProduct(w, f)
            ).sum;
    }

    auto riskOne(X, Y)(X x, Y y) {
        import std.algorithm : maxElement;
        return search(x, y).maxElement - this.weightedFeature(x, y);
    }

    auto risk(X, Y)(X xs, Y ys) {
        assert(xs.shape[0] == ys.shape[0]);
        return this.prior / this.penalty
            + iota(xs.shape[0]).map!(i => riskOne(xs[i], ys[i])).sum;
    }

    /* user defined parts
    auto feature(X, Y)(X xs, Y y);
    auto loss(Y)(Y yTrue, Y yExpect);
    auto search(X, Y)(X x, Y y);
    */
}

class BinarySVM(Float=double) {
    import mir.ndslice;

    mixin StructuralSVM!Float;

    enum outputs = [-1L, 1L];

    auto feature(X, Y)(X xs, Y y) {
        assert(xs.shape == [this.numFeature]);
        return xs.map!(x => x * y / 2.0);
    }

    auto loss(long yTrue, long yExpect) {
        return yTrue == yExpect ? 0.0 : 1.0;
    }

    auto search(X)(X x, long y) {
        return this.outputs.sliced
            .map!((yi) => this.loss(y, yi) + this.weightedFeature(x, yi));
    }

    auto evaluate(X, Y)(X xs, Y ys) {
        assert(xs.shape[0] == ys.shape[0]);
        auto numBatch = xs.shape[0];

        auto result = 0.0;
        foreach (b; 0 .. numBatch) {
            auto ye = this.outputs.sliced.map!(i => this.weightedFeature(xs[b], i)).maxIndex;
            result += this.loss(ys[b], ye);
        }
        return result / numBatch;
    }
}


auto index_select(S, I)(S sl, I idx) {
    return idx.map!(i => sl[i]);
}


class SubgradientOptimizer(alias SSVM) {
    import mir.ndslice;
    import numir = numir;
    alias FloatT = SSVM.FloatT;

    SSVM model;
    double lr;
    size_t maxIter, batchSize;

    this(SSVM model, double lr=1e-2, size_t maxIter=100, size_t batchSize=10) {
        this.model = model;
        this.lr = lr;
        this.maxIter = maxIter;
        this.batchSize = batchSize;
    }

    auto grad(X, Y)(X xs, Y ys) {
        auto numBatch = xs.shape[0];

        auto sumDiff = numir.zeros!FloatT(this.model.numFeature);
        foreach (b; 0 .. numBatch) {
            auto ye = this.model.search(xs[b], ys[b]).maxIndex;
            auto xe = this.model.feature(xs[b], ye);
            auto xy = this.model.feature(xs[b], ys[b]);
            foreach (f; 0 .. this.model.numFeature) {
                sumDiff[f] += xe[f] - xy[f];
            }
        }

        auto gradW = numir.zeros_like(this.model.weight);
        foreach (f; 0 .. this.model.numFeature) {
            foreach (c; 0.. this.model.numClass) {
                gradW[c, f] = this.model.penalty * this.model.weight[c, f] + sumDiff[f];
            }
        }
        return gradW;
    }

    void update(G)(G gradWeight) {
        assert(this.model.weight.shape == gradWeight.shape);
        foreach (f; 0 .. this.model.numFeature) {
            foreach (c; 0.. this.model.numClass) {
                this.model.weight[c, f] -= this.lr * gradWeight[c, f];
            }
        }
    }

    auto fit(X, Y)(X xs, Y ys) {
        for (size_t i = 0; i < maxIter; ++i) {
            foreach (batchIds; MiniBatchRange(xs.shape[0], this.batchSize)) {
                auto bxs = xs.index_select(batchIds);
                auto bys = ys.index_select(batchIds);
                auto gs = this.grad(bxs, bys);
                this.update(gs);
            }
            // this.model.risk(xs, ys).writeln;
        }
    }
}


void main() {
    // FIXME: add bias by stacking ones to input
    // first, run python res/digits.py
    // TODO: create from here https://github.com/pystruct/pystruct/blob/master/examples/plot_binary_svm.py
    auto trainInput = loadNpy!(double, 2)("./res/train_data.npy");
    auto trainTarget = loadNpy!(long, 1)("./res/train_target.npy");
    trainTarget.each!((ref i) => i = (i % 2) * 2 - 1);

    auto testInput = loadNpy!(double, 2)("./res/test_data.npy");
    auto testTarget = loadNpy!(long, 1)("./res/test_target.npy");
    testTarget.each!((ref i) => i = (i % 2) * 2 - 1);

    const n_sample = trainInput.shape[0];
    const numFeature = trainInput.shape[1];
    const numClass = 2;

    auto model = new BinarySVM!()(numFeature, numClass, 10.0);
    writeln("risk: ", model.risk(testInput, testTarget));
    writeln("accuracy: ", model.evaluate(testInput, testTarget));

    auto optimizer = new SubgradientOptimizer!(typeof(model))(model);
    optimizer.fit(trainInput, trainTarget);

    writeln("risk: ", model.risk(testInput, testTarget));
    writeln("accuracy: ", model.evaluate(testInput, testTarget));
}
