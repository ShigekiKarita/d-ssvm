module dssvm.model;

import std.algorithm : sum;
import std.stdio;

import mir.ndslice : each, map, Slice, Universal;

import numir : view, RNG;
import numir.core : maxIndex;
import numir.random : uniform;


mixin template StructuralSVM(Float=double) {
    alias FloatT = Float;

    version(LDC) { import ldc.intrinsics : fmax = llvm_maxnum; }
    else { import std.math : fmax; }
    import mir.ndslice;
    import std.numeric : dotProduct;
    import std.algorithm : sum;

    const size_t numFeature, numClass;
    Slice!(Contiguous, [2LU], Float*) weight;
    Float C;

    auto prior() {
        // Spherical Gaussian prior: Normal(0.0, 1.0) a.k.a. L2 regularizer
        return reduce!("a + b ^^ 2")(0.0, this.weight) / 2.0;
    }

    auto riskOne(X, Y)(X x, Y y) {
        import std.algorithm : maxElement;
        return search(x, y).maxElement - this.weightedFeature(x, y);
    }

    auto risk(X, Y)(X xs, Y ys) {
        assert(xs.shape[0] == ys.shape[0]);
        return this.prior / this.C
            + iota(xs.shape[0]).map!(i => riskOne(xs[i], ys[i])).sum;
    }

    auto search(X, Y)(X x, Y y) {
        return this.outputs.sliced
            .map!(yi => this.loss(y, yi) + this.weightedFeature(x, yi));
    }

    auto evaluate(X, Y)(X xs, Y ys) {
        assert(xs.shape[0] == ys.shape[0]);
        auto numBatch = xs.shape[0];

        auto result = 0.0;
        foreach (b; 0 .. numBatch) {
            auto maxId = this.outputs.sliced.map!(i => this.weightedFeature(xs[b], i)).maxIndex;
            result += this.loss(ys[b], this.outputs[maxId]);
        }
        return result / numBatch;
    }

    /* user defined parts
    auto feature(X, Y)(X xs, Y y);
    auto loss(Y)(Y yTrue, Y yExpect);
    auto search(X, Y)(X x, Y y);
    */
}


class BinarySVM(Float=double) {
    import mir.ndslice;

    mixin StructuralSVM!Float; // TODO: use Base class

    enum outputs = [-1L, 1L];

    this (size_t numFeature, size_t numClass, Float C = 1.0) {
        this.numFeature = numFeature;
        this.numClass = numClass;
        this.weight = uniform!Float(numClass, numFeature);
        this.weight.each!((ref a) => a = a * 0.01);
        this.C = C;
    }

    auto feature(X)(X xs, long y) {
        assert(xs.shape == [this.numFeature]);
        return xs.map!(x => x * y / 2.0);
    }

    auto weightedFeature(X, Y)(X x, Y y) {
        // TODO: use matrix-vector product http://docs.mir.dlang.io/latest/mir_glas_l2.html#gemv
        auto f = this.feature(x, y);
        return this.weight.pack!1.map!(
            w => dotProduct(w, f)
            ).sum;
    }

    auto loss(long yTrue, long yExpect) {
        return yTrue == yExpect ? 0.0 : 1.0;
    }

    auto predict(X)(X x) {
        import std.math : exp;
        auto prob = this.outputs.sliced.map!(i => this.weightedFeature(x, i).exp);
        return prob[1] / (prob[0] + prob[1]);
    }

    auto getTargetId(long y) {
        return y == -1 ? 0 : 1;
    }
}


class MultiSVM(Float=double) {
    import mir.ndslice;

    mixin StructuralSVM!Float;

    const long[] outputs;

    this (size_t numFeature, size_t numClass, Float C = 1.0) {
        import std.array : array;
        import std.range : iota;
        this.numFeature = numFeature;
        this.numClass = numClass;
        this.weight = uniform!Float(numClass, numFeature * numClass);
        this.weight.each!((ref a) => a = a * 0.01);
        this.C = C;
        this.outputs = cast(long[]) iota(numClass).array;
    }

    auto feature(X)(X xs, long y) {
        assert(xs.shape == [this.numFeature]);
        auto pre = y * this.numFeature;
        auto post = (this.numClass - y) * this.numFeature;
        return xs.universal.pad!"pre"(0, [pre]).pad!"post"(0, [post]);
    }

    auto loss(long yTrue, long yExpect) {
        return yTrue == yExpect ? 0.0 : 1.0;
    }

    auto weightedFeature(X, Y)(X x, Y y) {
        auto ws = this.weight[0 .. $, y * this.numFeature .. (y + 1) * this.numFeature];
        return ws.pack!1.map!(
            w => dotProduct(w, x)
            ).sum;
    }

    auto getTargetId(long y) {
        return y;
    }
}
