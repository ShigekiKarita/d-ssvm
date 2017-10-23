module dssvm.model;

import std.stdio;

import mir.ndslice : each, map, Slice, Universal;

import numir : view, RNG;
import numir.core : maxIndex;
import numir.random : uniform;


mixin template StructuralSVM(Float=double) {
    alias FloatT = Float;

    import mir.math.common: fmax, sqrt;
    import mir.math.sum : sum;
    import mir.ndslice;

    const size_t numFeature, numClass;
    ContiguousVector!Float weight;
    Float C;

    auto prior() {
        // Spherical Gaussian prior: Normal(0.0, 1.0) a.k.a. L2 regularizer
        return sum!"fast"(this.weight * this.weight) / 2;
    }

    auto riskOne(X, Y)(X x, Y y) {
        import std.algorithm : maxElement;
        return search(x, y).maxElement - this.weightedFeature(x, y);
    }

    auto risk(X, Y)(X xs, Y ys) {
        assert(xs.length == ys.length);
        return this.prior / this.C
            + zip!(xs.ipack!1, ys.ipack!1).map((x, y) => riskOne(x, y)).sum;
    }

    auto search(X, Y)(X x, Y y) {
        return this.outputs.sliced
            .map!(yi => this.loss(y, yi) + this.weightedFeature(x, yi));
    }

    auto evaluate(X, Y)(X xs, Y ys) {
        assert(xs.length == ys.length);
        auto numBatch = xs.length;

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

    this (size_t numFeature, Float C = 1.0) {
        this.numFeature = numFeature;
        this.numClass = 2;
        this.weight = uniform!Float(numFeature).slice;
        this.weight[] *= Float(0.01);
        this.C = C;
    }

    auto feature(X)(X xs, long y)
        if (isVector!X)
    {
        assert(xs.length == this.numFeature);
        return xs * (y / Float(2));
    }

    auto weightedFeature(X, Y)(X x, Y y) {
        auto f = this.feature(x, y);
        return sum!"fast"(this.weight * f);
    }

    auto loss(long yTrue, long yExpect) {
        return yTrue == yExpect ? 0.0 : 1.0;
    }

    auto predict(X)(X x) {
        import mir.math.common : exp;
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
        this.numFeature = numFeature;
        this.numClass = numClass;
        this.weight = uniform!Float(numFeature * numClass).slice;
        this.weight.each!((ref a) => a = a * 0.01);
        this.C = C;
        this.outputs = cast(long[]) iota!long(numClass).array;
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
        auto ws = this.weight[y * this.numFeature .. (y + 1) * this.numFeature];
        return sum!"fast"(ws * x);
    }

    auto getTargetId(long y) {
        return y;
    }
}

