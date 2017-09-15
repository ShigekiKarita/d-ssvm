module dssvm.trainer;

import dssvm.model;
import mir.ndslice : each, map, Slice, Universal;
import dssvm.ranges : MiniBatchRange;
import numir.core : maxIndex;


class SubgradientTrainer(alias SSVM) {
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
                auto bxs = xs[batchIds];
                auto bys = ys[batchIds];
                auto gs = this.grad(bxs, bys);
                this.update(gs);
            }
            // this.model.risk(xs, ys).writeln;
        }
    }
}
