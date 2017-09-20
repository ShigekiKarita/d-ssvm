module dssvm.trainer;

import dssvm.model;
import mir.ndslice : each, map, Slice, Universal, zip;
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

    auto update(X, Y)(X xs, Y ys, size_t totalSize) {
        auto numBatch = xs.shape[0];
        auto sumDiff = numir.zeros!FloatT(this.model.numClass, this.model.numFeature);
        foreach (b; 0 .. numBatch) {
            // TODO: is it possible to `zip!true(xs.pack!1, ys)` ?
            auto x = xs[b];
            auto y = ys[b];
            const yid = this.model.getTargetId(y);
            const maxId = this.model.search(x, y).maxIndex;
            auto e = this.model.outputs[maxId];
            auto xe = this.model.feature(x, e);
            auto xy = this.model.feature(x, y);
            zip!true(sumDiff[yid], xe, xy).each!((z) {
                    z[0] += z[1] - z[2];
                });
        }

        const denom = cast(double) xs.shape[0] * this.model.C;
        zip!true(this.model.weight, sumDiff).each!((z) {
                z[0] -= this.lr * (z[0] / denom + z[1]);
            });
    }

    auto fit(X, Y)(X xs, Y ys) {
        const totalSize = ys.shape[0];
        const miniBatchSize = this.batchSize == 0 ? totalSize : this.batchSize;
        foreach (i; 0 .. maxIter) {
            foreach (batchIds; MiniBatchRange(totalSize, miniBatchSize)) {
                auto bxs = xs[batchIds];
                auto bys = ys[batchIds];
                this.update(bxs, bys, totalSize);
            }
        }
    }
}
