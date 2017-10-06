module dssvm.trainer;

import std.stdio;

import dssvm.model;
import mir.ndslice : each, map, Slice, Universal, zip, iota, sliced, slice;
import std.numeric : dotProduct;
import dssvm.ranges : MiniBatchRange;
import numir.core : maxIndex;
import numir = numir;
import mir.math : sum, Summation;


class SubgradientTrainer(alias SSVM) {
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
        auto sumDiff = numir.zeros_like(this.model.weight);
        foreach (b; 0 .. numBatch) {
            // TODO: is it possible to `zip!true(xs.pack!1, ys)` ?
            auto x = xs[b];
            auto y = ys[b];
            // const yid = this.model.getTargetId(y);
            const maxId = this.model.search(x, y).maxIndex;
            auto e = this.model.outputs[maxId];
            auto xe = this.model.feature(x, e);
            auto xy = this.model.feature(x, y);
            foreach (i; 0 .. sumDiff.shape[0]) {
                sumDiff[i] += xe[i] - xy[i];
            }
            // zip!true(sumDiff, xe, xy).each!((z) {
            //         z[0] += z[1] - z[2];
            //     });
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




class OneSlackTrainer(alias SSVM) {
    import std.container : RedBlackTree;
    alias FloatT = SSVM.FloatT;

    SSVM model;
    FloatT slack, tol;
    size_t maxIter;

    this(SSVM model, size_t maxIter=100, FloatT tol=1e-6) {
        this.model = model;
        this.maxIter = maxIter;
        this.tol = tol;
    }

    auto solveQP(Vec)(Vec averageFeatDiff, FloatT averageLoss) {
        this.model.weight[] = averageFeatDiff * this.model.C;
        this.slack = averageLoss - dotProduct(this.model.weight, averageFeatDiff);
    }

    auto fit(X, Y)(X xs, Y ys) {
        this.slack = 0.0;
        this.model.weight.each!((a) { a = 0.0; });
        auto activeYs = new RedBlackTree!size_t;
        auto ysEstimated = numir.empty_like(ys);
        const numSample = xs.shape[0];

        foreach (niter; 0 .. this.maxIter) {
            if (niter > 1) {
                // TODO: reuse feat diff and weight from update section
                auto sumFeatDiff = numir.zeros!FloatT(this.model.numFeature);
                FloatT sumLoss = 0.0;
                foreach (i; 0 .. numSample) {
                    auto x = xs[i];
                    auto y = ys[i];
                    auto e = ysEstimated[i];
                    // TODO: make this defined in a model-class
                    sumFeatDiff[] += this.model.feature(x, y).slice - this.model.feature(x, e).slice;
                    sumLoss += this.model.loss(y, e);
                }
                auto averageFeatDiff = sumFeatDiff / numSample;
                auto averageLoss = sumLoss / numSample;

                if (averageLoss -  dotProduct(this.model.weight, averageFeatDiff) <= this.slack + this.tol) {
                    "converged".writeln;
                    return niter;
                }

                // TODO: cutting plane with activeYs
                foreach (i; 0 .. numSample) {
                    auto x = xs[i];
                    auto y = ys[i];
                    auto e = ysEstimated[i];
                    // TODO: make this defined in a model-class
                    sumFeatDiff[] += this.model.feature(x, y).slice - this.model.feature(x, e).slice;
                    sumLoss += this.model.loss(y, e);
                }

                this.solveQP(averageFeatDiff, averageLoss);
            }

            // update activeSet
            foreach (i; 0 .. numSample) {
                // TODO: make this searchMax as a method of model;
                auto yMaxId = this.model.outputs.sliced
                    .map!(y => this.model.loss(y, ys[i]) + this.model.weightedFeature(xs[i], ys[i]))
                    .maxIndex;
                auto yMax = this.model.outputs[yMaxId];
                ysEstimated[i] = yMax;
                activeYs.insert(yMax);
            }
        }
        return this.maxIter;
    }
}

unittest {
    import std.stdio;
    import dssvm.model;
    import mir.ndslice;
    import numir.random;

    size_t numFeature = 3;
    auto model = new BinarySVM!double(numFeature);
    auto xs = normal(50, numFeature).slice;
    auto ts = xs.ipack!1.map!(x => x[0] > 0 ? 1L : -1L).slice;
    writeln(xs.shape, ts.shape);

    auto trainer = new OneSlackTrainer!(typeof(model))(model, 10);
    trainer.fit(xs, ts).writeln;
}
