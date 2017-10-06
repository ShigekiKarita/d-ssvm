import std.getopt;
import std.stdio : writeln;
import std.datetime : StopWatch;
import std.format : format;

import mir.ndslice;
import numir.io : loadNpy;
import numir.random : RNG;

import dssvm : MultiSVM, SubgradientTrainer, OneSlackTrainer;


auto addBias(S)(S s) {
    return s.pad!"post"(1, [0, 1]).slice;
}


class Config {
    double lr = 1e-6, C = 0.1;
    size_t maxIter = 1000, batchSize = 0, seed = 777;

    this(string[] args) {
        auto parsed = getopt(
            args,
            "lr", "learning rate for SubgradientTrainer", &lr,
            "C", "weakness of SSVM gaussian prior (higher C results lower regularization)", &C,
            "batchSize", "mini batch size for SubgradientTrainer", &batchSize,
            "maxIter", "maximum number of iterations for SubgradientTrainer", &maxIter,
            "seed", "seed for random number generator", &seed
        );
        if (parsed.helpWanted) {
            defaultGetoptPrinter("d-ssvm example of multi classification for digits (0-9)",
                                 parsed.options);
        }
    }
}

void main(string[] args) {
    auto config = new Config(args);
    RNG.setSeed(cast(uint) config.seed);

    auto trainInput = loadNpy!(double, 2)("./train_data.npy").addBias;
    auto trainTarget = loadNpy!(long, 1)("./train_target.npy");

    auto testInput = loadNpy!(double, 2)("./test_data.npy").addBias;
    auto testTarget = loadNpy!(long, 1)("./test_target.npy");

    const n_sample = trainInput.shape[0];
    const numFeature = trainInput.shape[1];
    const numClass = 10;

    {
        auto model = new MultiSVM!double(numFeature, numClass, config.C);
        // NOTE: pystruct example uses batch_size=None that means minibatch=len(X)
        // https://github.com/pystruct/pystruct/blob/master/pystruct/learners/subgradient_ssvm.py#L259
        auto trainer = new OneSlackTrainer!(typeof(model))(model, 100, 1e-10);

        StopWatch sw;
        sw.start();
        auto niter = trainer.fit(trainInput, trainTarget);
        {
            const accuracy = 1.0 - model.evaluate(trainInput, trainTarget);
            const hns = sw.peek().hnsecs;
            auto elapsed = cast(double) hns / 1e7;  // 1 hnsecs = 100 nsecs = 1e-7 secs
            "Train Score with d-ssvm 1-slack ssvm: %f (took %f seconds) %d iter".format(accuracy, elapsed, niter).writeln;
        }
        {
            const accuracy = 1.0 - model.evaluate(testInput, testTarget);
            const hns = sw.peek().hnsecs;
            auto elapsed = cast(double) hns / 1e7;  // 1 hnsecs = 100 nsecs = 1e-7 secs
            "Score with d-ssvm 1-slack ssvm: %f (took %f seconds) %d iter".format(accuracy, elapsed, niter).writeln;
        }
    }

    {
        auto model = new MultiSVM!double(numFeature, numClass, config.C);
        // NOTE: pystruct example uses batch_size=None that means minibatch=len(X)
        // https://github.com/pystruct/pystruct/blob/master/pystruct/learners/subgradient_ssvm.py#L259
        auto trainer = new SubgradientTrainer!(typeof(model))(model, config.lr, config.maxIter, config.batchSize);

        StopWatch sw;
        sw.start();
        trainer.fit(trainInput, trainTarget);
        {
            const accuracy = 1.0 - model.evaluate(trainInput, trainTarget);
            const hns = sw.peek().hnsecs;
            auto elapsed = cast(double) hns / 1e7;  // 1 hnsecs = 100 nsecs = 1e-7 secs
            "Train Score with d-ssvm subgradient ssvm: %f (took %f seconds)".format(accuracy, elapsed).writeln;
        }
        {
            const accuracy = 1.0 - model.evaluate(testInput, testTarget);
            const hns = sw.peek().hnsecs;
            auto elapsed = cast(double) hns / 1e7;  // 1 hnsecs = 100 nsecs = 1e-7 secs
            "Score with d-ssvm subgradient ssvm: %f (took %f seconds)".format(accuracy, elapsed).writeln;
        }
    }


}
