import std.stdio : writeln;
import std.datetime : StopWatch;
import std.format : format;

import mir.ndslice;
import numir.io : loadNpy;

import dssvm : BinarySVM, SubgradientTrainer;



void main() {
    // FIXME: add bias by stacking ones to input
    // first, run python res/digits.py
    // TODO: create from here https://github.com/pystruct/pystruct/blob/master/examples/plot_binary_svm.py
    auto trainInput = loadNpy!(double, 2)("./train_data.npy");
    auto trainTarget = loadNpy!(long, 1)("./train_target.npy");
    trainTarget.each!((ref i) => i = (i % 2) * 2 - 1);

    auto testInput = loadNpy!(double, 2)("./test_data.npy");
    auto testTarget = loadNpy!(long, 1)("./test_target.npy");
    testTarget.each!((ref i) => i = (i % 2) * 2 - 1);

    const n_sample = trainInput.shape[0];
    const numFeature = trainInput.shape[1];
    const numClass = 2;

    auto model = new BinarySVM!double(numFeature, numClass, 10.0);
    auto trainer = new SubgradientTrainer!(typeof(model))(model);

    StopWatch sw;
    sw.start();
    trainer.fit(trainInput, trainTarget);
    const accuracy = model.evaluate(testInput, testTarget);
    const hns = sw.peek().hnsecs;
    auto elapsed = cast(double) hns / 1e7;  // 1 hnsecs = 100 nsecs = 1e-7 secs
    "Score with d-ssvm subgradient ssvm: %f (took %f seconds)".format(accuracy, elapsed).writeln;
}
