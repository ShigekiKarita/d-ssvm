import std.stdio : writeln;

import dssvm : BinarySVM, SubgradientTrainer;


void plotSurface(SSVM, Xs, Ys)(SSVM model, Xs xs, Ys ys, size_t resolution=100) {
    import std.algorithm : map, cartesianProduct, minElement, maxElement;
    import std.array : array;
    import std.range : iota;

    import ggplotd.aes : aes;
    import ggplotd.geom : geomPolygon, geomPoint;
    import ggplotd.ggplotd : GGPlotD, putIn, title;
    import ggplotd.colour : colourGradient;
    import ggplotd.colourspace : XYZ;

    import mir.ndslice : ndarray, ndmap = map, pack;
    import numir : empty;

    const xmin = minElement(xs[0..$, 0]);
    const xmax = maxElement(xs[0..$, 0]);
    const ymin = minElement(xs[0..$, 1]);
    const ymax = maxElement(xs[0..$, 1]);
    const xstep = (xmax - xmin) / resolution;
    const ystep = (ymax - ymin) / resolution;

    double[][] gridArr = cartesianProduct(iota(xmin, xmax, xstep), iota(ymin, ymax, ystep)).map!"[a[0], a[1]]".array;
    auto grid = empty(gridArr.length, 3);
    foreach (i; 0 .. gridArr.length) {
        foreach (j; 0 .. 2) {
            grid[i, j] = gridArr[i][j];
        }
        grid[i, 2] = 1.0;
    }

    auto gridPreds = grid.pack!1.ndmap!(i => model.predict(i)).ndarray;

    auto gg =
       iota(grid.length)
       .map!(i => aes!("x", "y", "colour", "size")(grid[i][0], grid[i][1], gridPreds[i], 1.0))
       .geomPoint
        .putIn(GGPlotD());

    gg = iota(xs.shape[0])
        .map!(i => aes!("x", "y", "colour", "size")(xs[i,0], xs[i,1], ys[i] == 1 ? 1 : 0, 1.0))
        .geomPoint
        .putIn(gg);

    gg = colourGradient!XYZ( "cornflowerBlue-white-crimson" )
        .putIn(gg);

    gg.save("plot.png");
}

void main() {
    import mir.ndslice : map, slice, iota;
    import mir.random : Random, unpredictableSeed;
    import mir.random.variable : BernoulliVariable ;
    import numir.random : normal;

    auto nsamples = 200;
    auto ndim = 2;
    auto xs = normal(nsamples, ndim + 1).slice; // for bias
    // TODO: add to numir.random
    auto gen = Random(unpredictableSeed);
    auto rv = BernoulliVariable!double(0.5);
    auto ys = iota(nsamples).map!(i => cast(long) rv(gen) * 2 - 1).slice;
    foreach (i; 0 .. nsamples) {
        auto x = xs[i];
        auto y = ys[i];
        if (y == 1.0) {
            x[0] += 2.0;
            x[1] += 2.0;
        }
        x[2] = 1.0;  // for bias
    }

    auto model = new BinarySVM!double(3, 0.1);
    auto trainer = new SubgradientTrainer!(typeof(model))(model);
    trainer.fit(xs, ys);
    plotSurface(model, xs, ys);
}
