module ranges;

import mir.ndslice;


// TODO: add to numir
import mir.random.engine.xorshift;
auto gen = Xorshift(1);

auto permutation(T...)(T t) {
    import numir.core : arange;
    import mir.ndslice : slice;
    import mir.random.algorithm : shuffle;
    auto a = arange(t).slice;
    shuffle(gen, a);
    return a;
}


struct MiniBatchRange {
    import numir.core : arange;
    import mir.ndslice : Slice, Contiguous, slice;
    import mir.utility : min;
    import std.conv : to;

    const size_t total;
    const size_t batchSize;
    const bool isSkipLast = false;
    private size_t _consumed = 0;
    const Slice!(Contiguous, [1LU], size_t*) indices;

    this (size_t total, size_t batchSize, bool isSkipLast=false, bool isShuffle=true) {
        this.total = total;
        this.batchSize = batchSize;
        this.isSkipLast = isSkipLast;
        this.indices = isShuffle ? permutation(total) : arange(total).slice;
    }

    @property auto front() {
        return this.indices[_consumed .. min(_consumed + batchSize, $)];
    }

    void popFront() {
        this._consumed = min(this.total, this._consumed + this.batchSize);
    }

    @property bool empty() {
        return this.isSkipLast
            ? this.total <= this._consumed + this.batchSize
            : this.total <= this._consumed;
    }
}


unittest {
    import std.stdio;
    import mir.ndslice;

    size_t[][] actual;

    foreach (ids; MiniBatchRange(7, 3, false, false)) {
        actual ~= cast(size_t[]) ids.ndarray;
    }
    assert(actual == [[0, 1, 2], [3, 4, 5], [6]]);
    actual = [];
    foreach (ids; MiniBatchRange(8, 3, false, false)) {
        actual ~= cast(size_t[]) ids.ndarray;
    }
    assert(actual == [[0, 1, 2], [3, 4, 5], [6, 7]]);

    actual = [];
    foreach (ids; MiniBatchRange(7, 3, true, false)) {
        actual ~= cast(size_t[]) ids.ndarray;
    }
    assert(actual == [[0, 1, 2], [3, 4, 5]]);
    actual = [];
    foreach (ids; MiniBatchRange(8, 3, true, false)) {
        actual ~= cast(size_t[]) ids.ndarray;
    }
    assert(actual == [[0, 1, 2], [3, 4, 5]]);

    actual = [];
    foreach (ids; MiniBatchRange(1, 3, false, false)) {
        actual ~= cast(size_t[]) ids.ndarray;
    }
    assert(actual == [[0]]);
    actual = [];
    foreach (ids; MiniBatchRange(2, 3, true, false)) {
        actual ~= cast(size_t[]) ids.ndarray;
    }
    assert(actual == []);
}
