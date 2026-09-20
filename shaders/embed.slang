import goldy_exp;

[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(
    BufRO<float> embed,
    BufRO<uint> control,
    Scattered<float> x,
    uint dim,
    ThreadId id
) {
    uint i = id.x;
    if (i >= dim) return;
    uint token = control[0];
    x[i] = embed[token * dim + i];
}
