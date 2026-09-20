import goldy_exp;

[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(
    BufRO<float> x,
    BufRO<float> w,
    Scattered<float> xout,
    BufRO<uint> control,
    uint n,
    uint d,
    uint w_offset,
    uint out_base,
    uint stride_from_pos,
    ThreadId id
) {
    uint i = id.x;
    if (i >= d) return;
    uint pos = control[1];
    uint out_i = out_base + pos * stride_from_pos + i;
    float sum = 0.0;
    for (uint j = 0u; j < n; j++) {
        sum += w[w_offset + i * n + j] * x[j];
    }
    xout[out_i] = sum;
}
