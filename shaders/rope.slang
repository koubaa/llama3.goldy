import goldy_exp;

[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(
    Scattered<float> q,
    Scattered<float> k,
    BufRO<uint> control,
    uint kv_dim,
    uint head_size,
    uint dim,
    uint loff,
    ThreadId id
) {
    uint i = id.x * 2u;
    if (i >= dim) return;
    uint pos = control[1];
    uint k_base = loff + pos * kv_dim;
    int head_dim = (int)(i % head_size);
    float freq = 1.0 / pow(10000.0, (float)head_dim / (float)head_size);
    float val = (float)pos * freq;
    float fcr = cos(val);
    float fci = sin(val);
    uint rotn = i < kv_dim ? 2u : 1u;
    for (uint v = 0u; v < rotn; v++) {
        float v0;
        float v1;
        if (v == 0u) {
            v0 = q[i];
            v1 = q[i + 1u];
            q[i] = v0 * fcr - v1 * fci;
            q[i + 1u] = v0 * fci + v1 * fcr;
        } else {
            v0 = k[k_base + i];
            v1 = k[k_base + i + 1u];
            k[k_base + i] = v0 * fcr - v1 * fci;
            k[k_base + i + 1u] = v0 * fci + v1 * fcr;
        }
    }
}
