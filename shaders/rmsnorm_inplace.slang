import goldy_exp;

groupshared float scratch[256];

[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(
    Scattered<float> x,
    BufRO<float> weight,
    uint size,
    uint weight_offset,
    GroupThreadId lid
) {
    uint local = lid.x;
    float ss = 0.0;
    for (uint j = local; j < size; j += 256u) {
        float v = x[j];
        ss += v * v;
    }
    scratch[local] = ss;
    for (uint i = 0u; i < 8u; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local + (1u << i) < 256u)
            ss = ss + scratch[local + (1u << i)];
        GroupMemoryBarrierWithGroupSync();
        scratch[local] = ss;
    }
    GroupMemoryBarrierWithGroupSync();
    ss = scratch[0];
    ss /= (float)size;
    ss += 1e-5;
    ss = 1.0 / sqrt(ss);
    for (uint j = local; j < size; j += 256u) {
        x[j] = weight[weight_offset + j] * (ss * x[j]);
    }
}
