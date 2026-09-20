import goldy_exp;

[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(Scattered<float> hb, BufRO<float> hb2, uint hidden_dim, ThreadId id) {
    uint i = id.x;
    if (i >= hidden_dim) return;
    float val = hb[i];
    val *= (1.0 / (1.0 + exp(-val)));
    val *= hb2[i];
    hb[i] = val;
}
