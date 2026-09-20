import goldy_exp;

[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(Scattered<float> a, BufRO<float> b, uint size, ThreadId id) {
    uint i = id.x;
    if (i >= size) return;
    a[i] += b[i];
}
