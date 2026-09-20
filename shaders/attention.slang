import goldy_exp;

groupshared float scratch[256];

[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(
    BufRO<float> q,
    Scattered<float> att,
    Scattered<float> xb,
    BufRO<float> key_cache,
    BufRO<float> value_cache,
    BufRO<uint> control,
    uint kv_dim,
    uint kv_mul,
    uint head_size,
    uint seq_len,
    uint loff,
    GroupThreadId lid,
    GroupId gid
) {
    uint h = gid.x;
    uint local = lid.x;
    uint pos = control[1];
    uint q_base = h * head_size;
    uint att_base = h * seq_len;
    uint kv_head = h / kv_mul;

    float local_max = -1e30;
    for (uint t = local; t <= pos; t += 256u) {
        float score = 0.0;
        uint k_base = loff + t * kv_dim + kv_head * head_size;
        for (uint i = 0u; i < head_size; i++) {
            score += q[q_base + i] * key_cache[k_base + i];
        }
        score /= sqrt((float)head_size);
        att[att_base + t] = score;
        if (score > local_max) local_max = score;
    }

    float val = local_max;
    scratch[local] = val;
    for (uint i = 0u; i < 8u; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local + (1u << i) < 256u)
            val = max(val, scratch[local + (1u << i)]);
        GroupMemoryBarrierWithGroupSync();
        scratch[local] = val;
    }
    GroupMemoryBarrierWithGroupSync();
    float max_val = scratch[0];

    float local_sum = 0.0;
    for (uint t = local; t <= pos; t += 256u) {
        float e = exp(att[att_base + t] - max_val);
        att[att_base + t] = e;
        local_sum += e;
    }
    val = local_sum;
    scratch[local] = val;
    for (uint i = 0u; i < 8u; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local + (1u << i) < 256u)
            val = val + scratch[local + (1u << i)];
        GroupMemoryBarrierWithGroupSync();
        scratch[local] = val;
    }
    GroupMemoryBarrierWithGroupSync();
    float sum = scratch[0];
    for (uint t = local; t <= pos; t += 256u) {
        att[att_base + t] = att[att_base + t] / sum;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint i = local; i < head_size; i += 256u) {
        float acc = 0.0;
        for (uint t = 0u; t <= pos; t++) {
            uint v_base = loff + t * kv_dim + kv_head * head_size;
            acc += att[att_base + t] * value_cache[v_base + i];
        }
        xb[q_base + i] = acc;
    }
}
