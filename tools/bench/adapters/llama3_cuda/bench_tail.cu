// Spliced over llama3.cu from the "utilities: time" marker (see build.py).
// Adds checkpoint/tokenizer/count/context CLI, CLOCK_MONOTONIC timing,
// cudaDeviceSynchronize around measured forwards, and schema_version=1 JSON Lines.

#define BENCH_MAX_STEPS 4096
#define BENCH_TEXT_CAP (1 << 20)

#ifdef _WIN32
#define BENCH_EXECUTION "native"
#define BENCH_EXECUTION_NOTE \
    "native Windows build (nvcc + MSVC) of pinned llama3.cuda with a Win32 POSIX shim (mmap/clock_gettime), not WSL"
#else
#define BENCH_EXECUTION "wsl"
#define BENCH_EXECUTION_NOTE "WSL build of pinned llama3.cuda"
#endif

static double time_now_s(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double) time.tv_sec + (double) time.tv_nsec * 1e-9;
}

static void json_escape(FILE *out, const char *s) {
    fputc('"', out);
    if (!s) {
        fputc('"', out);
        return;
    }
    for (; *s; s++) {
        unsigned char c = (unsigned char) *s;
        switch (c) {
            case '"': fputs("\\\"", out); break;
            case '\\': fputs("\\\\", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            case '\t': fputs("\\t", out); break;
            default:
                if (c < 0x20) fprintf(out, "\\u%04x", c);
                else fputc(c, out);
        }
    }
    fputc('"', out);
}

static void json_ints(FILE *out, const int *v, int n) {
    fputc('[', out);
    for (int i = 0; i < n; i++) {
        if (i) fputc(',', out);
        fprintf(out, "%d", v[i]);
    }
    fputc(']', out);
}

static void json_doubles(FILE *out, const double *v, int n) {
    fputc('[', out);
    for (int i = 0; i < n; i++) {
        if (i) fputc(',', out);
        fprintf(out, "%.9g", v[i]);
    }
    fputc(']', out);
}

typedef struct {
    const char *mode;
    const char *prompt;
    const char *checkpoint_path;
    const char *tokenizer_path;
    const char *sha256;
    int json;
    int max_new_tokens;
    int context_len;
    int decode_steps;
    int warmups;
    int reps;
} BenchArgs;

typedef struct {
    int pos;
    int n_prompt;
    int prompt[BENCH_MAX_STEPS];
    int n_generated;
    int generated[BENCH_MAX_STEPS];
    char text[BENCH_TEXT_CAP];
    int text_len;
    double prompt_s;
    double ttft_s;
    double compat_elapsed_s;
    int n_decode;
    double decode_step_s[BENCH_MAX_STEPS];
} BenchRun;

static void append_piece(BenchRun *run, const char *piece) {
    if (!piece) return;
    size_t n = strlen(piece);
    if (run->text_len + (int) n >= BENCH_TEXT_CAP - 1) return;
    memcpy(run->text + run->text_len, piece, n);
    run->text_len += (int) n;
    run->text[run->text_len] = 0;
}

static void append_printable(BenchRun *run, char *piece) {
    if (piece == NULL || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = (unsigned char) piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) return;
    }
    int xff = 0xff;
    unsigned char fbit = (piece[0] & xff);
    unsigned char sbit = (piece[1] & xff);
    unsigned char mask = 0x40;
    char tmp[8];
    switch (fbit) {
        case 0xC3:
            tmp[0] = (char) (sbit | mask);
            tmp[1] = 0;
            append_piece(run, tmp);
            break;
        case 0xC2:
            tmp[0] = (char) sbit;
            tmp[1] = 0;
            append_piece(run, tmp);
            break;
        default:
            append_piece(run, piece);
            break;
    }
}

static void pad_or_trim_tokens(int *tokens, int *n_tokens, int length) {
    if (length <= 0) return;
    if (*n_tokens >= length) {
        *n_tokens = length;
        return;
    }
    int filler = 13;
    for (int i = *n_tokens - 1; i >= 0; i--) {
        if (tokens[i] != 1) {
            filler = tokens[i];
            break;
        }
    }
    while (*n_tokens < length) {
        tokens[(*n_tokens)++] = filler;
    }
}

static void generate_bench(Transformer *transformer, Tokenizer *tokenizer, BenchArgs *args, BenchRun *run) {
    memset(run, 0, sizeof(*run));
    int cap = (int) strlen(args->prompt) + 8 + args->context_len + args->max_new_tokens;
    if (cap < 64) cap = 64;
    int *prompt_tokens = (int *) malloc((size_t) cap * sizeof(int));
    int num_prompt_tokens = 0;
    encode(tokenizer, (char *) args->prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens > 1 && prompt_tokens[1] == 306) prompt_tokens[1] = 76;
    if (strcmp(args->mode, "scaling") == 0 && args->context_len > 0) {
        pad_or_trim_tokens(prompt_tokens, &num_prompt_tokens, args->context_len);
    }
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }
    run->n_prompt = num_prompt_tokens;
    if (num_prompt_tokens > BENCH_MAX_STEPS) {
        fprintf(stderr, "prompt too long\n");
        exit(EXIT_FAILURE);
    }
    memcpy(run->prompt, prompt_tokens, (size_t) num_prompt_tokens * sizeof(int));

    int stop_on_bos = strcmp(args->mode, "compatibility") == 0;
    int token = prompt_tokens[0];
    int pos = 0;
    int first_forward_done = 0;
    double compat_start = 0.0;
    CUDA_CHECK(cudaDeviceSynchronize());
    double run_start = time_now_s();

    while (pos < args->max_new_tokens - 1) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = time_now_s();
        float *logits = forward(transformer, token, pos);
        CUDA_CHECK(cudaDeviceSynchronize());
        double t1 = time_now_s();

        int next;
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
            run->prompt_s += t1 - t0;
        } else {
            next = sample_argmax(logits, transformer->config.vocab_size);
            if (run->n_decode < BENCH_MAX_STEPS) {
                run->decode_step_s[run->n_decode] = t1 - t0;
                run->n_decode++;
            }
            if (run->n_decode == 1) {
                run->ttft_s = t1 - run_start;
            }
        }
        pos++;
        if (!first_forward_done) {
            compat_start = t1;
            first_forward_done = 1;
        }
        if (stop_on_bos && next == 1) break;

        if (run->n_generated < BENCH_MAX_STEPS) {
            run->generated[run->n_generated++] = next;
        }
        char *piece = decode(tokenizer, token, next);
        if (!args->json) {
            safe_printf(piece);
            fflush(stdout);
        } else {
            append_printable(run, piece);
        }
        token = next;
    }
    if (!args->json) printf("\n");
    run->pos = pos;
    run->compat_elapsed_s = first_forward_done ? (time_now_s() - compat_start) : 0.0;
    free(prompt_tokens);
}

static const char *DREAM_STORY =
    "I have a dream. He dreams of a big, beautiful garden full of flowers and trees. He dreams of playing with his friends and eating yummy snacks.\n"
    "One day, he was walking in the garden when he saw";

static void emit_json(BenchArgs *args, Transformer *t, BenchRun *run, double load_s, double warmup_s) {
    int match = 0;
    if (strcmp(args->mode, "compatibility") == 0) {
        match = strcmp(run->text, DREAM_STORY) == 0;
    } else {
        match = 1;
    }
    double decode_s = 0.0;
    for (int i = 0; i < run->n_decode; i++) decode_s += run->decode_step_s[i];
    double prompt_tok_s = (run->prompt_s > 0.0) ? ((double) (run->n_prompt - 1) / run->prompt_s) : 0.0;
    double decode_tok_s = (decode_s > 0.0) ? ((double) run->n_decode / decode_s) : 0.0;
    double legacy = (run->compat_elapsed_s > 0.0) ? ((double) (run->pos - 1) / run->compat_elapsed_s) : 0.0;
    int is_compat = strcmp(args->mode, "compatibility") == 0;
    int workload_decode_steps = is_compat ? (args->max_new_tokens - run->n_prompt) : args->decode_steps;

    fputs("{\"schema_version\":1,", stdout);
    fputs("\"engine\":\"llama3.cuda\",", stdout);
    fputs("\"execution\":\"" BENCH_EXECUTION "\",", stdout);
    fprintf(stdout, "\"checkpoint\":{\"path\":");
    json_escape(stdout, args->checkpoint_path);
    fprintf(stdout, ",\"sha256\":");
    json_escape(stdout, args->sha256 ? args->sha256 : "");
    fprintf(stdout,
            ",\"config\":{\"dim\":%d,\"hidden_dim\":%d,\"n_layers\":%d,\"n_heads\":%d,"
            "\"n_kv_heads\":%d,\"vocab_size\":%d,\"max_seq_len\":%d}}",
            t->config.dim, t->config.hidden_dim, t->config.n_layers, t->config.n_heads,
            t->config.n_kv_heads, t->config.vocab_size, t->config.max_seq_len);
    fprintf(stdout, ",\"workload\":{\"tier\":");
    json_escape(stdout, args->mode);
    fputs(",\"prompt\":", stdout);
    json_escape(stdout, args->prompt);
    fprintf(stdout,
            ",\"batch\":1,\"context_len\":%d,\"total_positions\":%d,\"decode_steps\":%d,\"sampling\":\"greedy\"}",
            args->context_len, args->max_new_tokens, workload_decode_steps);
    fputs(",\"precision\":{\"weights\":\"fp32\",\"activations\":\"fp32\",\"kv\":\"fp32\",\"tf32\":false}", stdout);
    fputs(",\"tokens\":{\"prompt\":", stdout);
    json_ints(stdout, run->prompt, run->n_prompt);
    fputs(",\"generated\":", stdout);
    json_ints(stdout, run->generated, run->n_generated);
    fputs(",\"text\":", stdout);
    json_escape(stdout, run->text);
    fprintf(stdout, ",\"match_expected\":%s}", match ? "true" : "false");
    fprintf(stdout,
            ",\"phases\":{\"load_s\":%.9g,\"warmup_s\":%.9g,\"prompt_s\":%.9g,\"ttft_s\":%.9g,\"decode_step_s\":",
            load_s, warmup_s, run->prompt_s, run->ttft_s);
    json_doubles(stdout, run->decode_step_s, run->n_decode);
    fprintf(stdout, ",\"compat_elapsed_s\":%.9g}", run->compat_elapsed_s);
    fprintf(stdout,
            ",\"metrics\":{\"prompt_tok_s\":%.9g,\"decode_tok_s\":%.9g,\"legacy_compat_tok_s\":%.9g}",
            prompt_tok_s, decode_tok_s, legacy);
    fputs(",\"engine_native_notes\":[\"" BENCH_EXECUTION_NOTE "\","
          "\"bench_tail.cu splice: CLOCK_MONOTONIC timing, cudaDeviceSynchronize around each measured forward\","
          "\"load_s includes tokenizer load and cuBLAS handle creation\"",
          stdout);
    if (!is_compat) fputs(",\"scaling pads/trims prompt tokens to context_len; filler is last non-BOS id\"", stdout);
    fputs("]", stdout);
    fputs(",\"build\":{\"opt\":\"-O3\",\"cublas\":true}", stdout);
    fputs(",\"replay_stats\":null}\n", stdout);
    fflush(stdout);
    if (strcmp(args->mode, "compatibility") == 0 && !match) {
        fprintf(stderr, "token mismatch\n--- got ---\n%s\n--- expected ---\n%s\n", run->text, DREAM_STORY);
        exit(EXIT_FAILURE);
    }
}

static void print_usage(void) {
    fprintf(stderr,
            "Usage: runcuda [--checkpoint PATH] [--tokenizer PATH] [-n N] [--context N]\n"
            "               [--mode compatibility|scaling] [--decode-steps N]\n"
            "               [--json] [--warmups N] [--reps N] [--sha256 HEX] [prompt]\n");
}

int main(int argc, char *argv[]) {
    BenchArgs args;
    memset(&args, 0, sizeof(args));
    args.checkpoint_path = "stories15M.bin";
    args.tokenizer_path = "tokenizer.bin";
    args.max_new_tokens = 50;
    args.context_len = 50;
    args.decode_steps = 16;
    args.warmups = 0;
    args.reps = 1;
    args.mode = "compatibility";
    args.prompt = "I have a dream";
    args.sha256 = "";

    for (int i = 1; i < argc; i++) {
        char *a = argv[i];
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            print_usage();
            return 0;
        } else if (strcmp(a, "--checkpoint") == 0 && i + 1 < argc) {
            args.checkpoint_path = argv[++i];
        } else if ((strcmp(a, "--tokenizer") == 0 || strcmp(a, "-z") == 0) && i + 1 < argc) {
            args.tokenizer_path = argv[++i];
        } else if ((strcmp(a, "-n") == 0 || strcmp(a, "--n") == 0) && i + 1 < argc) {
            args.max_new_tokens = atoi(argv[++i]);
        } else if (strcmp(a, "--context") == 0 && i + 1 < argc) {
            args.context_len = atoi(argv[++i]);
        } else if (strcmp(a, "--mode") == 0 && i + 1 < argc) {
            args.mode = argv[++i];
        } else if (strcmp(a, "--decode-steps") == 0 && i + 1 < argc) {
            args.decode_steps = atoi(argv[++i]);
        } else if (strcmp(a, "--json") == 0) {
            args.json = 1;
        } else if (strcmp(a, "--warmups") == 0 && i + 1 < argc) {
            args.warmups = atoi(argv[++i]);
        } else if (strcmp(a, "--reps") == 0 && i + 1 < argc) {
            args.reps = atoi(argv[++i]);
        } else if (strcmp(a, "--sha256") == 0 && i + 1 < argc) {
            args.sha256 = argv[++i];
        } else if (a[0] == '-') {
            fprintf(stderr, "unknown flag %s\n", a);
            print_usage();
            return 1;
        } else {
            args.prompt = a;
        }
    }

    int scaling = strcmp(args.mode, "scaling") == 0;
    if (!scaling && strcmp(args.mode, "compatibility") != 0) {
        fprintf(stderr, "unknown mode %s\n", args.mode);
        return 1;
    }

    double t_load0 = time_now_s();
    Transformer transformer;
    build_transformer(&transformer, (char *) args.checkpoint_path);
    int max_seq_len = transformer.config.max_seq_len;
    if (scaling) {
        // Context is min(requested, max_seq_len); the loop runs forwards at 0 .. max_new_tokens-2,
        // so max_seq_len + 1 still keeps every KV row inside the cache.
        if (args.context_len <= 0) args.context_len = 8;
        if (args.context_len > max_seq_len) args.context_len = max_seq_len;
        args.max_new_tokens = args.context_len + args.decode_steps;
        if (args.max_new_tokens > max_seq_len + 1) args.max_new_tokens = max_seq_len + 1;
    } else {
        if (args.max_new_tokens > max_seq_len) args.max_new_tokens = max_seq_len;
        args.context_len = args.max_new_tokens;
    }
    if (args.max_new_tokens > BENCH_MAX_STEPS) {
        fprintf(stderr, "total positions %d exceed BENCH_MAX_STEPS %d\n", args.max_new_tokens, BENCH_MAX_STEPS);
        return 1;
    }
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, (char *) args.tokenizer_path, transformer.config.vocab_size);
    create_cublas_handle();
    CUDA_CHECK(cudaDeviceSynchronize());
    double load_s = time_now_s() - t_load0;

    // ~1.1 MB: too large for the default 1 MB Windows main-thread stack.
    static BenchRun run;
    double warmup_s = 0.0;
    int saved_json = args.json;
    for (int w = 0; w < args.warmups; w++) {
        args.json = 1;
        double t0 = time_now_s();
        generate_bench(&transformer, &tokenizer, &args, &run);
        CUDA_CHECK(cudaDeviceSynchronize());
        warmup_s += time_now_s() - t0;
    }
    args.json = saved_json;
    if (args.reps < 1) args.reps = 1;
    for (int r = 0; r < args.reps; r++) {
        generate_bench(&transformer, &tokenizer, &args, &run);
        if (args.json) {
            emit_json(&args, &transformer, &run, load_s, warmup_s);
        } else if (run.pos > 1) {
            fprintf(stderr, "Token count: %d, elapsed: %fs, %d tokens/s\n",
                    run.pos + 1, (float) run.compat_elapsed_s,
                    (int) ((run.pos - 1) / run.compat_elapsed_s));
        }
    }

    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    destroy_cublas_handle();
    return 0;
}
