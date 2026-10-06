// CPU↔GPU synchronisation latencies on the default Metal device.
//
// Each measurement runs a tiny kernel after the GPU has idled for `--idle-us`
// (a decode step leaves the GPU idle a few hundred microseconds between tokens)
// and reports the median over `--iters` repetitions, in microseconds:
//
//   launch           commit → GPUStartTime
//   wait_return      GPUEndTime → waitUntilCompleted returns
//   handler          GPUEndTime → completed handler runs
//   event_poll       GPUEndTime → MTLSharedEvent.signaledValue observed (spin)
//   flag_poll        commit → flag written by the kernel observed (spin), and
//                    the same against GPUEndTime (negative: seen before completion)
//   fresh_e2e        commit → flag observed: a cold submit's whole round trip
//   gated_e2e        CPU signals a shared event → flag observed, for a command
//                    buffer committed earlier and waiting on that event
//   cb_gap           GPU idle between two command buffers committed together
//   encoder_gap      GPU time of one CB with two encoders minus one encoder
//   dispatch_gap     per extra dependent dispatch in one serial encoder
//   dispatch_gap_conc            the same in a concurrent encoder, no barriers
//   dispatch_gap_conc_barrier    concurrent, a buffer-scope barrier before each
//   dispatch_gap_serial_barrier  serial, a resource barrier before each
//
//   clang -fobjc-arc -O2 -framework Metal -framework Foundation \
//       tools/sync_latency/main.m -o /tmp/sync_latency && /tmp/sync_latency

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdatomic.h>

static NSString *const kSource = @"#include <metal_stdlib>\n"
                                  "using namespace metal;\n"
                                  "kernel void tick(device atomic_uint *flag [[buffer(0)]],\n"
                                  "                 device float *scratch [[buffer(1)]],\n"
                                  "                 constant uint &value [[buffer(2)]],\n"
                                  "                 uint tid [[thread_position_in_grid]]) {\n"
                                  "  scratch[tid] = scratch[tid] * 0.5f + 1.0f;\n"
                                  "  if (tid == 0) atomic_store_explicit(flag, value, memory_order_relaxed);\n"
                                  "}\n";

static double now_us(void) { return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e3; }
static double host_us(CFTimeInterval s) { return s * 1e6; }

static void idle(double us) {
    double until = now_us() + us;
    while (now_us() < until) {
    }
}

static double median(NSMutableArray<NSNumber *> *v) {
    NSArray *s = [v sortedArrayUsingSelector:@selector(compare:)];
    return [s[s.count / 2] doubleValue];
}

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;
static id<MTLComputePipelineState> g_pso;
static id<MTLBuffer> g_flag, g_scratch;
static uint32_t g_value;

static void encode_tick(id<MTLComputeCommandEncoder> e, uint32_t value) {
    [e setComputePipelineState:g_pso];
    [e setBuffer:g_flag offset:0 atIndex:0];
    [e setBuffer:g_scratch offset:0 atIndex:1];
    [e setBytes:&value length:4 atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static id<MTLCommandBuffer> tick_cb(uint32_t value, int dispatches, int encoders) {
    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    for (int en = 0; en < encoders; en++) {
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        for (int i = 0; i < dispatches; i++) encode_tick(e, value);
        [e endEncoding];
    }
    return cb;
}

static volatile uint32_t *flag_ptr(void) { return (volatile uint32_t *)g_flag.contents; }

static void spin_flag(uint32_t value) {
    while (atomic_load_explicit((_Atomic uint32_t *)flag_ptr(), memory_order_acquire) != value) {
    }
}

int main(int argc, const char **argv) {
    int iters = 200;
    double idle_us = 400;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--iters")) iters = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "--idle-us")) idle_us = atof(argv[i + 1]);
    }
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        g_q = [g_dev newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:kSource options:nil error:&err];
        if (!lib) {
            NSLog(@"%@", err);
            return 1;
        }
        g_pso = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"tick"] error:&err];
        g_flag = [g_dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        g_scratch = [g_dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        id<MTLSharedEvent> ev = [g_dev newSharedEvent];
        uint64_t ev_value = 0;
        MTLSharedEventListener *listener =
            [[MTLSharedEventListener alloc] initWithDispatchQueue:dispatch_queue_create("listener", NULL)];
        printf("device %s, iters %d, idle %.0f us\n", g_dev.name.UTF8String, iters, idle_us);

        NSMutableDictionary<NSString *, NSMutableArray *> *r = [NSMutableDictionary dictionary];
        NSArray *keys = @[
            @"launch", @"wait_return", @"handler", @"event_poll", @"event_wait", @"listener", @"event_nap_poll",
            @"flag_poll_vs_commit", @"flag_poll_vs_gpu_end",
            @"fresh_e2e", @"gated_e2e", @"cb_gap", @"encoder_gap", @"dispatch_gap", @"dispatch_gap_conc",
            @"dispatch_gap_conc_barrier", @"dispatch_gap_serial_barrier"
        ];
        for (NSString *k in keys) r[k] = [NSMutableArray array];

        for (int it = 0; it < iters + 10; it++) {
            BOOL keep = it >= 10;
            @autoreleasepool {
                // launch + wait_return
                idle(idle_us);
                id<MTLCommandBuffer> cb = tick_cb(++g_value, 1, 1);
                double t_commit = now_us();
                [cb commit];
                [cb waitUntilCompleted];
                double t_ret = now_us();
                if (keep) {
                    [r[@"launch"] addObject:@(host_us(cb.GPUStartTime) - t_commit)];
                    [r[@"wait_return"] addObject:@(t_ret - host_us(cb.GPUEndTime))];
                }

                // completed handler
                idle(idle_us);
                cb = tick_cb(++g_value, 1, 1);
                __block _Atomic double t_handler = 0;
                [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
                  atomic_store(&t_handler, now_us());
                }];
                [cb commit];
                while (atomic_load(&t_handler) == 0) {
                }
                if (keep) [r[@"handler"] addObject:@(atomic_load(&t_handler) - host_us(cb.GPUEndTime))];

                // shared event poll
                idle(idle_us);
                cb = tick_cb(++g_value, 1, 1);
                [cb encodeSignalEvent:ev value:++ev_value];
                [cb commit];
                while (ev.signaledValue < ev_value) {
                }
                double t_ev = now_us();
                [cb waitUntilCompleted];
                if (keep) [r[@"event_poll"] addObject:@(t_ev - host_us(cb.GPUEndTime))];

                // shared event: blocking wait on the event itself
                if ([ev respondsToSelector:@selector(waitUntilSignaledValue:timeoutMS:)]) {
                    idle(idle_us);
                    cb = tick_cb(++g_value, 1, 1);
                    [cb encodeSignalEvent:ev value:++ev_value];
                    [cb commit];
                    [ev waitUntilSignaledValue:ev_value timeoutMS:1000];
                    double t_evw = now_us();
                    [cb waitUntilCompleted];
                    if (keep) [r[@"event_wait"] addObject:@(t_evw - host_us(cb.GPUEndTime))];
                }

                // shared event listener callback
                idle(idle_us);
                cb = tick_cb(++g_value, 1, 1);
                [cb encodeSignalEvent:ev value:++ev_value];
                __block _Atomic double t_listen = 0;
                [ev notifyListener:listener
                           atValue:ev_value
                             block:^(id<MTLSharedEvent> e, uint64_t value) {
                               atomic_store(&t_listen, now_us());
                             }];
                [cb commit];
                while (atomic_load(&t_listen) == 0) {
                }
                [cb waitUntilCompleted];
                if (keep) [r[@"listener"] addObject:@(atomic_load(&t_listen) - host_us(cb.GPUEndTime))];

                // shared event polled between short sleeps
                idle(idle_us);
                cb = tick_cb(++g_value, 1, 1);
                [cb encodeSignalEvent:ev value:++ev_value];
                [cb commit];
                int naps = 0;
                while (ev.signaledValue < ev_value) {
                    usleep(20);
                    naps++;
                }
                double t_nap = now_us();
                [cb waitUntilCompleted];
                if (keep) [r[@"event_nap_poll"] addObject:@(t_nap - host_us(cb.GPUEndTime))];

                // flag poll (fresh submit end to end)
                idle(idle_us);
                uint32_t v = ++g_value;
                cb = tick_cb(v, 1, 1);
                t_commit = now_us();
                [cb commit];
                spin_flag(v);
                double t_flag = now_us();
                [cb waitUntilCompleted];
                if (keep) {
                    [r[@"flag_poll_vs_commit"] addObject:@(t_flag - t_commit)];
                    [r[@"flag_poll_vs_gpu_end"] addObject:@(t_flag - host_us(cb.GPUEndTime))];
                    [r[@"fresh_e2e"] addObject:@(t_flag - t_commit)];
                }

                // gated: committed early, released by a CPU-side event signal
                v = ++g_value;
                cb = [g_q commandBuffer];
                uint64_t gate = ++ev_value;
                [cb encodeWaitForEvent:ev value:gate];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                encode_tick(e, v);
                [e endEncoding];
                [cb commit];
                idle(idle_us);
                double t_sig = now_us();
                ev.signaledValue = gate;
                spin_flag(v);
                double t_gflag = now_us();
                [cb waitUntilCompleted];
                if (keep) [r[@"gated_e2e"] addObject:@(t_gflag - t_sig)];

                // gap between two CBs committed back to back
                idle(idle_us);
                id<MTLCommandBuffer> a = tick_cb(++g_value, 1, 1), b = tick_cb(++g_value, 1, 1);
                [a commit];
                [b commit];
                [b waitUntilCompleted];
                if (keep) [r[@"cb_gap"] addObject:@(host_us(b.GPUStartTime) - host_us(a.GPUEndTime))];

                // encoder and dispatch costs inside one CB
                idle(idle_us);
                id<MTLCommandBuffer> one = tick_cb(++g_value, 16, 1);
                [one commit];
                [one waitUntilCompleted];
                idle(idle_us);
                id<MTLCommandBuffer> two = tick_cb(++g_value, 8, 2);
                [two commit];
                [two waitUntilCompleted];
                idle(idle_us);
                id<MTLCommandBuffer> single = tick_cb(++g_value, 1, 1);
                [single commit];
                [single waitUntilCompleted];
                double d16 = host_us(one.GPUEndTime) - host_us(one.GPUStartTime);
                double d2e = host_us(two.GPUEndTime) - host_us(two.GPUStartTime);
                double d1 = host_us(single.GPUEndTime) - host_us(single.GPUStartTime);
                if (keep) {
                    [r[@"encoder_gap"] addObject:@(d2e - d16)];
                    [r[@"dispatch_gap"] addObject:@((d16 - d1) / 15.0)];
                }

                // the same 16 dispatches in a concurrent encoder, with and without a
                // buffer-scope barrier between each
                NSArray *variants =
                    @[ @"dispatch_gap_conc", @"dispatch_gap_conc_barrier", @"dispatch_gap_serial_barrier" ];
                for (NSUInteger variant = 0; variant < variants.count; variant++) {
                    idle(idle_us);
                    id<MTLCommandBuffer> vcb = [g_q commandBuffer];
                    id<MTLComputeCommandEncoder> ve =
                        variant == 2 ? [vcb computeCommandEncoder]
                                     : [vcb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    for (int i = 0; i < 16; i++) {
                        if (variant == 1 && i) [ve memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        if (variant == 2 && i) {
                            id<MTLResource> res[1] = {g_scratch};
                            [ve memoryBarrierWithResources:res count:1];
                        }
                        encode_tick(ve, ++g_value);
                    }
                    [ve endEncoding];
                    [vcb commit];
                    [vcb waitUntilCompleted];
                    double dv = host_us(vcb.GPUEndTime) - host_us(vcb.GPUStartTime);
                    if (keep) [r[variants[variant]] addObject:@((dv - d1) / 15.0)];
                }
            }
        }
        for (NSString *k in keys) printf("%22s %8.1f us\n", k.UTF8String, median(r[k]));
    }
    return 0;
}
