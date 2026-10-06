// Metal API call log for any process, injected with DYLD_INSERT_LIBRARIES.
//
// Hooks the concrete command queue, command buffer, compute/blit encoder and device
// classes and writes one line per call, in call order, to $METAL_API_TRACE:
//
//   <seq> t<thread> <object> <call> <args> @<us>
//
// Command buffers (cb), encoders (ce/be), buffers (b) and pipelines are named by
// creation order; pipelines by their function name or descriptor label. `@<us>` is
// host time in microseconds since injection. Every committed command buffer also
// logs `cbN gpu start=<us> end=<us>` on completion, on the same clock.
//
//   clang -dynamiclib -fobjc-arc -O2 -framework Metal -framework Foundation \
//       tools/metal_api_trace/trace.m -o /tmp/metal_api_trace.dylib
//   METAL_API_TRACE=/tmp/trace.txt DYLD_INSERT_LIBRARIES=/tmp/metal_api_trace.dylib <cmd>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <pthread.h>

static FILE *g_out;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long g_seq;
static unsigned long g_next_cb, g_next_enc, g_next_buf, g_next_thread;
static char g_name_key, g_thread_key_init;
static pthread_key_t g_thread_key;
static uint64_t g_t0_ns;

static double now_us(void) { return (double)(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - g_t0_ns) / 1e3; }

/// Host-time seconds (`GPUStartTime` / `GPUEndTime`) on the `now_us` clock.
static double host_s_to_us(CFTimeInterval s) { return s * 1e6 - (double)g_t0_ns / 1e3; }

static unsigned long thread_index(void) {
    uintptr_t v = (uintptr_t)pthread_getspecific(g_thread_key);
    if (v == 0) {
        v = ++g_next_thread;
        pthread_setspecific(g_thread_key, (void *)v);
    }
    return (unsigned long)v;
}

static void emit(NSString *object, const char *call, NSString *args) {
    if (!g_out) return;
    pthread_mutex_lock(&g_lock);
    fprintf(g_out, "%llu t%lu %s %s %s @%.1f\n", ++g_seq, thread_index(), object.UTF8String, call,
            args ? args.UTF8String : "", now_us());
    pthread_mutex_unlock(&g_lock);
}

static NSString *name_of(id obj) {
    if (!obj) return @"nil";
    return objc_getAssociatedObject(obj, &g_name_key) ?: [NSString stringWithFormat:@"%p", obj];
}

static void set_name(id obj, NSString *name) {
    if (obj) objc_setAssociatedObject(obj, &g_name_key, name, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

static const char *storage(MTLResourceOptions o) {
    switch ((o & MTLResourceStorageModeMask) >> MTLResourceStorageModeShift) {
        case MTLStorageModeShared: return "shared";
        case MTLStorageModeManaged: return "managed";
        case MTLStorageModePrivate: return "private";
        default: return "memoryless";
    }
}

/// Buffers are named on first use, so heap and device allocations look alike.
static NSString *buf(id<MTLBuffer> b) {
    if (!b) return @"nil";
    NSString *n = objc_getAssociatedObject(b, &g_name_key);
    if (n) return n;
    pthread_mutex_lock(&g_lock);
    n = [NSString stringWithFormat:@"b%lu", ++g_next_buf];
    pthread_mutex_unlock(&g_lock);
    set_name(b, n);
    const char *hazard = b.hazardTrackingMode == MTLHazardTrackingModeUntracked ? "untracked" : "tracked";
    emit(n, "first_use", [NSString stringWithFormat:@"len=%lu %s %s heap=%d label=%@", (unsigned long)b.length,
                                                     storage(b.resourceOptions), hazard, b.heap != nil, b.label]);
    return n;
}

static NSString *size3(MTLSize s) {
    return [NSString stringWithFormat:@"%lux%lux%lu", (unsigned long)s.width, (unsigned long)s.height,
                                      (unsigned long)s.depth];
}

#define ORIG(name) static IMP orig_##name
#define CALL(name, type, ...) ((type)orig_##name)(self, _cmd, ##__VA_ARGS__)

// --- command queue ---------------------------------------------------------

static id<MTLCommandBuffer> name_cb(id<MTLCommandBuffer> cb, const char *how) {
    pthread_mutex_lock(&g_lock);
    NSString *n = [NSString stringWithFormat:@"cb%lu", ++g_next_cb];
    pthread_mutex_unlock(&g_lock);
    set_name(cb, n);
    emit(n, how, nil);
    return cb;
}

ORIG(commandBuffer);
static id hook_commandBuffer(id self, SEL _cmd) {
    return name_cb(CALL(commandBuffer, id (*)(id, SEL)), "queue.commandBuffer");
}
ORIG(commandBufferUnretained);
static id hook_commandBufferUnretained(id self, SEL _cmd) {
    return name_cb(CALL(commandBufferUnretained, id (*)(id, SEL)), "queue.commandBufferWithUnretainedReferences");
}
ORIG(commandBufferWithDescriptor);
static id hook_commandBufferWithDescriptor(id self, SEL _cmd, MTLCommandBufferDescriptor *d) {
    return name_cb(CALL(commandBufferWithDescriptor, id (*)(id, SEL, id), d), "queue.commandBufferWithDescriptor");
}

// --- command buffer --------------------------------------------------------

static id name_enc(id enc, id cb, const char *kind, const char *how) {
    pthread_mutex_lock(&g_lock);
    NSString *n = [NSString stringWithFormat:@"%s%lu", kind, ++g_next_enc];
    pthread_mutex_unlock(&g_lock);
    set_name(enc, n);
    emit(name_of(cb), how, n);
    return enc;
}

ORIG(computeEncoder);
static id hook_computeEncoder(id self, SEL _cmd) {
    return name_enc(CALL(computeEncoder, id (*)(id, SEL)), self, "ce", "computeCommandEncoder");
}
ORIG(computeEncoderDispatchType);
static id hook_computeEncoderDispatchType(id self, SEL _cmd, MTLDispatchType t) {
    id enc = CALL(computeEncoderDispatchType, id (*)(id, SEL, MTLDispatchType), t);
    return name_enc(enc, self, "ce",
                    t == MTLDispatchTypeConcurrent ? "computeCommandEncoderWithDispatchType:concurrent"
                                                   : "computeCommandEncoderWithDispatchType:serial");
}
ORIG(computeEncoderDescriptor);
static id hook_computeEncoderDescriptor(id self, SEL _cmd, MTLComputePassDescriptor *d) {
    id enc = CALL(computeEncoderDescriptor, id (*)(id, SEL, id), d);
    return name_enc(enc, self, "ce",
                    d.dispatchType == MTLDispatchTypeConcurrent ? "computeCommandEncoderWithDescriptor:concurrent"
                                                                : "computeCommandEncoderWithDescriptor:serial");
}
ORIG(blitEncoder);
static id hook_blitEncoder(id self, SEL _cmd) {
    return name_enc(CALL(blitEncoder, id (*)(id, SEL)), self, "be", "blitCommandEncoder");
}
ORIG(addCompletedHandler);
ORIG(commit);
static void hook_commit(id self, SEL _cmd) {
    NSString *n = name_of(self);
    ((void (*)(id, SEL, id))orig_addCompletedHandler)(self, @selector(addCompletedHandler:), ^(id<MTLCommandBuffer> cb) {
      emit(n, "gpu", [NSString stringWithFormat:@"start=%.1f end=%.1f", host_s_to_us(cb.GPUStartTime),
                                                host_s_to_us(cb.GPUEndTime)]);
    });
    emit(n, "commit", nil);
    CALL(commit, void (*)(id, SEL));
}
ORIG(enqueue);
static void hook_enqueue(id self, SEL _cmd) {
    emit(name_of(self), "enqueue", nil);
    CALL(enqueue, void (*)(id, SEL));
}
ORIG(waitUntilCompleted);
static void hook_waitUntilCompleted(id self, SEL _cmd) {
    emit(name_of(self), "waitUntilCompleted", nil);
    CALL(waitUntilCompleted, void (*)(id, SEL));
    emit(name_of(self), "waitUntilCompleted.return", nil);
}
ORIG(waitUntilScheduled);
static void hook_waitUntilScheduled(id self, SEL _cmd) {
    emit(name_of(self), "waitUntilScheduled", nil);
    CALL(waitUntilScheduled, void (*)(id, SEL));
}
static void hook_addCompletedHandler(id self, SEL _cmd, id block) {
    emit(name_of(self), "addCompletedHandler", nil);
    CALL(addCompletedHandler, void (*)(id, SEL, id), block);
}
ORIG(addScheduledHandler);
static void hook_addScheduledHandler(id self, SEL _cmd, id block) {
    emit(name_of(self), "addScheduledHandler", nil);
    CALL(addScheduledHandler, void (*)(id, SEL, id), block);
}
ORIG(encodeSignalEvent);
static void hook_encodeSignalEvent(id self, SEL _cmd, id ev, uint64_t v) {
    emit(name_of(self), "encodeSignalEvent", [NSString stringWithFormat:@"value=%llu", v]);
    CALL(encodeSignalEvent, void (*)(id, SEL, id, uint64_t), ev, v);
}
ORIG(encodeWaitForEvent);
static void hook_encodeWaitForEvent(id self, SEL _cmd, id ev, uint64_t v) {
    emit(name_of(self), "encodeWaitForEvent", [NSString stringWithFormat:@"value=%llu", v]);
    CALL(encodeWaitForEvent, void (*)(id, SEL, id, uint64_t), ev, v);
}

// --- compute encoder -------------------------------------------------------

ORIG(setPSO);
static void hook_setPSO(id self, SEL _cmd, id pso) {
    emit(name_of(self), "setComputePipelineState", name_of(pso));
    CALL(setPSO, void (*)(id, SEL, id), pso);
}
ORIG(setBuffer);
static void hook_setBuffer(id self, SEL _cmd, id b, NSUInteger off, NSUInteger idx) {
    emit(name_of(self), "setBuffer", [NSString stringWithFormat:@"%@ off=%lu idx=%lu", buf(b), off, idx]);
    CALL(setBuffer, void (*)(id, SEL, id, NSUInteger, NSUInteger), b, off, idx);
}
ORIG(setBufferOffset);
static void hook_setBufferOffset(id self, SEL _cmd, NSUInteger off, NSUInteger idx) {
    emit(name_of(self), "setBufferOffset", [NSString stringWithFormat:@"off=%lu idx=%lu", off, idx]);
    CALL(setBufferOffset, void (*)(id, SEL, NSUInteger, NSUInteger), off, idx);
}
ORIG(setBuffers);
static void hook_setBuffers(id self, SEL _cmd, const id __unsafe_unretained *bs, const NSUInteger *offs, NSRange r) {
    NSMutableString *s = [NSMutableString stringWithFormat:@"idx=%lu..%lu", r.location, r.location + r.length];
    for (NSUInteger i = 0; i < r.length; i++) [s appendFormat:@" %@+%lu", buf(bs[i]), offs[i]];
    emit(name_of(self), "setBuffers", s);
    CALL(setBuffers, void (*)(id, SEL, const id __unsafe_unretained *, const NSUInteger *, NSRange), bs, offs, r);
}
ORIG(setBytes);
static void hook_setBytes(id self, SEL _cmd, const void *p, NSUInteger len, NSUInteger idx) {
    emit(name_of(self), "setBytes", [NSString stringWithFormat:@"len=%lu idx=%lu", len, idx]);
    CALL(setBytes, void (*)(id, SEL, const void *, NSUInteger, NSUInteger), p, len, idx);
}
ORIG(setTGMem);
static void hook_setTGMem(id self, SEL _cmd, NSUInteger len, NSUInteger idx) {
    emit(name_of(self), "setThreadgroupMemoryLength", [NSString stringWithFormat:@"len=%lu idx=%lu", len, idx]);
    CALL(setTGMem, void (*)(id, SEL, NSUInteger, NSUInteger), len, idx);
}
ORIG(dispatchTG);
static void hook_dispatchTG(id self, SEL _cmd, MTLSize g, MTLSize t) {
    emit(name_of(self), "dispatchThreadgroups", [NSString stringWithFormat:@"grid=%@ tg=%@", size3(g), size3(t)]);
    CALL(dispatchTG, void (*)(id, SEL, MTLSize, MTLSize), g, t);
}
ORIG(dispatchThreads);
static void hook_dispatchThreads(id self, SEL _cmd, MTLSize g, MTLSize t) {
    emit(name_of(self), "dispatchThreads", [NSString stringWithFormat:@"threads=%@ tg=%@", size3(g), size3(t)]);
    CALL(dispatchThreads, void (*)(id, SEL, MTLSize, MTLSize), g, t);
}
ORIG(dispatchIndirect);
static void hook_dispatchIndirect(id self, SEL _cmd, id b, NSUInteger off, MTLSize t) {
    emit(name_of(self), "dispatchThreadgroupsWithIndirectBuffer",
         [NSString stringWithFormat:@"%@ off=%lu tg=%@", buf(b), off, size3(t)]);
    CALL(dispatchIndirect, void (*)(id, SEL, id, NSUInteger, MTLSize), b, off, t);
}
ORIG(barrierScope);
static void hook_barrierScope(id self, SEL _cmd, MTLBarrierScope s) {
    emit(name_of(self), "memoryBarrierWithScope", [NSString stringWithFormat:@"scope=%lu", (unsigned long)s]);
    CALL(barrierScope, void (*)(id, SEL, MTLBarrierScope), s);
}
ORIG(barrierResources);
static void hook_barrierResources(id self, SEL _cmd, const id __unsafe_unretained *rs, NSUInteger n) {
    NSMutableString *s = [NSMutableString stringWithFormat:@"count=%lu", n];
    for (NSUInteger i = 0; i < n && i < 8; i++)
        [s appendFormat:@" %@", [rs[i] conformsToProtocol:@protocol(MTLBuffer)] ? buf(rs[i]) : name_of(rs[i])];
    emit(name_of(self), "memoryBarrierWithResources", s);
    CALL(barrierResources, void (*)(id, SEL, const id __unsafe_unretained *, NSUInteger), rs, n);
}
ORIG(useResource);
static void hook_useResource(id self, SEL _cmd, id r, MTLResourceUsage u) {
    emit(name_of(self), "useResource",
         [NSString stringWithFormat:@"%@ usage=%lu", [r conformsToProtocol:@protocol(MTLBuffer)] ? buf(r) : name_of(r),
                                    (unsigned long)u]);
    CALL(useResource, void (*)(id, SEL, id, MTLResourceUsage), r, u);
}
ORIG(useResources);
static void hook_useResources(id self, SEL _cmd, const id __unsafe_unretained *rs, NSUInteger n, MTLResourceUsage u) {
    emit(name_of(self), "useResources", [NSString stringWithFormat:@"count=%lu usage=%lu", n, (unsigned long)u]);
    CALL(useResources, void (*)(id, SEL, const id __unsafe_unretained *, NSUInteger, MTLResourceUsage), rs, n, u);
}
ORIG(useHeap);
static void hook_useHeap(id self, SEL _cmd, id h) {
    emit(name_of(self), "useHeap", nil);
    CALL(useHeap, void (*)(id, SEL, id), h);
}
ORIG(useHeaps);
static void hook_useHeaps(id self, SEL _cmd, const id __unsafe_unretained *hs, NSUInteger n) {
    emit(name_of(self), "useHeaps", [NSString stringWithFormat:@"count=%lu", n]);
    CALL(useHeaps, void (*)(id, SEL, const id __unsafe_unretained *, NSUInteger), hs, n);
}
ORIG(pushDebugGroup);
static void hook_pushDebugGroup(id self, SEL _cmd, NSString *s) {
    emit(name_of(self), "pushDebugGroup", s);
    CALL(pushDebugGroup, void (*)(id, SEL, id), s);
}
ORIG(popDebugGroup);
static void hook_popDebugGroup(id self, SEL _cmd) {
    emit(name_of(self), "popDebugGroup", nil);
    CALL(popDebugGroup, void (*)(id, SEL));
}
ORIG(computeEnd);
static void hook_computeEnd(id self, SEL _cmd) {
    emit(name_of(self), "endEncoding", nil);
    CALL(computeEnd, void (*)(id, SEL));
}
ORIG(computeUpdateFence);
static void hook_computeUpdateFence(id self, SEL _cmd, id f) {
    emit(name_of(self), "updateFence", nil);
    CALL(computeUpdateFence, void (*)(id, SEL, id), f);
}
ORIG(computeWaitFence);
static void hook_computeWaitFence(id self, SEL _cmd, id f) {
    emit(name_of(self), "waitForFence", nil);
    CALL(computeWaitFence, void (*)(id, SEL, id), f);
}

// --- blit encoder ----------------------------------------------------------

ORIG(copyBuffer);
static void hook_copyBuffer(id self, SEL _cmd, id s, NSUInteger so, id d, NSUInteger dO, NSUInteger n) {
    emit(name_of(self), "copyFromBuffer",
         [NSString stringWithFormat:@"%@+%lu -> %@+%lu size=%lu", buf(s), so, buf(d), dO, n]);
    CALL(copyBuffer, void (*)(id, SEL, id, NSUInteger, id, NSUInteger, NSUInteger), s, so, d, dO, n);
}
ORIG(fillBuffer);
static void hook_fillBuffer(id self, SEL _cmd, id b, NSRange r, uint8_t v) {
    emit(name_of(self), "fillBuffer", [NSString stringWithFormat:@"%@ +%lu size=%lu", buf(b), r.location, r.length]);
    CALL(fillBuffer, void (*)(id, SEL, id, NSRange, uint8_t), b, r, v);
}
ORIG(blitEnd);
static void hook_blitEnd(id self, SEL _cmd) {
    emit(name_of(self), "endEncoding", nil);
    CALL(blitEnd, void (*)(id, SEL));
}

// --- device ----------------------------------------------------------------

ORIG(psoFunction);
static id hook_psoFunction(id self, SEL _cmd, id<MTLFunction> f, NSError **e) {
    id pso = CALL(psoFunction, id (*)(id, SEL, id, NSError **), f, e);
    set_name(pso, f.label ?: f.name);
    emit(@"device", "newComputePipelineStateWithFunction", name_of(pso));
    return pso;
}
ORIG(psoFunctionOptions);
static id hook_psoFunctionOptions(id self, SEL _cmd, id<MTLFunction> f, MTLPipelineOption o, id *r, NSError **e) {
    id pso = CALL(psoFunctionOptions, id (*)(id, SEL, id, MTLPipelineOption, id *, NSError **), f, o, r, e);
    set_name(pso, f.label ?: f.name);
    emit(@"device", "newComputePipelineStateWithFunction:options", name_of(pso));
    return pso;
}
ORIG(psoDescriptor);
static id hook_psoDescriptor(id self, SEL _cmd, MTLComputePipelineDescriptor *d, MTLPipelineOption o, id *r,
                             NSError **e) {
    id pso = CALL(psoDescriptor, id (*)(id, SEL, id, MTLPipelineOption, id *, NSError **), d, o, r, e);
    set_name(pso, d.label ?: d.computeFunction.name);
    emit(@"device", "newComputePipelineStateWithDescriptor", name_of(pso));
    return pso;
}

static void hook(Class cls, SEL sel, IMP imp, IMP *orig) {
    Method m = class_getInstanceMethod(cls, sel);
    if (!m || *orig) return;
    *orig = method_getImplementation(m);
    class_replaceMethod(cls, sel, imp, method_getTypeEncoding(m));
}

__attribute__((constructor)) static void install(void) {
    const char *path = getenv("METAL_API_TRACE");
    if (!path) return;
    g_t0_ns = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    g_out = fopen(path, "w");
    if (!g_out) return;
    setvbuf(g_out, NULL, _IOFBF, 1 << 20);
    if (!g_thread_key_init) {
        pthread_key_create(&g_thread_key, NULL);
        g_thread_key_init = 1;
    }
    atexit_b(^{
      pthread_mutex_lock(&g_lock);
      fflush(g_out);
      pthread_mutex_unlock(&g_lock);
    });

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLCommandBuffer> cbu = [q commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        Class ceCls = [ce class];
        [ce endEncoding];
        id<MTLComputeCommandEncoder> cc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        Class ccCls = [cc class];
        [cc endEncoding];
        id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
        Class beCls = [be class];
        [be endEncoding];

        hook([q class], @selector(commandBuffer), (IMP)hook_commandBuffer, &orig_commandBuffer);
        hook([q class], @selector(commandBufferWithUnretainedReferences), (IMP)hook_commandBufferUnretained,
             &orig_commandBufferUnretained);
        hook([q class], @selector(commandBufferWithDescriptor:), (IMP)hook_commandBufferWithDescriptor,
             &orig_commandBufferWithDescriptor);

        for (Class c in @[ [cb class], [cbu class] ]) {
            hook(c, @selector(computeCommandEncoder), (IMP)hook_computeEncoder, &orig_computeEncoder);
            hook(c, @selector(computeCommandEncoderWithDispatchType:), (IMP)hook_computeEncoderDispatchType,
                 &orig_computeEncoderDispatchType);
            hook(c, @selector(computeCommandEncoderWithDescriptor:), (IMP)hook_computeEncoderDescriptor,
                 &orig_computeEncoderDescriptor);
            hook(c, @selector(blitCommandEncoder), (IMP)hook_blitEncoder, &orig_blitEncoder);
            hook(c, @selector(commit), (IMP)hook_commit, &orig_commit);
            hook(c, @selector(enqueue), (IMP)hook_enqueue, &orig_enqueue);
            hook(c, @selector(waitUntilCompleted), (IMP)hook_waitUntilCompleted, &orig_waitUntilCompleted);
            hook(c, @selector(waitUntilScheduled), (IMP)hook_waitUntilScheduled, &orig_waitUntilScheduled);
            hook(c, @selector(addCompletedHandler:), (IMP)hook_addCompletedHandler, &orig_addCompletedHandler);
            hook(c, @selector(addScheduledHandler:), (IMP)hook_addScheduledHandler, &orig_addScheduledHandler);
            hook(c, @selector(encodeSignalEvent:value:), (IMP)hook_encodeSignalEvent, &orig_encodeSignalEvent);
            hook(c, @selector(encodeWaitForEvent:value:), (IMP)hook_encodeWaitForEvent, &orig_encodeWaitForEvent);
        }
        if (ccCls != ceCls) NSLog(@"metal_api_trace: concurrent encoder class differs; hooking %@ only", ceCls);
        Class c = ceCls;
        hook(c, @selector(endEncoding), (IMP)hook_computeEnd, &orig_computeEnd);
        hook(beCls, @selector(endEncoding), (IMP)hook_blitEnd, &orig_blitEnd);
        // METAL_API_TRACE_LIGHT keeps only command buffer and encoder boundaries, so
        // host timestamps are not inflated by per-dispatch logging.
        if (getenv("METAL_API_TRACE_LIGHT")) return;
        hook(c, @selector(setComputePipelineState:), (IMP)hook_setPSO, &orig_setPSO);
        hook(c, @selector(setBuffer:offset:atIndex:), (IMP)hook_setBuffer, &orig_setBuffer);
        hook(c, @selector(setBufferOffset:atIndex:), (IMP)hook_setBufferOffset, &orig_setBufferOffset);
        hook(c, @selector(setBuffers:offsets:withRange:), (IMP)hook_setBuffers, &orig_setBuffers);
        hook(c, @selector(setBytes:length:atIndex:), (IMP)hook_setBytes, &orig_setBytes);
        hook(c, @selector(setThreadgroupMemoryLength:atIndex:), (IMP)hook_setTGMem, &orig_setTGMem);
        hook(c, @selector(dispatchThreadgroups:threadsPerThreadgroup:), (IMP)hook_dispatchTG, &orig_dispatchTG);
        hook(c, @selector(dispatchThreads:threadsPerThreadgroup:), (IMP)hook_dispatchThreads, &orig_dispatchThreads);
        hook(c, @selector(dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:),
             (IMP)hook_dispatchIndirect, &orig_dispatchIndirect);
        hook(c, @selector(memoryBarrierWithScope:), (IMP)hook_barrierScope, &orig_barrierScope);
        hook(c, @selector(memoryBarrierWithResources:count:), (IMP)hook_barrierResources, &orig_barrierResources);
        hook(c, @selector(useResource:usage:), (IMP)hook_useResource, &orig_useResource);
        hook(c, @selector(useResources:count:usage:), (IMP)hook_useResources, &orig_useResources);
        hook(c, @selector(useHeap:), (IMP)hook_useHeap, &orig_useHeap);
        hook(c, @selector(useHeaps:count:), (IMP)hook_useHeaps, &orig_useHeaps);
        hook(c, @selector(pushDebugGroup:), (IMP)hook_pushDebugGroup, &orig_pushDebugGroup);
        hook(c, @selector(popDebugGroup), (IMP)hook_popDebugGroup, &orig_popDebugGroup);
        hook(c, @selector(updateFence:), (IMP)hook_computeUpdateFence, &orig_computeUpdateFence);
        hook(c, @selector(waitForFence:), (IMP)hook_computeWaitFence, &orig_computeWaitFence);

        hook(beCls, @selector(copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:), (IMP)hook_copyBuffer,
             &orig_copyBuffer);
        hook(beCls, @selector(fillBuffer:range:value:), (IMP)hook_fillBuffer, &orig_fillBuffer);

        Class d = [dev class];
        hook(d, @selector(newComputePipelineStateWithFunction:error:), (IMP)hook_psoFunction, &orig_psoFunction);
        hook(d, @selector(newComputePipelineStateWithFunction:options:reflection:error:),
             (IMP)hook_psoFunctionOptions, &orig_psoFunctionOptions);
        hook(d, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), (IMP)hook_psoDescriptor,
             &orig_psoDescriptor);
    }
}
