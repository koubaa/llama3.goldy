// GPU time of decode-shaped f32 GEMV (y = W x, W row-major [m, k]) on Metal.
//
// Candidates: MPSMatrixMultiplication (n = 1), MPSMatrixVectorMultiplication,
// llama.cpp's kernel_mul_mv_f32_f32_4 structure, and Goldy's stdlib gemv_f32.
// Weights rotate through a pool larger than the system-level cache so every
// dispatch streams from DRAM, as decode does.
//
//   swiftc -O tools/gemv_bench/main.swift -o /tmp/gemv_bench && /tmp/gemv_bench

import Foundation
import Metal
import MetalPerformanceShaders

let source = """
#include <metal_stdlib>
using namespace metal;

struct Args { uint m; uint k; };

// llama.cpp kernel_mul_mv_t_t_4_impl<float, float4, float, float4, NR0 = 2>.
// Threadgroup: nsg simdgroups, 2 rows. Lane (ix, il) reads 16 floats of each
// 32-float block; blocks stride by nsg * 16.
kernel void llama_mv(constant Args & a [[buffer(0)]],
                     device const float * W [[buffer(1)]],
                     device const float * x [[buffer(2)]],
                     device float * y [[buffer(3)]],
                     threadgroup float * shmem [[threadgroup(0)]],
                     uint3 tgpig [[threadgroup_position_in_grid]],
                     ushort tiisg [[thread_index_in_simdgroup]],
                     ushort sgitg [[simdgroup_index_in_threadgroup]],
                     ushort nsg [[simdgroups_per_threadgroup]]) {
    constexpr short NR0 = 2, NB = 32, NF = 16, NF4 = 4, NW = 32;
    const int nb = a.k / NB;
    const int r0 = tgpig.x * NR0;
    device const float4 * ax4[NR0];
    device const float * ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        uint r = min(uint(r0 + row), a.m - 1);
        ax[row] = W + (uint64_t)r * a.k;
        ax4[row] = (device const float4 *) ax[row];
    }
    float sumf[NR0] = {0.f, 0.f};
    const short ix = tiisg / (NW / NF);
    const short il = tiisg % (NW / NF);
    const int ib0 = sgitg * NF + ix;
    float4 yl4[NF4];
    device const float4 * yb4 = (device const float4 *) x + (ib0 * NB + il * NF) / 4;
    for (int ib = ib0; ib < nb; ib += nsg * NF) {
        for (short i = 0; i < NF4; ++i) yl4[i] = yb4[i];
        for (short row = 0; row < NR0; ++row) {
            device const float4 * xb4 = ax4[row] + (ib * NB + il * NF) / 4;
            float s = 0.f;
            for (short i = 0; i < NF4; ++i) s += dot(xb4[i], yl4[i]);
            sumf[row] += s;
        }
        yb4 += nsg * NF * NW / 4;
    }
    for (int i = nb * NB + sgitg * NW + tiisg; i < int(a.k); i += NW * nsg)
        for (short row = 0; row < NR0; ++row) sumf[row] += ax[row][i] * x[i];
    for (short row = 0; row < NR0; ++row) {
        if (sgitg == 0) shmem[NW * row + tiisg] = 0.f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; ++row)
        if (tiisg == 0) shmem[NW * row + sgitg] = sumf[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0 && uint(r0 + row) < a.m; ++row) {
        float tot = simd_sum(shmem[NW * row + tiisg]);
        if (tiisg == 0 && sgitg == 0) y[r0 + row] = tot;
    }
}

// Goldy ops/matmul_kernel.rs gemv_f32: 128 threads, 4 rows, 32 lanes per row,
// two strided accumulators, threadgroup tree reduction.
kernel void goldy_gemv(constant Args & a [[buffer(0)]],
                       device const float * W [[buffer(1)]],
                       device const float * x [[buffer(2)]],
                       device float * y [[buffer(3)]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint gid [[threadgroup_position_in_grid]]) {
    threadgroup float partial[128];
    uint lane = lid % 32;
    uint row = gid * 4 + lid / 32;
    float acc0 = 0.f, acc1 = 0.f;
    if (row < a.m) {
        uint base = row * a.k;
        uint j = lane;
        while (j + 32 < a.k) {
            acc0 += W[base + j] * x[j];
            acc1 += W[base + j + 32] * x[j + 32];
            j += 64;
        }
        if (j < a.k) acc0 += W[base + j] * x[j];
    }
    partial[lid] = acc0 + acc1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 16; s > 0; s /= 2) {
        if (lane < s) partial[lid] += partial[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0 && row < a.m) y[row] = partial[lid];
}

// Goldy ops/matmul_kernel.rs matmul_f32 (no transposes): one thread per output.
struct GemmArgs { uint m; uint n; uint k; };
kernel void goldy_gemm(constant GemmArgs & a [[buffer(0)]],
                       device const float * A [[buffer(1)]],
                       device const float * B [[buffer(2)]],
                       device float * C [[buffer(3)]],
                       uint idx [[thread_position_in_grid]]) {
    if (idx >= a.m * a.n) return;
    uint i = idx / a.n, j = idx - i * a.n;
    float acc = 0.f;
    for (uint p = 0; p < a.k; ++p) acc += A[i * a.k + p] * B[p * a.n + j];
    C[i * a.n + j] = acc;
}
"""

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let library = try! device.makeLibrary(source: source, options: nil)
let llamaPSO = try! device.makeComputePipelineState(function: library.makeFunction(name: "llama_mv")!)
let goldyPSO = try! device.makeComputePipelineState(function: library.makeFunction(name: "goldy_gemv")!)

let poolBytes = 256 << 20
let pool = device.makeBuffer(length: poolBytes, options: .storageModePrivate)!
let xBuf = device.makeBuffer(length: 4 << 20, options: .storageModeShared)!
let yBuf = device.makeBuffer(length: 4 << 20, options: .storageModeShared)!
do {
    let p = xBuf.contents().bindMemory(to: Float.self, capacity: 1 << 20)
    for i in 0..<(1 << 20) { p[i] = Float(i % 7) * 0.01 }
}

enum Kind: String, CaseIterable { case mpsGemm = "MPS GEMM n=1", mpsGemv = "MPS MatrixVector", llama = "llama mul_mv_4", goldy = "goldy gemv_f32" }

/// Median GPU microseconds per GEMV over `reps` dispatches, weights rotating through the pool.
func measure(_ kind: Kind, m: Int, k: Int, reps: Int = 64, trials: Int = 7) -> Double {
    let bytes = m * k * 4
    let slots = max(1, min(reps, poolBytes / bytes))
    var samples: [Double] = []
    for _ in 0..<trials {
        let cb = queue.makeCommandBuffer()!
        for r in 0..<reps {
            let offset = (r % slots) * bytes
            switch kind {
            case .mpsGemm:
                let dW = MPSMatrixDescriptor(rows: m, columns: k, rowBytes: k * 4, dataType: .float32)
                let dx = MPSMatrixDescriptor(rows: k, columns: 1, rowBytes: 4, dataType: .float32)
                let dy = MPSMatrixDescriptor(rows: m, columns: 1, rowBytes: 4, dataType: .float32)
                let op = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
                                                 resultRows: m, resultColumns: 1, interiorColumns: k, alpha: 1, beta: 0)
                op.encode(commandBuffer: cb,
                          leftMatrix: MPSMatrix(buffer: pool, offset: offset, descriptor: dW),
                          rightMatrix: MPSMatrix(buffer: xBuf, descriptor: dx),
                          resultMatrix: MPSMatrix(buffer: yBuf, descriptor: dy))
            case .mpsGemv:
                let dW = MPSMatrixDescriptor(rows: m, columns: k, rowBytes: k * 4, dataType: .float32)
                let op = MPSMatrixVectorMultiplication(device: device, transpose: false, rows: m, columns: k, alpha: 1, beta: 0)
                op.encode(commandBuffer: cb,
                          inputMatrix: MPSMatrix(buffer: pool, offset: offset, descriptor: dW),
                          inputVector: MPSVector(buffer: xBuf, descriptor: MPSVectorDescriptor(length: k, dataType: .float32)),
                          resultVector: MPSVector(buffer: yBuf, descriptor: MPSVectorDescriptor(length: m, dataType: .float32)))
            case .llama, .goldy:
                // One encoder per dispatch batch is what llama.cpp does; keep all reps in one encoder.
                break
            }
        }
        if kind == .llama || kind == .goldy {
            let enc = cb.makeComputeCommandEncoder()!
            var args = (UInt32(m), UInt32(k))
            for r in 0..<reps {
                let offset = (r % slots) * bytes
                enc.setBytes(&args, length: 8, index: 0)
                enc.setBuffer(pool, offset: offset, index: 1)
                enc.setBuffer(xBuf, offset: 0, index: 2)
                enc.setBuffer(yBuf, offset: 0, index: 3)
                if kind == .llama {
                    let nsg = min(4, (k + 127) / 128)
                    enc.setComputePipelineState(llamaPSO)
                    enc.setThreadgroupMemoryLength(32 * 2 * 4, index: 0)
                    enc.dispatchThreadgroups(MTLSize(width: (m + 1) / 2, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 32 * nsg, height: 1, depth: 1))
                } else {
                    enc.setComputePipelineState(goldyPSO)
                    enc.dispatchThreadgroups(MTLSize(width: (m + 3) / 4, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                }
            }
            enc.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        samples.append((cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(reps))
    }
    samples.sort()
    return samples[samples.count / 2]
}

func check() {
    // Correctness of the two hand kernels against MPS on one shape.
    let m = 768, k = 288
    let wShared = device.makeBuffer(length: m * k * 4, options: .storageModeShared)!
    let w = wShared.contents().bindMemory(to: Float.self, capacity: m * k)
    for i in 0..<(m * k) { w[i] = Float((i * 31) % 17) * 0.001 - 0.008 }
    let x = xBuf.contents().bindMemory(to: Float.self, capacity: k)
    var ref = [Float](repeating: 0, count: m)
    for r in 0..<m { var s: Float = 0; for j in 0..<k { s += w[r * k + j] * x[j] }; ref[r] = s }
    for (pso, tg, rows) in [(llamaPSO, 32 * min(4, (k + 127) / 128), 2), (goldyPSO, 128, 4)] {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        var args = (UInt32(m), UInt32(k))
        enc.setComputePipelineState(pso)
        enc.setBytes(&args, length: 8, index: 0)
        enc.setBuffer(wShared, offset: 0, index: 1)
        enc.setBuffer(xBuf, offset: 0, index: 2)
        enc.setBuffer(yBuf, offset: 0, index: 3)
        if pso === llamaPSO { enc.setThreadgroupMemoryLength(256, index: 0) }
        enc.dispatchThreadgroups(MTLSize(width: (m + rows - 1) / rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let y = yBuf.contents().bindMemory(to: Float.self, capacity: m)
        var err: Float = 0
        for r in 0..<m { err = max(err, abs(y[r] - ref[r])) }
        print("check \(pso === llamaPSO ? "llama" : "goldy") max abs err \(err)")
    }
}

let gemmPSO = try! device.makeComputePipelineState(function: library.makeFunction(name: "goldy_gemm")!)

/// Median GPU microseconds of C[m,n] = A[m,k] B[k,n]: MPS GEMM, or Goldy's stdlib GEMM.
func measureGemm(mps: Bool, m: Int, n: Int, k: Int, reps: Int = 32, trials: Int = 5) -> Double {
    let bytes = m * k * 4
    let slots = max(1, min(reps, poolBytes / bytes))
    var samples: [Double] = []
    for _ in 0..<trials {
        let cb = queue.makeCommandBuffer()!
        if mps {
            let op = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
                                             resultRows: m, resultColumns: n, interiorColumns: k, alpha: 1, beta: 0)
            let dA = MPSMatrixDescriptor(rows: m, columns: k, rowBytes: k * 4, dataType: .float32)
            let dB = MPSMatrixDescriptor(rows: k, columns: n, rowBytes: n * 4, dataType: .float32)
            let dC = MPSMatrixDescriptor(rows: m, columns: n, rowBytes: n * 4, dataType: .float32)
            for r in 0..<reps {
                op.encode(commandBuffer: cb,
                          leftMatrix: MPSMatrix(buffer: pool, offset: (r % slots) * bytes, descriptor: dA),
                          rightMatrix: MPSMatrix(buffer: xBuf, descriptor: dB),
                          resultMatrix: MPSMatrix(buffer: yBuf, descriptor: dC))
            }
        } else {
            let enc = cb.makeComputeCommandEncoder()!
            var args = (UInt32(m), UInt32(n), UInt32(k))
            enc.setComputePipelineState(gemmPSO)
            for r in 0..<reps {
                enc.setBytes(&args, length: 12, index: 0)
                enc.setBuffer(pool, offset: (r % slots) * bytes, index: 1)
                enc.setBuffer(xBuf, offset: 0, index: 2)
                enc.setBuffer(yBuf, offset: 0, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: (m * n + 255) / 256, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            enc.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        samples.append((cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(reps))
    }
    samples.sort()
    return samples[samples.count / 2]
}

print("device \(device.name)")
if CommandLine.arguments.contains("gemm") {
    print("m     k     n    MPS GEMM us   goldy matmul_f32 us   goldy gemv x n us")
    for (m, k) in [(768, 768), (2048, 768), (32000, 768)] {
        let gemv = measure(.goldy, m: m, k: k)
        for n in [1, 2, 3, 4, 8, 16, 32, 64] where m * n * 4 <= 4 << 20 {
            let a = measureGemm(mps: true, m: m, n: n, k: k)
            let b = measureGemm(mps: false, m: m, n: n, k: k)
            print(String(format: "%-5d %-5d %-4d %10.1f %18.1f %18.1f", m, k, n, a, b, gemv * Double(n)))
        }
    }
    exit(0)
}

check()
let shapes: [(String, Int, Int)] = [
    ("15M  wq/wo  288x288", 288, 288), ("15M  w1/w3  768x288", 768, 288), ("15M  w2    288x768", 288, 768),
    ("15M  cls  32000x288", 32000, 288),
    ("110M wq/wo  768x768", 768, 768), ("110M w1/w3 2048x768", 2048, 768), ("110M w2   768x2048", 768, 2048),
    ("110M cls  32000x768", 32000, 768),
]
print("shape                  " + Kind.allCases.map { $0.rawValue.padding(toLength: 18, withPad: " ", startingAt: 0) }.joined())
for (name, m, k) in shapes {
    var line = name.padding(toLength: 23, withPad: " ", startingAt: 0)
    for kind in Kind.allCases {
        let us = measure(kind, m: m, k: k)
        let gbs = Double(m * k * 4) / (us * 1e-6) / 1e9
        line += String(format: "%7.1fus %5.1fGB/s ", us, gbs)
    }
    print(line)
}
