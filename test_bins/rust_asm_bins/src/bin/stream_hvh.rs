//! STREAM Benchmark Kernels - RISC-V Vector (RVV) Assembly Implementation
//!
//! This module implements the four STREAM benchmark kernels using inline RVV assembly:
//! - Copy: c[i] = a[i]
//! - Scale: b[i] = scalar * c[i]
//! - Add: c[i] = a[i] + b[i]
//! - Triad: a[i] = b[i] + scalar * c[i]
//!
//! All kernels use strip-mining with vsetvli for vector-length-agnostic code.
//! All kernels have integrated looping, to ensure all assembly is in a single asm!() block.

use serde::Serialize;
use serde_json;
use std::arch::asm;
use std::time::Instant;

use rust_asm_bins::rave;

#[derive(Debug, Copy, Clone, Serialize)]
/// Represents different RVV LMUL values
pub enum Lmul {
    M1,
    M8,
}

#[derive(Serialize)]
struct KernelResult {
    best_rate_mbps: Option<f64>,
    avg_time: Option<f64>,
    min_time: Option<f64>,
    max_time: Option<f64>,
}

#[derive(Serialize)]
struct LmulResult {
    lmul: Lmul,
    copy: Option<KernelResult>,
    scale: Option<KernelResult>,
    add: Option<KernelResult>,
    triad: Option<KernelResult>,
}

/// Copy Kernel: c[i] = a[i]
///
/// Pure memory copy using RVV vector loads and stores.
/// Uses e64 (double precision) elements with unit-stride access.
#[inline(never)]
pub fn stream_copy(a: &[f64], c: &mut [f64], lmul: Lmul) -> Result<usize, ()> {
    if a.len() != c.len() {
        return Err(());
    }

    let remaining = a.len();
    let a_ptr = a.as_ptr();
    let c_ptr = c.as_mut_ptr();

    match lmul {
        Lmul::M1 => unsafe {
            let _r = rave::Region::new(b"copy_m1");
            asm!(
                "1:",                                       // Loop label
                "vsetvli {vl}, {avl}, e64, m1, ta, ma",     // Set vector length
                // Ensure register vN where (N % LMUL) == 0, sigILL if not!
                "vle64.v v8, ({src})",                      // Load from a
                "vse64.v v8, ({dst})",                      // Store to c
                "sub {avl}, {avl}, {vl}",                   // Decrement remaining count
                "slli {tmp}, {vl}, 3",                      // Compute byte offset (vl * 8)
                "add {src}, {src}, {tmp}",                  // Advance source pointer
                "add {dst}, {dst}, {tmp}",                  // Advance destination pointer
                "bnez {avl}, 1b",                           // Loop if elements remain

                vl = out(reg) _,
                avl = inout(reg) remaining => _,
                src = inout(reg) a_ptr => _,
                dst = inout(reg) c_ptr => _,
                tmp = out(reg) _,
                options(nostack)
            );
        },
        Lmul::M8 => unsafe {
            let _r = rave::Region::new(b"copy_m8");
            asm!(
                "1:",                                       // Loop label
                "vsetvli {vl}, {avl}, e64, m8, ta, ma",     // Set vector length
                // Ensure register vN where (N % LMUL) == 0, sigILL if not!
                "vle64.v v8, ({src})",                      // Load from a
                "vse64.v v8, ({dst})",                      // Store to c
                "sub {avl}, {avl}, {vl}",                   // Decrement remaining count
                "slli {tmp}, {vl}, 3",                      // Compute byte offset (vl * 8)
                "add {src}, {src}, {tmp}",                  // Advance source pointer
                "add {dst}, {dst}, {tmp}",                  // Advance destination pointer
                "bnez {avl}, 1b",                           // Loop if elements remain

                vl = out(reg) _,
                avl = inout(reg) remaining => _,
                src = inout(reg) a_ptr => _,
                dst = inout(reg) c_ptr => _,
                tmp = out(reg) _,
                options(nostack)
            );
        },
    }

    Ok(a.len())
}

/// Scale Kernel: b[i] = scalar * c[i]
///
/// Scales array c by a scalar value and stores in b.
/// Uses vfmv.v.f to broadcast scalar, then vfmul.vv for multiplication.
#[inline(never)]
pub fn stream_scale(scalar: f64, c: &[f64], b: &mut [f64], lmul: Lmul) -> Result<usize, ()> {
    if c.len() != b.len() {
        return Err(());
    }

    let remaining = c.len();
    let c_ptr = c.as_ptr();
    let b_ptr = b.as_mut_ptr();

    match lmul {
        Lmul::M1 => unsafe {
            let _r = rave::Region::new(b"scale_m1");
            asm!(
                "vsetvli {vl}, zero, e64, m1, ta, ma",      // Set max vector length
                "vfmv.v.f v5, {scalar}",                    // Broadcast scalar to v5
                "1:",                                       // Loop label
                "vsetvli {vl}, {avl}, e64, m1, ta, ma",     // Set vector length
                "vle64.v v8, ({src})",                      // Load from c
                "vfmul.vv v9, v8, v5",                      // Multiply: v9 = v8 * v5
                "vse64.v v9, ({dst})",                      // Store to b
                "sub {avl}, {avl}, {vl}",                   // Decrement remaining count
                "slli {tmp}, {vl}, 3",                      // Compute byte offset (vl * 8)
                "add {src}, {src}, {tmp}",                  // Advance source pointer
                "add {dst}, {dst}, {tmp}",                  // Advance destination pointer
                "bnez {avl}, 1b",                           // Loop if elements remain

                scalar = in(freg) scalar,
                vl = out(reg) _,
                avl = inout(reg) remaining => _,
                src = inout(reg) c_ptr => _,
                dst = inout(reg) b_ptr => _,
                tmp = out(reg) _,
                out("v5") _,
                options(nostack)
            );
        },
        Lmul::M8 => unsafe {
            let _r = rave::Region::new(b"scale_m8");
            asm!(
                "vsetvli {vl}, zero, e64, m8, ta, ma",      // Set max vector length
                "vfmv.v.f v8, {scalar}",                    // Broadcast scalar to v8
                "1:",                                       // Loop label
                "vsetvli {vl}, {avl}, e64, m8, ta, ma",     // Set vector length
                "vle64.v v16, ({src})",                     // Load from c
                "vfmul.vv v24, v16, v8",                    // Multiply: v24 = v16 * v8
                "vse64.v v24, ({dst})",                     // Store to b
                "sub {avl}, {avl}, {vl}",                   // Decrement remaining count
                "slli {tmp}, {vl}, 3",                      // Compute byte offset (vl * 8)
                "add {src}, {src}, {tmp}",                  // Advance source pointer
                "add {dst}, {dst}, {tmp}",                  // Advance destination pointer
                "bnez {avl}, 1b",                           // Loop if elements remain

                scalar = in(freg) scalar,
                vl = out(reg) _,
                avl = inout(reg) remaining => _,
                src = inout(reg) c_ptr => _,
                dst = inout(reg) b_ptr => _,
                tmp = out(reg) _,
                out("v8") _,
                options(nostack)
            );
        },
    }

    Ok(c.len())
}

/// Add Kernel: c[i] = a[i] + b[i]
///
/// Adds two arrays element-wise.
/// Dual vector loads followed by vfadd.vv for addition.
#[inline(never)]
pub fn stream_add(a: &[f64], b: &[f64], c: &mut [f64], lmul: Lmul) -> Result<usize, ()> {
    if a.len() != b.len() || a.len() != c.len() {
        return Err(());
    }

    let remaining = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    match lmul {
        Lmul::M1 => unsafe {
            let _r = rave::Region::new(b"add_m1");
            asm!(
                "1:",                                       // Loop label
                "vsetvli {vl}, {avl}, e64, m1, ta, ma",     // Set vector length
                "vle64.v v8, ({src_a})",                    // Load from a
                "vle64.v v9, ({src_b})",                    // Load from b
                "vfadd.vv v10, v9, v8",                     // Add: v10 = v9 + v8
                "vse64.v v10, ({dst})",                     // Store to c
                "sub {avl}, {avl}, {vl}",                   // Decrement remaining count
                "slli {tmp}, {vl}, 3",                      // Compute byte offset (vl * 8)
                "add {src_a}, {src_a}, {tmp}",              // Advance pointer a
                "add {src_b}, {src_b}, {tmp}",              // Advance pointer b
                "add {dst}, {dst}, {tmp}",                  // Advance pointer c
                "bnez {avl}, 1b",                           // Loop if elements remain

                vl = out(reg) _,
                avl = inout(reg) remaining => _,
                src_a = inout(reg) a_ptr => _,
                src_b = inout(reg) b_ptr => _,
                dst = inout(reg) c_ptr => _,
                tmp = out(reg) _,
                options(nostack)
            );
        },
        Lmul::M8 => unsafe {
            let _r = rave::Region::new(b"add_m8");
            asm!(
                "1:",                                       // Loop label
                "vsetvli {vl}, {avl}, e64, m8, ta, ma",     // Set vector length
                "vle64.v v8, ({src_a})",                    // Load from a
                "vle64.v v16, ({src_b})",                   // Load from b
                "vfadd.vv v24, v16, v8",                    // Add: v24 = v16 + v8
                "vse64.v v24, ({dst})",                     // Store to c
                "sub {avl}, {avl}, {vl}",                   // Decrement remaining count
                "slli {tmp}, {vl}, 3",                      // Compute byte offset (vl * 8)
                "add {src_a}, {src_a}, {tmp}",              // Advance pointer a
                "add {src_b}, {src_b}, {tmp}",              // Advance pointer b
                "add {dst}, {dst}, {tmp}",                  // Advance pointer c
                "bnez {avl}, 1b",                           // Loop if elements remain

                vl = out(reg) _,
                avl = inout(reg) remaining => _,
                src_a = inout(reg) a_ptr => _,
                src_b = inout(reg) b_ptr => _,
                dst = inout(reg) c_ptr => _,
                tmp = out(reg) _,
                options(nostack)
            );
        },
    }

    Ok(a.len())
}

/// Triad Kernel: a[i] = b[i] + scalar * c[i]
///
/// Fused multiply-add operation combining scaling and addition.
/// Uses vfmacc.vv for efficient computation: a = b + scalar * c
#[inline(never)]
pub fn stream_triad(
    scalar: f64,
    b: &[f64],
    c: &[f64],
    a: &mut [f64],
    lmul: Lmul,
) -> Result<usize, ()> {
    if a.len() != b.len() || a.len() != c.len() {
        return Err(());
    }

    let remaining = a.len();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_ptr();
    let a_ptr = a.as_mut_ptr();

    match lmul {
        Lmul::M1 => unsafe {
            let _r = rave::Region::new(b"triad_m1");
            asm!(
                "vsetvli {vl}, zero, e64, m1, ta, ma",      // Set max vector length
                "vfmv.v.f v5, {scalar}",                    // Broadcast scalar to v5
                "1:",                                       // Loop label
                "vsetvli {vl}, {avl}, e64, m1, ta, ma",     // Set vector length
                "vle64.v v8, ({src_c})",                    // Load from c
                "vle64.v v9, ({src_b})",                    // Load from b
                "vfmacc.vv v9, v5, v8",                     // FMA: v9 = v9 + v5 * v8
                "vse64.v v9, ({dst})",                      // Store to a
                "sub {avl}, {avl}, {vl}",                   // Decrement remaining count
                "slli {tmp}, {vl}, 3",                      // Compute byte offset (vl * 8)
                "add {src_c}, {src_c}, {tmp}",              // Advance pointer c
                "add {src_b}, {src_b}, {tmp}",              // Advance pointer b
                "add {dst}, {dst}, {tmp}",                  // Advance pointer a
                "bnez {avl}, 1b",                           // Loop if elements remain

                scalar = in(freg) scalar,
                vl = out(reg) _,
                avl = inout(reg) remaining => _,
                src_c = inout(reg) c_ptr => _,
                src_b = inout(reg) b_ptr => _,
                dst = inout(reg) a_ptr => _,
                tmp = out(reg) _,
                out("v5") _,
                options(nostack)
            );
        },
        Lmul::M8 => unsafe {
            let _r = rave::Region::new(b"triad_m8");
            asm!(
                "vsetvli {vl}, zero, e64, m8, ta, ma",      // Set max vector length
                "vfmv.v.f v8, {scalar}",                    // Broadcast scalar to v8
                "1:",                                       // Loop label
                "vsetvli {vl}, {avl}, e64, m8, ta, ma",     // Set vector length
                "vle64.v v16, ({src_c})",                   // Load from c
                "vle64.v v24, ({src_b})",                   // Load from b
                "vfmacc.vv v24, v8, v16",                   // FMA: v24 = v24 + v8 * v16
                "vse64.v v24, ({dst})",                     // Store to a
                "sub {avl}, {avl}, {vl}",                   // Decrement remaining count
                "slli {tmp}, {vl}, 3",                      // Compute byte offset (vl * 8)
                "add {src_c}, {src_c}, {tmp}",              // Advance pointer c
                "add {src_b}, {src_b}, {tmp}",              // Advance pointer b
                "add {dst}, {dst}, {tmp}",                  // Advance pointer a
                "bnez {avl}, 1b",                           // Loop if elements remain

                scalar = in(freg) scalar,
                vl = out(reg) _,
                avl = inout(reg) remaining => _,
                src_c = inout(reg) c_ptr => _,
                src_b = inout(reg) b_ptr => _,
                dst = inout(reg) a_ptr => _,
                tmp = out(reg) _,
                out("v8") _,
                options(nostack)
            );
        },
    }

    Ok(a.len())
}

fn main() {
    // Parse CLI arguments
    let args: Vec<String> = std::env::args().collect();

    // Check which algorithms are requested
    let mut enable_copy = None;
    let mut enable_scale = None;
    let mut enable_add = None;
    let mut enable_triad = None;
    let mut lmul_values: Vec<Lmul> = Vec::new();
    let mut results: Vec<LmulResult> = Vec::new();
    let mut ntimes: usize = 10;
    let mut stream_array_size: usize = 16_000_000;

    for arg in &args[1..] {
        if arg.starts_with("--lmul=") {
            let value_str = &arg[7..];
            match value_str {
                "1" => {
                    if !lmul_values.iter().any(|l| matches!(l, Lmul::M1)) {
                        lmul_values.push(Lmul::M1);
                    }
                }
                "8" => {
                    if !lmul_values.iter().any(|l| matches!(l, Lmul::M8)) {
                        lmul_values.push(Lmul::M8);
                    }
                }
                _ => {
                    eprintln!(
                        "Error: --lmul only accepts values 1 or 8, got: {}",
                        value_str
                    );
                    std::process::exit(1);
                }
            }
        } else if arg.starts_with("--ntimes=") {
            let value_str = &arg[9..];
            match value_str.parse::<usize>() {
                Ok(value) => ntimes = value,
                Err(_) => {
                    eprintln!(
                        "Error: --ntimes must be a valid positive integer, got: {}",
                        value_str
                    );
                    std::process::exit(1);
                }
            }
        } else if arg.starts_with("--array_size=") {
            let value_str = &arg[13..];
            match value_str.parse::<usize>() {
                Ok(value) => stream_array_size = value,
                Err(_) => {
                    eprintln!(
                        "Error: --array_size must be a valid positive integer, got: {}",
                        value_str
                    );
                    std::process::exit(1);
                }
            }
        } else {
            match arg.as_str() {
                "--copy" => enable_copy = Some(()),
                "--scale" => enable_scale = Some(()),
                "--add" => enable_add = Some(()),
                "--triad" => enable_triad = Some(()),
                _ => {}
            }
        }
    }

    // If no algorithms specified, enable all
    let any_specified = enable_copy.is_some()
        || enable_scale.is_some()
        || enable_add.is_some()
        || enable_triad.is_some();

    if !any_specified {
        enable_copy = Some(());
        enable_scale = Some(());
        enable_add = Some(());
        enable_triad = Some(());
    }

    // If no LMUL values specified, test both
    if lmul_values.is_empty() {
        lmul_values.push(Lmul::M1);
        lmul_values.push(Lmul::M8);
    }

    println!("-------------------------------------------------------------");
    println!("STREAM Benchmark - Rust/HVH Edition");
    println!("-------------------------------------------------------------");

    const OFFSET: usize = 0;
    const SCALAR: f64 = 3.0;

    println!("-------------------------------------------------------------");
    println!(
        "Array size = {} (elements), Offset = {} (elements)",
        stream_array_size, OFFSET
    );

    let bytes_per_array = (stream_array_size * std::mem::size_of::<f64>()) as f64;
    let mib_per_array = bytes_per_array / (1024.0 * 1024.0);
    let gib_per_array = bytes_per_array / (1024.0 * 1024.0 * 1024.0);

    println!(
        "Memory per array = {:.1} MiB (= {:.1} GiB).",
        mib_per_array, gib_per_array
    );
    println!(
        "Total memory required = {:.1} MiB (= {:.1} GiB).",
        mib_per_array * 3.0,
        gib_per_array * 3.0
    );
    println!("Each kernel will be executed {} times.", ntimes);
    println!(" The *best* time for each kernel (excluding the first iteration)");
    println!(" will be used to compute the reported bandwidth.");
    println!("-------------------------------------------------------------");

    // Allocate arrays
    let mut a = vec![1.0f64; stream_array_size];
    let mut b = vec![2.0f64; stream_array_size];
    let mut c = vec![0.0f64; stream_array_size];

    // Timing arrays
    let mut copy_times = vec![0.0f64; ntimes];
    let mut scale_times = vec![0.0f64; ntimes];
    let mut add_times = vec![0.0f64; ntimes];
    let mut triad_times = vec![0.0f64; ntimes];

    // Run benchmarks
    println!("Running benchmarks...");
    println!();

    for lmul in &lmul_values {
        let lmul = lmul.clone();

        for k in 0..ntimes {
            // Copy: c[i] = a[i]
            if enable_copy.is_some() {
                let start = Instant::now();
                stream_copy(&a, &mut c, lmul).unwrap();
                let elapsed = start.elapsed();
                copy_times[k] = elapsed.as_secs_f64();
            }

            // Scale: b[i] = scalar * c[i]
            if enable_scale.is_some() {
                let start = Instant::now();
                stream_scale(SCALAR, &c, &mut b, lmul).unwrap();
                let elapsed = start.elapsed();
                scale_times[k] = elapsed.as_secs_f64();
            }

            // Add: c[i] = a[i] + b[i]
            if enable_add.is_some() {
                let start = Instant::now();
                stream_add(&a, &b, &mut c, lmul).unwrap();
                let elapsed = start.elapsed();
                add_times[k] = elapsed.as_secs_f64();
            }

            // Triad: a[i] = b[i] + scalar * c[i]
            if enable_triad.is_some() {
                let start = Instant::now();
                stream_triad(SCALAR, &b, &c, &mut a, lmul).unwrap();
                let elapsed = start.elapsed();
                triad_times[k] = elapsed.as_secs_f64();
            }
        }

        // Lambda to calculate statistics (excluding first iteration)
        let calc_stats = |times: &[f64]| -> (f64, f64, f64) {
            let min = times[1..].iter().copied().fold(f64::INFINITY, f64::min);
            let avg = times[1..].iter().sum::<f64>() / (ntimes - 1) as f64;
            let max = times[1..].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (min, avg, max)
        };

        // Lambda to calculate bandwidth in MB/s
        let calc_bandwidth = |num_arrays: f64, min_time: f64| -> f64 {
            let bytes = num_arrays * stream_array_size as f64 * std::mem::size_of::<f64>() as f64;
            (bytes / (1024.0 * 1024.0)) / min_time
        };

        // Calculate statistics for each kernel
        let copy_stats = enable_copy.map(|_| calc_stats(&copy_times));
        let scale_stats = enable_scale.map(|_| calc_stats(&scale_times));
        let add_stats = enable_add.map(|_| calc_stats(&add_times));
        let triad_stats = enable_triad.map(|_| calc_stats(&triad_times));

        // Calculate bandwidth in MB/s
        let copy_mbps = copy_stats.map(|(min, _, _)| calc_bandwidth(2.0, min)); // Copy: 2 arrays (read a, write c)
        let scale_mbps = scale_stats.map(|(min, _, _)| calc_bandwidth(2.0, min)); // Scale: 2 arrays (read c, write b)
        let add_mbps = add_stats.map(|(min, _, _)| calc_bandwidth(3.0, min)); // Add: 3 arrays (read a, read b, write c)
        let triad_mbps = triad_stats.map(|(min, _, _)| calc_bandwidth(3.0, min)); // Triad: 3 arrays (read b, read c, write a)

        let lmul_result = LmulResult {
            lmul: lmul.clone(),
            copy: copy_mbps
                .zip(copy_stats)
                .map(|(mbps, (min, avg, max))| KernelResult {
                    best_rate_mbps: Some(mbps),
                    avg_time: Some(avg),
                    min_time: Some(min),
                    max_time: Some(max),
                }),
            scale: scale_mbps
                .zip(scale_stats)
                .map(|(mbps, (min, avg, max))| KernelResult {
                    best_rate_mbps: Some(mbps),
                    avg_time: Some(avg),
                    min_time: Some(min),
                    max_time: Some(max),
                }),
            add: add_mbps
                .zip(add_stats)
                .map(|(mbps, (min, avg, max))| KernelResult {
                    best_rate_mbps: Some(mbps),
                    avg_time: Some(avg),
                    min_time: Some(min),
                    max_time: Some(max),
                }),
            triad: triad_mbps
                .zip(triad_stats)
                .map(|(mbps, (min, avg, max))| KernelResult {
                    best_rate_mbps: Some(mbps),
                    avg_time: Some(avg),
                    min_time: Some(min),
                    max_time: Some(max),
                }),
        };

        results.push(lmul_result);
    }

    // Print results as JSON
    let json = serde_json::to_string(&results).unwrap();
    println!("{}", json);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let kernel = KernelResult {
            best_rate_mbps: Some(123.4),
            avg_time: Some(0.001),
            min_time: Some(0.0009),
            max_time: Some(0.0011),
        };
        let lmul_result = LmulResult {
            lmul: Lmul::M1,
            copy: Some(kernel),
            scale: None,
            add: None,
            triad: None,
        };
        let json = serde_json::to_string(&lmul_result).unwrap();
        // Verify it's valid JSON
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_stream_copy() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut c = vec![0.0; 5];

        let result = stream_copy(&a, &mut c, Lmul::M1);
        assert!(result.is_ok());
        assert_eq!(c, a);

        let result = stream_copy(&a, &mut c, Lmul::M8);
        assert!(result.is_ok());
        assert_eq!(c, a);
    }

    #[test]
    fn test_stream_scale() {
        let c = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut b = vec![0.0; 5];
        let scalar = 2.5;

        let result = stream_scale(scalar, &c, &mut b, Lmul::M1);
        assert!(result.is_ok());

        for i in 0..5 {
            assert!((b[i] - c[i] * scalar).abs() < 1e-10);
        }

        let result = stream_scale(scalar, &c, &mut b, Lmul::M8);
        assert!(result.is_ok());

        for i in 0..5 {
            assert!((b[i] - c[i] * scalar).abs() < 1e-10);
        }
    }

    #[test]
    fn test_stream_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let mut c = vec![0.0; 5];

        let result = stream_add(&a, &b, &mut c, Lmul::M1);
        assert!(result.is_ok());

        for i in 0..5 {
            assert_eq!(c[i], a[i] + b[i]);
        }

        let result = stream_add(&a, &b, &mut c, Lmul::M8);
        assert!(result.is_ok());

        for i in 0..5 {
            assert_eq!(c[i], a[i] + b[i]);
        }
    }

    #[test]
    fn test_stream_triad() {
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let c = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let mut a = vec![0.0; 5];
        let scalar = 2.0;

        let result = stream_triad(scalar, &b, &c, &mut a, Lmul::M1);
        assert!(result.is_ok());

        for i in 0..5 {
            assert!((a[i] - (b[i] + scalar * c[i])).abs() < 1e-10);
        }

        let result = stream_triad(scalar, &b, &c, &mut a, Lmul::M8);
        assert!(result.is_ok());

        for i in 0..5 {
            assert!((a[i] - (b[i] + scalar * c[i])).abs() < 1e-10);
        }
    }

    #[test]
    fn test_stream_copy_stress() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for lmul in [Lmul::M1, Lmul::M8] {
            for iteration in 0..100 {
                // Vary array length from 4096 to 8192
                let len = 4096 + (iteration * 40);

                // Generate random input data
                let a: Vec<f64> = (0..len).map(|_| rng.gen_range(-1000.0..1000.0)).collect();
                let mut c = vec![0.0; len];

                let result = stream_copy(&a, &mut c, lmul);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), len);

                // Verify all elements were copied correctly
                for i in 0..len {
                    assert_eq!(
                        c[i], a[i],
                        "Mismatch at index {} in iteration {} with {:?}",
                        i, iteration, lmul
                    );
                }
            }
        }
    }

    #[test]
    fn test_stream_scale_stress() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for lmul in [Lmul::M1, Lmul::M8] {
            for iteration in 0..100 {
                // Vary array length from 4096 to 8192
                let len = 4096 + (iteration * 40);

                // Generate random input data and scalar
                let c: Vec<f64> = (0..len).map(|_| rng.gen_range(-1000.0..1000.0)).collect();
                let mut b = vec![0.0; len];
                let scalar = rng.gen_range(-10.0..10.0);

                let result = stream_scale(scalar, &c, &mut b, lmul);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), len);

                // Verify all elements were scaled correctly
                for i in 0..len {
                    let expected = c[i] * scalar;
                    let diff = (b[i] - expected).abs();
                    assert!(
                        diff < 1e-10,
                        "Mismatch at index {} in iteration {} with {:?}: expected {}, got {}",
                        i,
                        iteration,
                        lmul,
                        expected,
                        b[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_stream_add_stress() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for lmul in [Lmul::M1, Lmul::M8] {
            for iteration in 0..100 {
                // Vary array length from 4096 to 8192
                let len = 4096 + (iteration * 40);

                // Generate random input data
                let a: Vec<f64> = (0..len).map(|_| rng.gen_range(-1000.0..1000.0)).collect();
                let b: Vec<f64> = (0..len).map(|_| rng.gen_range(-1000.0..1000.0)).collect();
                let mut c = vec![0.0; len];

                let result = stream_add(&a, &b, &mut c, lmul);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), len);

                // Verify all elements were added correctly
                for i in 0..len {
                    let expected = a[i] + b[i];
                    let diff = (c[i] - expected).abs();
                    assert!(
                        diff < 1e-10,
                        "Mismatch at index {} in iteration {} with {:?}: expected {}, got {}",
                        i,
                        iteration,
                        lmul,
                        expected,
                        c[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_stream_triad_stress() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for lmul in [Lmul::M1, Lmul::M8] {
            for iteration in 0..100 {
                // Vary array length from 4096 to 8192
                let len = 4096 + (iteration * 40);

                // Generate random input data and scalar
                let b: Vec<f64> = (0..len).map(|_| rng.gen_range(-1000.0..1000.0)).collect();
                let c: Vec<f64> = (0..len).map(|_| rng.gen_range(-1000.0..1000.0)).collect();
                let mut a = vec![0.0; len];
                let scalar = rng.gen_range(-10.0..10.0);

                let result = stream_triad(scalar, &b, &c, &mut a, lmul);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), len);

                // Verify all elements computed correctly
                for i in 0..len {
                    let expected = b[i] + scalar * c[i];
                    let diff = (a[i] - expected).abs();
                    assert!(
                        diff < 1e-10,
                        "Mismatch at index {} in iteration {} with {:?}: expected {}, got {}",
                        i,
                        iteration,
                        lmul,
                        expected,
                        a[i]
                    );
                }
            }
        }
    }
}
