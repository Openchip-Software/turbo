//! RISC-V Vector (RVV) inline assembly integration tests
//!
//! This test demonstrates explicit vectorization using RVV inline assembly
//! to perform floating-point array addition: result = a + b

#![cfg(target_arch = "riscv64")]

use libc::{gettimeofday, timeval};
use std::arch::asm;

/// Vectorized floating-point array addition using RVV inline assembly
///
/// Performs: result[i] = a[i] + b[i] for all elements
#[inline(never)]
pub fn vec_float_add(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<usize, ()> {
    if a.len() != b.len() || a.len() != result.len() {
        return Err(());
    }

    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let result_ptr = result.as_mut_ptr();

    // Process the array using RVV instructions
    // We'll process in chunks based on the vector length (vl)
    let mut processed: usize = 0;

    while processed < len {
        let remaining = len - processed;
        let vl: usize;

        // # Safety: RVV Vector extension instructions used, CPU must support them
        unsafe {
            let cur_a = a_ptr.add(processed);
            let cur_b = b_ptr.add(processed);
            let cur_result = result_ptr.add(processed);

            asm!(
                "vsetvli {vl}, {avl}, e32, m1, ta, ma",
                "vle32.v v8, ({a_ptr})",
                "vle32.v v9, ({b_ptr})",
                "vfadd.vv v10, v8, v9",
                "vse32.v v10, ({result_ptr})",

                vl = out(reg) vl,
                avl = in(reg) remaining,
                a_ptr = in(reg) cur_a,
                b_ptr = in(reg) cur_b,
                result_ptr = in(reg) cur_result,
                options(nostack)
            );
        }

        // println!("processed {vl} items this iteration");
        processed += vl;
    }

    return Ok(processed);
}

/// Scalar version for comparison
#[inline(never)]
fn scalar_float_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
}

fn test_vector_float_add_basic() {
    // Test with 4 elements
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [5.0_f32, 6.0, 7.0, 8.0];
    let mut result = [0.0_f32; 4];
    let expected = [6.0_f32, 8.0, 10.0, 12.0];

    vec_float_add(&a, &b, &mut result).unwrap();

    for i in 0..4 {
        assert_eq!(
            result[i], expected[i],
            "Mismatch at index {}: got {}, expected {}",
            i, result[i], expected[i]
        );
    }

    // println!("Basic vector float addition test passed");
}

fn test_vector_float_add_larger() {
    // Test with larger array (16 elements)
    let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..16).map(|i| (i * 2) as f32).collect();
    let mut result_vec = vec![0.0_f32; 16];
    let mut result_scalar = vec![0.0_f32; 16];

    // Compute with vector version
    vec_float_add(&a, &b, &mut result_vec).unwrap();

    // Compute with scalar version for verification
    scalar_float_add(&a, &b, &mut result_scalar);

    // Compare results
    for i in 0..16 {
        assert_eq!(
            result_vec[i], result_scalar[i],
            "Mismatch at index {}: vector={}, scalar={}",
            i, result_vec[i], result_scalar[i]
        );
    }

    println!("Larger vector float addition test passed (16 elements)");
}

fn test_vector_float_add_odd_size() {
    // Test with odd-sized array to ensure we handle remainders correctly
    let size = 100;
    let a: Vec<f32> = (0..size).map(|i| i as f32 + 0.5).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 3) as f32 + 0.25).collect();
    let mut result_vec = vec![0.0_f32; size];
    let mut result_scalar = vec![0.0_f32; size];

    // Compute with vector version
    for _ in 0..100 {
        vec_float_add(&a, &b, &mut result_vec).unwrap();
    }

    // Compute with scalar version for verification
    scalar_float_add(&a, &b, &mut result_scalar);

    // Compare results
    for i in 0..size {
        let diff = (result_vec[i] - result_scalar[i]).abs();
        assert!(
            diff < 1e-6,
            "Mismatch at index {}: vector={}, scalar={}, diff={}",
            i,
            result_vec[i],
            result_scalar[i],
            diff
        );
    }

    // println!(
    //     "Odd-sized vector float addition test passed ({} elements)",
    //     size
    // );
}

fn test_vector_float_add_negative_numbers() {
    // Test with negative numbers
    let a = [-1.0_f32, -2.5, 3.7, -4.2, 5.1];
    let b = [2.0_f32, -3.5, -1.3, 4.8, -2.1];
    let mut result = [0.0_f32; 5];
    let expected = [1.0_f32, -6.0, 2.4, 0.6, 3.0];

    vec_float_add(&a, &b, &mut result).unwrap();

    for i in 0..5 {
        let diff = (result[i] - expected[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: got {}, expected {}, diff={}",
            i,
            result[i],
            expected[i],
            diff
        );
    }

    println!("Vector float addition with negative numbers test passed");
}

fn test_vector_float_add_single_element() {
    // Edge case: single element
    let a = [42.0_f32];
    let b = [13.0_f32];
    let mut result = [0.0_f32];
    let expected = [55.0_f32];

    vec_float_add(&a, &b, &mut result).unwrap();

    assert_eq!(result[0], expected[0]);

    println!("Single element vector float addition test passed");
}

fn print_proc_self_maps() {
    use std::fs::File;
    use std::io::{BufRead, BufReader, Write};

    println!("\n=== /proc/self/maps ===");

    match File::open("/proc/self/maps") {
        Ok(file) => {
            let reader = BufReader::new(file);
            for line in reader.lines() {
                match line {
                    Ok(content) => println!("{}", content),
                    Err(e) => eprintln!("Error reading line: {}", e),
                }
            }
        }
        Err(e) => eprintln!("Error opening /proc/self/maps: {}", e),
    }

    println!("=== End of /proc/self/maps ===\n");

    // Parse for VDSO and dump
    let mut vdso_start = None;
    let mut vdso_end = None;
    match File::open("/proc/self/maps") {
        Ok(file) => {
            let reader = BufReader::new(file);
            for line in reader.lines() {
                if let Ok(content) = line {
                    if content.contains("[vdso]") {
                        let parts: Vec<&str> = content.split_whitespace().collect();
                        if parts.len() >= 6 {
                            let addr_range = parts[0];
                            let addr_parts: Vec<&str> = addr_range.split('-').collect();
                            if addr_parts.len() == 2 {
                                if let (Ok(start), Ok(end)) = (
                                    u64::from_str_radix(addr_parts[0], 16),
                                    u64::from_str_radix(addr_parts[1], 16),
                                ) {
                                    vdso_start = Some(start);
                                    vdso_end = Some(end);
                                }
                            }
                        }
                    }
                }
            }
        }
        Err(e) => eprintln!("Error opening /proc/self/maps for parsing: {}", e),
    }

    if let (Some(start), Some(end)) = (vdso_start, vdso_end) {
        let size = end - start;
        if size > 1024 * 1024 {
            panic!("VDSO size {} bytes > 1MB", size);
        }
        let slice = unsafe { std::slice::from_raw_parts(start as *const u8, size as usize) };
        let filename = format!("vdso_mem-{:x}-{:x}.dump", start, end);
        match File::create(&filename) {
            Ok(mut file) => {
                if let Err(e) = file.write_all(slice) {
                    eprintln!("Error writing to {}: {}", filename, e);
                } else {
                    println!("Dumped VDSO to {}", filename);
                }
            }
            Err(e) => eprintln!("Error creating {}: {}", filename, e),
        }
    } else {
        eprintln!("VDSO not found in /proc/self/maps");
    }
}

fn main() {
    let simple = std::env::args().any(|arg| arg == "--simple");
    if simple {
        for _ in 0..100 {
            test_vector_float_add_basic();
        }
        return;
    }

    println!("Running RISC-V Vector (RVV) inline assembly tests...\n");
    print_proc_self_maps();

    let mut tv = timeval {
        tv_sec: 0,
        tv_usec: 0,
    };
    unsafe {
        gettimeofday(&mut tv, std::ptr::null_mut());
    }
    println!("Current time: {}.{:06}", tv.tv_sec, tv.tv_usec);

    // Run all test cases
    test_vector_float_add_basic();
    test_vector_float_add_larger();
    test_vector_float_add_odd_size();
    test_vector_float_add_negative_numbers();
    test_vector_float_add_single_element();

    println!("\nAll tests completed successfully!");
}
