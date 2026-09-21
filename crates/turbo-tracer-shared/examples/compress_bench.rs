//! Measure what each deflate level costs and buys, on a real trace's payloads.
//!
//! End-to-end runs cannot answer this: compression is a fraction of a second
//! inside a QEMU phase that is seconds long and noisy, so levels 1 through 6
//! all land inside the run-to-run spread. This isolates the variable by
//! re-compressing the exact payloads the writer compressed, min-of-5.
//!
//! ```text
//! cargo b -r -p turbo_tracer_shared --example compress_bench
//! ./target/release/examples/compress_bench <workdir>/turbo_trace_vcpu0.bin
//! ```
//!
//! Re-run this before changing [`DEFAULT_COMPRESSION_LEVEL`] — the right level
//! depends on how compressible the workload's trace is, and that is a property
//! of the guest, not of the format.
//!
//! The last row is the per-record CRC-32 the framing carries, measured on the
//! same payloads and by the same min-of-5 method, so the checksum's cost can be
//! read directly against the deflate it rides behind. It is computed over the
//! COMPRESSED bytes, which is both what the writer does and why it is cheap:
//! that is the small side of a ~20x ratio.

use std::time::Instant;
use turbo_tracer_shared::*;

fn main() {
    let path = std::env::args().nth(1).expect("need a trace path");
    let records = read_record_log(std::path::Path::new(&path)).unwrap();
    // Re-serialize each batch to get the exact payloads the writer compresses.
    let payloads: Vec<Vec<u8>> = records
        .iter()
        .map(|r| postcard::to_allocvec(r).unwrap())
        .collect();
    let total: usize = payloads.iter().map(|p| p.len()).sum();
    println!(
        "{} batches, {:.1} MB of payload",
        payloads.len(),
        total as f64 / 1e6
    );
    println!(
        "{:>3} {:>12} {:>9} {:>10} {:>12}",
        "lvl",
        "compressed",
        "ratio",
        "MB/s",
        format!("cost/{:.0}MB", total as f64 / 1e6)
    );

    let mut default_level_seconds = f64::NAN;
    let mut default_level_compressed = 0usize;
    for level in [0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9] {
        let mut best = f64::MAX;
        let mut out_total = 0usize;
        for _ in 0..5 {
            let mut c = flate2::Compress::new(flate2::Compression::new(level), false);
            let mut scratch = Vec::new();
            let t = Instant::now();
            out_total = 0;
            for p in &payloads {
                c.reset();
                scratch.clear();
                scratch.reserve(p.len() / 4 + 1024);
                loop {
                    let consumed = c.total_in() as usize;
                    let before = c.total_out();
                    if scratch.len() == scratch.capacity() {
                        scratch.reserve(scratch.capacity().max(1024));
                    }
                    match c
                        .compress_vec(&p[consumed..], &mut scratch, flate2::FlushCompress::Finish)
                        .unwrap()
                    {
                        flate2::Status::StreamEnd => break,
                        _ => {
                            if c.total_out() == before {
                                scratch.reserve(scratch.capacity().max(1024));
                            }
                        }
                    }
                }
                out_total += scratch.len();
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        let mbs = total as f64 / best / 1e6;
        println!(
            "{level:>3} {:>12} {:>8.1}x {mbs:>10.0} {:>11.2}s",
            out_total,
            total as f64 / out_total as f64,
            total as f64 / 1e6 / mbs
        );
        if level == DEFAULT_COMPRESSION_LEVEL {
            default_level_seconds = best;
            default_level_compressed = out_total;
        }
    }

    // What the framing's CRC-32 adds, on the bytes it actually covers: the
    // compressed payloads, exactly as the writer checksums them (it runs over
    // the frame buffer it is about to hand to `write_all`).
    let compressed: Vec<Vec<u8>> = {
        let mut c =
            flate2::Compress::new(flate2::Compression::new(DEFAULT_COMPRESSION_LEVEL), false);
        payloads
            .iter()
            .map(|p| {
                let mut out = Vec::new();
                c.reset();
                loop {
                    let consumed = c.total_in() as usize;
                    let before = c.total_out();
                    if out.len() == out.capacity() {
                        out.reserve(out.capacity().max(1024));
                    }
                    match c
                        .compress_vec(&p[consumed..], &mut out, flate2::FlushCompress::Finish)
                        .unwrap()
                    {
                        flate2::Status::StreamEnd => break,
                        _ => {
                            if c.total_out() == before {
                                out.reserve(out.capacity().max(1024));
                            }
                        }
                    }
                }
                out
            })
            .collect()
    };

    let mut crc_best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let mut sum = 0u64;
        for c in &compressed {
            sum += u64::from(payload_crc32(c));
        }
        std::hint::black_box(sum);
        crc_best = crc_best.min(t.elapsed().as_secs_f64());
    }
    let crc_mbs = default_level_compressed as f64 / crc_best / 1e6;
    println!(
        "\ncrc32 over the {:.1} MB of level-{DEFAULT_COMPRESSION_LEVEL} output: \
         {crc_mbs:>.0} MB/s, {:.4}s",
        default_level_compressed as f64 / 1e6,
        crc_best
    );
    println!(
        "  = {:.2}% of the level-{DEFAULT_COMPRESSION_LEVEL} deflate it rides behind \
         ({:.4}s)",
        100.0 * crc_best / default_level_seconds,
        default_level_seconds
    );
}
