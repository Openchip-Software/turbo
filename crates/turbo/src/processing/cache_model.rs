//! A cache model: L1 and L2, private to one access stream.
//!
//! This is the cache model API, with a **stand-in
//! implementation**. The intent is that the body of [`Caches`] is replaced
//! wholesale by a real cache modelling library and nothing outside this file
//! changes — so everything here is either the API itself, or a set-associative
//! LRU simple enough to be obviously correct and to give the integration tests
//! numbers they can compute by hand.
//!
//! What it does model: two levels, configurable size and associativity, LRU
//! replacement, an L1 miss looking up L2, and stores allocating exactly as loads
//! do. What it does not: writebacks, dirty state, write-through, inclusion
//! policy, prefetching, timing. Those belong to the real library; the counters
//! here would not become more truthful by guessing at them.
//!
//! # The shape of the API, and why
//!
//! [`Caches::access`] takes a [`Footprint`] rather than an `(address, size)`
//! pair. A single RVV instruction can touch 256 cache lines, and its footprint
//! is one of exactly three shapes — contiguous, strided, or an explicit line
//! list — so describing the shape lets one instruction be **one call** whatever
//! its addressing mode. The alternative costs one call per element for strided
//! ops and one per line for gathers; on `membench_loads` that is 262 calls per
//! iteration against 6.

use serde::{Deserialize, Serialize};

use turbo_tracer_shared::vmem::CACHE_LINE_BYTES;

/// Cache line size, in bytes.
///
/// A constant rather than a parameter because the *tracer* already quantises
/// indexed footprints to this granularity at capture time (the index vector
/// only exists while the guest runs), so a configurable smaller line would be a
/// knob whose lower half silently did nothing. Kept equal to the tracer's
/// constant by the assertion below rather than by hoping.
pub const LINE_BYTES: u64 = CACHE_LINE_BYTES;

const _: () = assert!(LINE_BYTES.is_power_of_two());

/// Whether an access reads or writes. A parameter rather than two methods
/// because every caller already holds this as data (`VecMemOp::is_store`), and
/// branching on it to pick a method is work at the call site to undo work the
/// trace did once.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AccessKind {
    Load,
    Store,
}

/// What one access, or a whole run, did.
///
/// One type for both: a single access's counts and an accumulated total mean
/// the same thing, so attribution is `+=` rather than a conversion.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Counts {
    pub l1_hits: u64,
    pub l1_misses: u64,
    /// L2 lookups are made only for lines that missed L1, so
    /// `l2_hits + l2_misses == l1_misses`.
    pub l2_hits: u64,
    pub l2_misses: u64,
}

impl Counts {
    /// Lines this access (or run) looked up, i.e. the size of the footprint in
    /// lines. Callers cross-check this against their own line count.
    pub fn lines(&self) -> u64 {
        self.l1_hits + self.l1_misses
    }
}

impl std::ops::AddAssign for Counts {
    fn add_assign(&mut self, o: Self) {
        self.l1_hits += o.l1_hits;
        self.l1_misses += o.l1_misses;
        self.l2_hits += o.l2_hits;
        self.l2_misses += o.l2_misses;
    }
}

/// The bytes one memory instruction touched, in the three shapes real
/// instructions come in.
///
/// These map one-to-one onto the trace's captured address groups
/// (`MemGroupKind::Base`, `BaseStride`, `Lines`), which is not a coincidence:
/// the wire format keeps a strided access folded as `(base, stride)` precisely
/// so nothing downstream has to unroll it.
#[derive(Clone, Copy, Debug)]
pub enum Footprint<'a> {
    /// Contiguous bytes: scalar accesses, unit-stride, whole-register, mask,
    /// fault-only-first.
    Range { addr: u64, len: u64 },
    /// `count` runs of `len` bytes, `stride` apart. `stride` is signed because
    /// RVV strides routinely are, and a caller-side unroll is where that gets
    /// mishandled.
    Strided {
        base: u64,
        stride: i64,
        count: u64,
        len: u64,
    },
    /// Line-aligned addresses, for gather/scatter, whose footprint has no
    /// analytic form. Need not be sorted or unique — this model deduplicates,
    /// because a model that double-counted a repeated line would be a trap for
    /// every caller that is not our tracer (ours are already sorted and
    /// unique).
    Lines(&'a [u64]),
}

/// L1/L2 geometry.
///
/// **Deliberately has no `Default`.** The cache being modelled is a property of
/// the target core, and a guessed one produces numbers indistinguishable from
/// measured ones. So it is configuration: `cpu_config.json`'s `cache` block (see
/// [`crate::config::CpuConfig`]), and `--cache-sim` fails rather than falls back
/// when that block is missing.
///
/// `deny_unknown_fields` for the same reason: with no default, a typo'd
/// `l1_way` would otherwise read as "no L1 associativity given" and there would
/// be nothing to give it.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub l1_bytes: u64,
    pub l1_ways: u32,
    pub l2_bytes: u64,
    pub l2_ways: u32,
}

impl Config {
    /// Reject a geometry this model cannot represent.
    ///
    /// Separate from [`Caches::new`] (which calls it) so that the CLI can check
    /// a config file at startup, instead of a bad `cache` block surfacing after
    /// a recording has already been made.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let check = |what: &'static str, v: u64| -> Result<(), ConfigError> {
            if !v.is_power_of_two() {
                return Err(ConfigError::NotPowerOfTwo(what, v));
            }
            if v < LINE_BYTES {
                return Err(ConfigError::TooSmall(what));
            }
            Ok(())
        };
        check("l1_bytes", self.l1_bytes)?;
        check("l2_bytes", self.l2_bytes)?;
        if !self.l1_ways.is_power_of_two() {
            return Err(ConfigError::NotPowerOfTwo("l1_ways", self.l1_ways.into()));
        }
        if !self.l2_ways.is_power_of_two() {
            return Err(ConfigError::NotPowerOfTwo("l2_ways", self.l2_ways.into()));
        }
        if self.l2_bytes <= self.l1_bytes {
            return Err(ConfigError::L2NotLargerThanL1);
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ConfigError {
    NotPowerOfTwo(&'static str, u64),
    TooSmall(&'static str),
    L2NotLargerThanL1,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPowerOfTwo(what, v) => write!(f, "{what} must be a power of two, got {v}"),
            Self::TooSmall(what) => write!(f, "{what} is smaller than one line ({LINE_BYTES} B)"),
            Self::L2NotLargerThanL1 => write!(f, "the L2 must be larger than the L1"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// A two-level cache, private to one access stream.
///
/// One instance per hart. There is deliberately no sharing, no locking and no
/// interior mutability: a shared last level would need a cross-hart access
/// order, and per-hart trace logs carry none — any interleaving would be an
/// artefact of decode scheduling rather than of the guest.
#[derive(Debug)]
pub struct Caches {
    l1: Level,
    l2: Level,
    counts: Counts,
    /// Number of [`Caches::access`] calls. Not a cache statistic: it is how the
    /// tests check that a vector instruction costs one call and has not been
    /// silently unrolled per element or per line. See [`Caches::calls`].
    calls: u64,
    /// Deduplicated lines of the footprint being walked. Reused so the access
    /// path allocates nothing.
    scratch: Vec<u64>,
}

impl Caches {
    /// Validation happens up front (via [`Config::validate`]) so that
    /// [`Caches::access`] cannot fail and needs no `Result` on the hot path.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self {
            l1: Level::new(config.l1_bytes, config.l1_ways),
            l2: Level::new(config.l2_bytes, config.l2_ways),
            counts: Counts::default(),
            calls: 0,
            scratch: Vec::new(),
        })
    }

    /// Look up every line of `fp`, in order, chaining L1 misses into the L2.
    ///
    /// Returns the counts for *this* access; the running totals are also
    /// updated ([`Caches::counts`]). Reporting per access is what makes per-PC
    /// attribution free for the caller: one call per instruction means the
    /// return value already is that instruction's cache behaviour.
    pub fn access(&mut self, kind: AccessKind, fp: Footprint<'_>) -> Counts {
        self.calls += 1;

        // Store-allocate is the same as load-allocate here; a real library is
        // where write policy belongs.
        let _ = kind;

        let mut out = Counts::default();

        // A contiguous range needs no buffer at all: `lines_covered` yields its
        // lines strictly increasing, which is already the sorted, deduplicated
        // order the general path below goes to the trouble of producing. This is
        // the shape almost every access has (unit-stride and whole-register
        // loads and stores), so it is the one worth not allocating for.
        if let Footprint::Range { addr, len } = fp {
            for line in lines_covered(addr, len) {
                self.lookup(line, &mut out);
            }
            self.counts += out;
            return out;
        }

        let mut lines = std::mem::take(&mut self.scratch);
        lines.clear();
        match fp {
            Footprint::Range { .. } => unreachable!("handled above"),
            Footprint::Strided {
                base,
                stride,
                count,
                len,
            } => {
                for i in 0..count {
                    let addr = base.wrapping_add((i as i64).wrapping_mul(stride) as u64);
                    push_lines(&mut lines, addr, len);
                }
            }
            Footprint::Lines(ls) => lines.extend(ls.iter().map(|&l| l & !(LINE_BYTES - 1))),
        }
        // A line touched twice by one instruction is one lookup: the indexed
        // capture already deduplicates inside the tracer, and the derived kinds
        // must match it or the same reuse would count differently depending on
        // the addressing mode it came from.
        //
        // Checked rather than assumed, but checked in one pass: a positive
        // stride at least as long as its runs, and our tracer's own line lists,
        // both arrive strictly increasing already, and sorting them is pure
        // cost. A negative stride or an unsorted list from another producer
        // still gets the full sort, so the documented contract holds.
        if !is_strictly_increasing(&lines) {
            lines.sort_unstable();
            lines.dedup();
        }

        for &line in &lines {
            self.lookup(line, &mut out);
        }

        self.scratch = lines;
        self.counts += out;
        out
    }

    /// Look one line up in L1, chaining a miss into L2, and tally it in `out`.
    #[inline]
    fn lookup(&mut self, line: u64, out: &mut Counts) {
        if self.l1.access(line) {
            out.l1_hits += 1;
        } else {
            out.l1_misses += 1;
            if self.l2.access(line) {
                out.l2_hits += 1;
            } else {
                out.l2_misses += 1;
            }
        }
    }

    /// Cumulative counts since construction or the last [`Caches::reset_counts`].
    pub fn counts(&self) -> Counts {
        self.counts
    }

    /// How many times [`Caches::access`] has been called. Exposed for the
    /// wiring tests: the useful assertion about this integration is not only
    /// "the hit counts are right" but "the model was called once per
    /// instruction".
    pub fn calls(&self) -> u64 {
        self.calls
    }

    /// Zero the counters, leaving cache contents alone.
    ///
    /// Separate from [`Caches::invalidate_all`] on purpose: entering a region of
    /// interest should restart its counters without pretending the cache went
    /// cold, and an API that conflates the two makes warm-region numbers
    /// impossible to ask for.
    pub fn reset_counts(&mut self) {
        self.counts = Counts::default();
        self.calls = 0;
    }

    /// Drop all cached state, leaving the configuration and counters alone.
    pub fn invalidate_all(&mut self) {
        self.l1.invalidate_all();
        self.l2.invalidate_all();
    }
}

/// The lines covered by `[addr, addr + len)`, strictly increasing.
fn lines_covered(addr: u64, len: u64) -> impl Iterator<Item = u64> {
    let (first, last) = if len == 0 {
        // An empty range covers nothing; `(1, 0)` is the empty step range.
        (1, 0)
    } else {
        (
            addr & !(LINE_BYTES - 1),
            (addr + len - 1) & !(LINE_BYTES - 1),
        )
    };
    (first..=last).step_by(LINE_BYTES as usize)
}

/// Append the lines covered by `[addr, addr + len)`.
fn push_lines(out: &mut Vec<u64>, addr: u64, len: u64) {
    out.extend(lines_covered(addr, len));
}

/// Whether `lines` is already sorted with no duplicates — i.e. whether the
/// `sort_unstable` + `dedup` that would follow are both no-ops.
fn is_strictly_increasing(lines: &[u64]) -> bool {
    lines.windows(2).all(|w| w[0] < w[1])
}

/// One set-associative level with LRU replacement.
///
/// Sets are flat `Vec<u64>` windows of `ways` tags, most-recently-used first, so
/// a lookup is a short linear scan and a promotion is a rotate. At the
/// associativities caches actually have (4-16) that beats anything with a map in
/// it, and it keeps this file free of data structures worth reviewing.
#[derive(Debug)]
struct Level {
    /// `sets * ways` tags, `EMPTY` for an unused way.
    tags: Vec<u64>,
    ways: usize,
    set_mask: u64,
}

const EMPTY: u64 = u64::MAX;

impl Level {
    fn new(bytes: u64, ways: u32) -> Self {
        let sets = (bytes / LINE_BYTES / u64::from(ways)).max(1);
        Self {
            tags: vec![EMPTY; (sets * u64::from(ways)) as usize],
            ways: ways as usize,
            set_mask: sets - 1,
        }
    }

    /// Look `line` up, inserting it on a miss. Returns whether it was a hit.
    #[inline]
    fn access(&mut self, line: u64) -> bool {
        let set = (line / LINE_BYTES) & self.set_mask;
        let base = set as usize * self.ways;
        let set = &mut self.tags[base..base + self.ways];

        // Hitting the MRU way is both the most common hit and the one where the
        // promotion below is a rotate of a single element, i.e. does nothing.
        // Worth its own test ahead of the scan: a loop walking a line's worth of
        // elements, or re-reading the same array, lands here every time.
        if set[0] == line {
            return true;
        }

        if let Some(pos) = set.iter().position(|&t| t == line) {
            set[..=pos].rotate_right(1); // promote to MRU
            true
        } else {
            set.rotate_right(1); // evict LRU, make room at MRU
            set[0] = line;
            false
        }
    }

    fn invalidate_all(&mut self) {
        self.tags.fill(EMPTY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The geometry the unit tests reason about: 32 KiB 8-way L1, 512 KiB 8-way
    /// L2. Local to the tests rather than a `Config::DEFAULT` const, so that
    /// editing the shipped `cpu_config.json` cannot move hand-computed
    /// expectations out from under them.
    const TEST_CONFIG: Config = Config {
        l1_bytes: 32 * 1024,
        l1_ways: 8,
        l2_bytes: 512 * 1024,
        l2_ways: 8,
    };

    fn caches() -> Caches {
        Caches::new(TEST_CONFIG).expect("the test config is valid")
    }

    /// The three footprint shapes must agree about what "one line" is: the same
    /// bytes described three ways are the same lookups.
    #[test]
    fn the_three_shapes_agree_on_one_line() {
        for fp in [
            Footprint::Range {
                addr: 0x1000,
                len: 8,
            },
            Footprint::Strided {
                base: 0x1000,
                stride: 8,
                count: 8,
                len: 8,
            },
            Footprint::Lines(&[0x1000]),
        ] {
            let mut c = caches();
            let out = c.access(AccessKind::Load, fp);
            assert_eq!(out.lines(), 1, "{fp:?} covers exactly one line");
            assert_eq!(out.l1_misses, 1, "{fp:?} is a cold miss");
        }
    }

    /// A range spanning several lines is one call and several lookups -- the
    /// property the tuple return exists for.
    #[test]
    fn a_multi_line_range_is_one_call_and_many_lookups() {
        let mut c = caches();
        let out = c.access(
            AccessKind::Load,
            Footprint::Range {
                addr: 0x2000,
                len: 512,
            },
        );
        assert_eq!(
            (out.l1_hits, out.l1_misses),
            (0, 8),
            "512 B = 8 lines, cold"
        );
        assert_eq!(c.calls(), 1, "one instruction is one call");

        // Second time it is warm, and still one call.
        let out = c.access(
            AccessKind::Load,
            Footprint::Range {
                addr: 0x2000,
                len: 512,
            },
        );
        assert_eq!((out.l1_hits, out.l1_misses), (8, 0));
        assert_eq!(c.calls(), 2);
    }

    /// One access that hits some lines and misses others: the case a `bool`
    /// return could not express, and the reason the API is shaped this way.
    #[test]
    fn one_access_can_both_hit_and_miss() {
        let mut c = caches();
        c.access(
            AccessKind::Load,
            Footprint::Range {
                addr: 0x3000,
                len: 128,
            },
        );
        let out = c.access(
            AccessKind::Load,
            Footprint::Range {
                addr: 0x3000,
                len: 256,
            },
        );
        assert_eq!(
            (out.l1_hits, out.l1_misses),
            (2, 2),
            "the first two lines are warm, the next two are not"
        );
    }

    /// Lines repeated within one access are one lookup, however the repetition
    /// arises -- an unsorted duplicate list, or a stride shorter than the run.
    #[test]
    fn a_line_touched_twice_by_one_access_is_one_lookup() {
        let mut c = caches();
        let out = c.access(AccessKind::Load, Footprint::Lines(&[0x40, 0x0, 0x40, 0x0]));
        assert_eq!(out.lines(), 2, "two distinct lines from four entries");

        let mut c = caches();
        // 8 runs of 8 bytes, 8 bytes apart: 64 contiguous bytes = one line.
        let out = c.access(
            AccessKind::Load,
            Footprint::Strided {
                base: 0x4000,
                stride: 8,
                count: 8,
                len: 8,
            },
        );
        assert_eq!(out.lines(), 1);
    }

    /// Negative strides are ordinary: a footprint that walks backwards covers
    /// the same lines as the forward walk that ends where it starts.
    #[test]
    fn a_negative_stride_walks_backwards() {
        let mut c = caches();
        let out = c.access(
            AccessKind::Load,
            Footprint::Strided {
                base: 0x5000,
                stride: -64,
                count: 4,
                len: 8,
            },
        );
        assert_eq!(out.lines(), 4);
        assert_eq!(out.l1_misses, 4);
        // The same four lines, now warm, reached forwards.
        let out = c.access(
            AccessKind::Load,
            Footprint::Strided {
                base: 0x5000 - 3 * 64,
                stride: 64,
                count: 4,
                len: 8,
            },
        );
        assert_eq!(out.l1_hits, 4, "the backwards walk covered exactly these");
    }

    /// L1 misses, and only L1 misses, reach the L2.
    #[test]
    fn l2_sees_exactly_the_l1_misses() {
        let mut c = caches();
        let out = c.access(
            AccessKind::Load,
            Footprint::Range {
                addr: 0x6000,
                len: 256,
            },
        );
        assert_eq!(out.l2_hits + out.l2_misses, out.l1_misses);
        assert_eq!(out.l2_misses, 4, "cold in both levels");

        // Evict from L1 by streaming past its capacity, then come back: the L2
        // must still have the lines.
        for i in 0..(64 * 1024 / 64) {
            c.access(
                AccessKind::Load,
                Footprint::Range {
                    addr: 0x100000 + i * 64,
                    len: 64,
                },
            );
        }
        let out = c.access(
            AccessKind::Load,
            Footprint::Range {
                addr: 0x6000,
                len: 256,
            },
        );
        assert_eq!(
            (out.l1_misses, out.l2_hits, out.l2_misses),
            (4, 4, 0),
            "evicted from L1, still resident in the larger L2"
        );
    }

    /// A working set larger than the cache must miss again on the next pass --
    /// otherwise capacity is not modelled at all and every number here is just
    /// counting first touches.
    #[test]
    fn capacity_is_modelled() {
        let cfg = Config {
            l1_bytes: 1024,
            l1_ways: 2,
            l2_bytes: 2048,
            l2_ways: 2,
        };
        let mut c = Caches::new(cfg).unwrap();
        let sweep = |c: &mut Caches| {
            let mut n = Counts::default();
            for i in 0..64u64 {
                n += c.access(
                    AccessKind::Load,
                    Footprint::Range {
                        addr: 0x10000 + i * 64,
                        len: 64,
                    },
                );
            }
            n
        };
        assert_eq!(sweep(&mut c).l1_misses, 64, "cold");
        assert_eq!(
            sweep(&mut c).l1_misses,
            64,
            "4 KiB working set through a 1 KiB L1 misses every time"
        );

        // The same sweep inside the L1 hits on the second pass.
        let mut c = Caches::new(cfg).unwrap();
        let small = |c: &mut Caches| {
            let mut n = Counts::default();
            for i in 0..8u64 {
                n += c.access(
                    AccessKind::Load,
                    Footprint::Range {
                        addr: 0x20000 + i * 64,
                        len: 64,
                    },
                );
            }
            n
        };
        assert_eq!(small(&mut c).l1_misses, 8);
        assert_eq!(small(&mut c).l1_hits, 8, "512 B fits in a 1 KiB L1");
    }

    /// Counters and contents are reset separately: an ROI wants its own counts
    /// without being told the cache went cold.
    #[test]
    fn resetting_counts_leaves_the_cache_warm() {
        let mut c = caches();
        let fp = Footprint::Range {
            addr: 0x7000,
            len: 64,
        };
        c.access(AccessKind::Load, fp);
        c.reset_counts();
        assert_eq!(c.counts(), Counts::default());
        assert_eq!(c.calls(), 0);

        assert_eq!(c.access(AccessKind::Load, fp).l1_hits, 1, "still warm");
        c.invalidate_all();
        assert_eq!(c.access(AccessKind::Load, fp).l1_misses, 1, "now cold");
    }

    #[test]
    fn an_invalid_config_is_rejected_at_construction() {
        let bad = Config {
            l1_bytes: 1000,
            ..TEST_CONFIG
        };
        assert_eq!(
            Caches::new(bad).unwrap_err(),
            ConfigError::NotPowerOfTwo("l1_bytes", 1000)
        );
        let bad = Config {
            l2_bytes: 1024,
            l1_bytes: 2048,
            ..TEST_CONFIG
        };
        assert_eq!(
            Caches::new(bad).unwrap_err(),
            ConfigError::L2NotLargerThanL1
        );
    }
}
