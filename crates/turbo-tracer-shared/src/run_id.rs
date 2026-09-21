//! The per-run identifier: a fixed-width, self-validating token that the host
//! mints once per QEMU invocation and every producer stamps into what it
//! writes.
//!
//! [`RunId`] is a `[u8; RUN_ID_BYTES]` newtype rather than a `String` because
//! the id has exactly one shape and always has had. Making that shape the type
//! moves every check to the three boundaries where an id genuinely arrives from
//! outside -- a plugin argument, a record-log header, `turbo_metadata.json` --
//! and leaves everything past those boundaries infallible, `Copy` and
//! allocation-free. Notably it makes the record-log header a CONSTANT width, so
//! neither writer nor reader carries a length field, a length limit, or the
//! error cases those imply.
//!
//! "Absent" is [`Option<RunId>`], not an empty id: a producer invoked by hand
//! rather than by `turbo run` has no run of its own, and a consumer must be
//! able to see that rather than compare against a sentinel that silently
//! matches nothing.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Width of a [`RunId`] in bytes: the `run_` prefix plus 16 hex digits.
pub const RUN_ID_BYTES: usize = 20;

/// Number of hex digits behind the prefix, i.e. one `u64` rendered in full.
const RUN_ID_HEX_DIGITS: usize = 16;

/// The `run_` prefix is load-bearing, not decoration: the id is passed to the
/// plugin as a QEMU plugin argument, and QEMU's argument parser types each
/// value by *content* -- it tries bool, then `i64`, and only then string. A
/// bare 16-digit hex id that happens to contain no `a`-`f` (~1 run in 11k)
/// parses as an integer, so a plugin reading the parsed value as a string sees
/// nothing and stamps no run id into its metadata, which the consumer then
/// rejects as "not from this run". The prefix makes the value un-parseable as
/// anything but a string, and [`RunId::parse`] rejecting an id without it makes
/// that a property of the type rather than of the one function that formats it.
// nosemgrep -- a 4-byte format tag, not a credential
const RUN_ID_PREFIX: &[u8] = b"run_";

/// An opaque, per-run identifier: unique enough to distinguish one QEMU
/// invocation from any other (including a previous run that reused the same
/// work_dir), so consumers can tell "this file is from a run that hasn't
/// finished writing yet" and "this file is a stale leftover from a different
/// run" apart from "this file is genuinely mine".
///
/// Not a cryptographic UUID -- just PID + a monotonic in-process counter +
/// wall-clock nanos, hashed down to a short hex string. Its only job is to
/// differ between two runs that share a directory.
///
/// The bytes are always `run_` followed by 16 lowercase hex digits: ASCII by
/// construction, which is why [`as_str`](Self::as_str) is infallible and no
/// reader of one needs a UTF-8 error path.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RunId([u8; RUN_ID_BYTES]);

impl RunId {
    /// Mint a fresh id. Infallible: this is the only way one comes into
    /// existence that is not a parse of somebody else's bytes.
    pub fn generate() -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let counter = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let seed = format!("{nanos}-{}-{counter}", std::process::id());
        let text = format!("run_{:016x}", crate::fnv1a_hash_64(seed.as_bytes()));
        let mut bytes = [0u8; RUN_ID_BYTES];
        bytes.copy_from_slice(text.as_bytes());
        Self(bytes)
    }

    /// Parse an id out of bytes that came from somewhere else -- a plugin
    /// argument, a file header, a JSON field. `None` for anything that is not
    /// exactly `run_` + 16 lowercase hex digits, which covers both "this field
    /// was empty/absent" and "these bytes are damaged": neither is an id, and a
    /// caller that needs to tell them apart already knows which it is looking
    /// at.
    pub fn from_bytes(bytes: [u8; RUN_ID_BYTES]) -> Option<Self> {
        if !bytes.starts_with(RUN_ID_PREFIX) {
            return None;
        }
        let digits = &bytes[RUN_ID_PREFIX.len()..];
        debug_assert_eq!(digits.len(), RUN_ID_HEX_DIGITS);
        if !digits
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
        {
            return None;
        }
        Some(Self(bytes))
    }

    /// As [`from_bytes`](Self::from_bytes), for a string of unknown length.
    /// Anything not exactly [`RUN_ID_BYTES`] long is `None`.
    pub fn parse(s: &str) -> Option<Self> {
        let bytes: [u8; RUN_ID_BYTES] = s.as_bytes().try_into().ok()?;
        Self::from_bytes(bytes)
    }

    /// The id's bytes, for writing into a fixed-width field.
    pub fn as_bytes(&self) -> &[u8; RUN_ID_BYTES] {
        &self.0
    }

    /// The id as text. Infallible: every construction path has already checked
    /// that these bytes are ASCII.
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).expect("a RunId is ASCII by construction")
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Quoted, like the `String` it replaced: run ids appear in diagnostics next to
/// paths and other quoted values, and an unquoted 20-character hex blob is
/// harder to pick out of one.
impl std::fmt::Debug for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

/// On the wire (JSON) an id is its string form, so `turbo_metadata.json` stays
/// human-readable and unchanged in shape.
impl Serialize for RunId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

/// Deserializes through [`RunId::parse`], refusing anything that is not one.
/// A field that may legitimately be absent should use
/// [`deserialize_optional`] instead of `Option<RunId>` on its own -- see there
/// for why.
impl<'de> Deserialize<'de> for RunId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        RunId::parse(&s).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "{s:?} is not a run id ({RUN_ID_BYTES} bytes: \"run_\" + 16 hex digits)"
            ))
        })
    }
}

/// `#[serde(default, deserialize_with = "...")]` for an `Option<RunId>` field
/// that means "the run this file belongs to, if it belongs to one".
///
/// Plain `Option<RunId>` would not do: serde only reaches `None` for an absent
/// or null field, so a present-but-unusable id -- `""` from a plugin invoked by
/// hand, or a damaged string -- would fail the whole document. A run id is a
/// staleness *hint*; not having a usable one means "skip the check", which is
/// exactly what every consumer already does, and it must not take the metadata
/// around it down with it.
pub fn deserialize_optional<'de, D: Deserializer<'de>>(d: D) -> Result<Option<RunId>, D::Error> {
    Ok(Option::<String>::deserialize(d)?
        .as_deref()
        .and_then(RunId::parse))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_generated_id_round_trips_through_its_text() {
        for _ in 0..1000 {
            let id = RunId::generate();
            assert_eq!(id.as_str().len(), RUN_ID_BYTES);
            assert_eq!(RunId::parse(id.as_str()), Some(id));
        }
    }

    #[test]
    // The id reaches the plugin as a QEMU plugin argument, and QEMU types each
    // argument by content: bool, then `i64`, then string. An id that parses as
    // an integer reaches a plugin reading `Args::parsed` as `Value::Integer`,
    // not `Value::String`. The `run_` prefix is what prevents that, and
    // `from_bytes` requires it -- so this holds for any RunId, not just a
    // freshly generated one.
    fn run_ids_never_type_as_bool_or_integer() {
        for _ in 0..10_000 {
            let id = RunId::generate();
            let s = id.as_str();
            assert!(s.parse::<i64>().is_err(), "run id {s} parses as an integer");
            assert!(
                !matches!(s, "on" | "off" | "true" | "false" | "yes" | "no"),
                "run id {s} parses as a bool"
            );
        }
    }

    #[test]
    fn run_ids_are_unique_within_a_process() {
        let ids: std::collections::HashSet<RunId> = (0..1000).map(|_| RunId::generate()).collect();
        assert_eq!(ids.len(), 1000);
    }

    #[test]
    fn only_the_one_shape_parses() {
        assert!(RunId::parse("run_0123456789abcdef").is_some());
        // Empty, short, long: all the ways an absent or truncated field looks.
        assert_eq!(RunId::parse(""), None);
        assert_eq!(RunId::parse("run_0123456789abcde"), None);
        assert_eq!(RunId::parse("run_0123456789abcdef0"), None);
        // Right length, wrong shape.
        assert_eq!(RunId::parse("xxx_0123456789abcdef"), None);
        assert_eq!(RunId::parse("run_0123456789ABCDEF"), None);
        assert_eq!(RunId::parse("run_0123456789abcdeg"), None);
        // Non-ASCII cannot survive the length check into `as_str`.
        assert_eq!(RunId::parse("run_0123456789abcdé"), None);
    }

    #[test]
    fn a_damaged_id_deserializes_as_none_not_an_error() {
        #[derive(serde::Deserialize)]
        struct Holder {
            #[serde(default, deserialize_with = "deserialize_optional")]
            run_id: Option<RunId>,
        }
        let ok: Holder = serde_json::from_str(r#"{"run_id":"run_0123456789abcdef"}"#).unwrap();
        assert_eq!(ok.run_id, RunId::parse("run_0123456789abcdef"));
        // The shapes a hand-run or older producer leaves behind.
        let empty: Holder = serde_json::from_str(r#"{"run_id":""}"#).unwrap();
        assert_eq!(empty.run_id, None);
        let absent: Holder = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(absent.run_id, None);
    }
}
