use anyhow::Result;
use murmur3::murmur3_x64_128;
use simd_minimizers::packed_seq::{PackedSeqVec, SeqVec};
use std::collections::{BTreeSet, HashMap};
use std::io::Cursor;
use tab_hash::{Tab64Simple, Tab64Twisted};

use crate::{AniConfig, HashValue, Offset, SeqId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimizerMode {
    Simd,
    Scalar,
    ScalarMinmer,
}

impl MinimizerMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Simd => "simd",
            Self::Scalar => "scalar",
            Self::ScalarMinmer => "scalar-minmer",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabulationMode {
    Twisted,
    Simple,
}

impl TabulationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Twisted => "twisted",
            Self::Simple => "simple",
        }
    }
}

pub(crate) enum TabulationHasher {
    Twisted(Tab64Twisted),
    Simple(Tab64Simple),
}

impl TabulationHasher {
    fn hash(&self, key: u64) -> u64 {
        match self {
            Self::Twisted(hasher) => hasher.hash(key),
            Self::Simple(hasher) => hasher.hash(key),
        }
    }
}

pub(crate) fn simd_compatible_window_size(k: usize, w: usize) -> usize {
    if (k + w - 1) % 2 == 1 {
        w
    } else if w > 1 {
        w - 1
    } else {
        w + 1
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Minimizer {
    pub(crate) hash: HashValue,
    pub(crate) seq_id: SeqId,
    pub(crate) wpos: Offset,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MinmerInterval {
    pub(crate) hash: HashValue,
    pub(crate) seq_id: SeqId,
    pub(crate) start: Offset,
    pub(crate) end: Offset,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QuerySeed {
    pub(crate) hash: HashValue,
    pub(crate) qpos: Offset,
}

pub(crate) fn sequence_minimizers(
    seq: &[u8],
    config: &AniConfig,
    w: usize,
    seq_id: SeqId,
    tab_hasher: &TabulationHasher,
) -> Result<Vec<Minimizer>> {
    match config.minimizer_mode {
        MinimizerMode::Simd => {
            simd_sequence_minimizers(seq, config.kmer_size, w, seq_id, tab_hasher)
        }
        MinimizerMode::Scalar => scalar_sequence_minimizers(seq, config.kmer_size, w, seq_id),
        MinimizerMode::ScalarMinmer => scalar_sequence_minmer_query_sketch(
            seq,
            config.kmer_size,
            config.resolved_minmer_sketch_size(),
            seq_id,
        ),
    }
}

fn simd_sequence_minimizers(
    seq: &[u8],
    k: usize,
    w: usize,
    seq_id: SeqId,
    tab_hasher: &TabulationHasher,
) -> Result<Vec<Minimizer>> {
    if seq.len() < k + w - 1 {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    let mut run_start = 0usize;

    while run_start < seq.len() {
        while run_start < seq.len() && !is_acgt(seq[run_start]) {
            run_start += 1;
        }
        if run_start >= seq.len() {
            break;
        }

        let mut run_end = run_start;
        while run_end < seq.len() && is_acgt(seq[run_end]) {
            run_end += 1;
        }

        if run_end - run_start >= k + w - 1 {
            let packed = PackedSeqVec::from_ascii(&seq[run_start..run_end]);
            let mut minimizer_positions = Vec::new();
            let mut super_kmers = Vec::new();
            let values = simd_minimizers::canonical_minimizers(k, w)
                .super_kmers(&mut super_kmers)
                .run(packed.as_slice(), &mut minimizer_positions)
                .values_u64()
                .collect::<Vec<_>>();

            for (wpos, canonical_value) in super_kmers.into_iter().zip(values) {
                result.push(Minimizer {
                    hash: minimizer_token(canonical_value, k, tab_hasher),
                    seq_id,
                    wpos: run_start + wpos as usize,
                });
            }
        }

        run_start = run_end;
    }

    Ok(result)
}

#[derive(Debug, Clone, Copy)]
struct ScalarDequeEntry {
    hash: HashValue,
    pos: Offset,
    emitted: bool,
}

fn scalar_sequence_minimizers(
    seq: &[u8],
    k: usize,
    w: usize,
    seq_id: SeqId,
) -> Result<Vec<Minimizer>> {
    if seq.len() < k || seq.len() < w {
        return Ok(Vec::new());
    }

    let seq_upper = seq
        .iter()
        .map(|&base| {
            if base.is_ascii_lowercase() {
                base.to_ascii_uppercase()
            } else {
                base
            }
        })
        .collect::<Vec<_>>();
    let seq_rev = fastani_reverse_complement(&seq_upper);
    let mut result = Vec::new();
    let mut deque: std::collections::VecDeque<ScalarDequeEntry> = std::collections::VecDeque::new();
    let max_i = seq_upper.len() - k + 1;

    for i in 0..max_i {
        let hash_fwd = murmurhash3_x64_128_low32(&seq_upper[i..i + k], 42)? as HashValue;
        let rc_start = seq_upper.len() - i - k;
        let hash_bwd =
            murmurhash3_x64_128_low32(&seq_rev[rc_start..rc_start + k], 42)? as HashValue;

        if hash_bwd == hash_fwd {
            continue;
        }

        let current_hash = hash_fwd.min(hash_bwd);

        if i >= w {
            let stale_pos = i - w;
            while deque.front().is_some_and(|entry| entry.pos <= stale_pos) {
                deque.pop_front();
            }
        }

        while deque.back().is_some_and(|entry| entry.hash >= current_hash) {
            deque.pop_back();
        }

        deque.push_back(ScalarDequeEntry {
            hash: current_hash,
            pos: i,
            emitted: false,
        });

        if i + 1 >= w {
            let current_window_id = i + 1 - w;
            if let Some(front) = deque.front_mut() {
                if !front.emitted {
                    front.emitted = true;
                    result.push(Minimizer {
                        hash: front.hash,
                        seq_id,
                        wpos: current_window_id,
                    });
                }
            }
        }
    }

    Ok(result)
}

pub(crate) fn scalar_sequence_minmer_query_sketch(
    seq: &[u8],
    k: usize,
    sketch_size: usize,
    seq_id: SeqId,
) -> Result<Vec<Minimizer>> {
    if seq.len() < k || sketch_size == 0 {
        return Ok(Vec::new());
    }

    let seq_upper = uppercase_ascii(seq);
    let seq_rev = fastani_reverse_complement(&seq_upper);
    let mut sketch = Vec::new();
    for (run_start, run_end) in acgt_runs(&seq_upper) {
        if run_end - run_start < k {
            continue;
        }
        for pos in run_start..=(run_end - k) {
            if let Some(hash) = canonical_murmur_hash_at(&seq_upper, &seq_rev, pos, k)? {
                sketch.push(Minimizer {
                    hash,
                    seq_id,
                    wpos: pos,
                });
            }
        }
    }

    sketch.sort_unstable_by_key(|m| (m.hash, m.wpos));
    sketch.dedup_by_key(|m| m.hash);
    sketch.truncate(sketch_size);
    sketch.sort_unstable_by_key(|m| m.hash);
    Ok(sketch)
}

pub(crate) fn scalar_sequence_minmer_intervals(
    seq: &[u8],
    k: usize,
    window_kmers: usize,
    sketch_size: usize,
    seq_id: SeqId,
) -> Result<Vec<MinmerInterval>> {
    if seq.len() < k || window_kmers == 0 || sketch_size == 0 {
        return Ok(Vec::new());
    }

    let seq_upper = uppercase_ascii(seq);
    let seq_rev = fastani_reverse_complement(&seq_upper);
    let mut intervals = Vec::new();
    for (run_start, run_end) in acgt_runs(&seq_upper) {
        if run_end - run_start < k + window_kmers - 1 {
            continue;
        }

        let mut hashes = Vec::with_capacity(run_end - run_start - k + 1);
        for pos in run_start..=(run_end - k) {
            if let Some(hash) = canonical_murmur_hash_at(&seq_upper, &seq_rev, pos, k)? {
                hashes.push(hash);
            }
        }
        intervals.extend(minmer_intervals_for_run(
            &hashes,
            run_start,
            seq_id,
            window_kmers,
            sketch_size,
        ));
    }

    intervals.sort_unstable_by_key(|iv| (iv.seq_id, iv.start, iv.end, iv.hash));
    Ok(intervals)
}

fn minmer_intervals_for_run(
    hashes: &[HashValue],
    run_start: Offset,
    seq_id: SeqId,
    window_kmers: usize,
    sketch_size: usize,
) -> Vec<MinmerInterval> {
    if hashes.len() < window_kmers {
        return Vec::new();
    }

    let sketch_size = sketch_size.min(window_kmers);
    let last_window_start = hashes.len() - window_kmers;
    let mut state = RollingBottomSet::new(sketch_size);
    for &hash in &hashes[..window_kmers] {
        state.initialize_insert(hash);
    }
    state.finish_initialization();

    let mut intervals = Vec::new();
    let mut open = HashMap::new();
    for hash in state.low_hashes() {
        open.insert(hash, run_start);
    }

    for window_start in 1..=last_window_start {
        let event_pos = run_start + window_start;
        state.remove_hash(
            hashes[window_start - 1],
            event_pos,
            &mut open,
            &mut intervals,
            seq_id,
        );
        state.insert_hash(
            hashes[window_start + window_kmers - 1],
            event_pos,
            &mut open,
            &mut intervals,
            seq_id,
        );
    }

    let final_end = run_start + last_window_start + 1;
    for (hash, start) in open {
        if start < final_end {
            intervals.push(MinmerInterval {
                hash,
                seq_id,
                start,
                end: final_end,
            });
        }
    }
    intervals
}

struct RollingBottomSet {
    sketch_size: usize,
    counts: HashMap<HashValue, usize>,
    low: BTreeSet<HashValue>,
    high: BTreeSet<HashValue>,
}

impl RollingBottomSet {
    fn new(sketch_size: usize) -> Self {
        Self {
            sketch_size,
            counts: HashMap::new(),
            low: BTreeSet::new(),
            high: BTreeSet::new(),
        }
    }

    fn initialize_insert(&mut self, hash: HashValue) {
        *self.counts.entry(hash).or_insert(0) += 1;
    }

    fn finish_initialization(&mut self) {
        let mut unique = self.counts.keys().copied().collect::<Vec<_>>();
        unique.sort_unstable();
        for hash in unique.into_iter().take(self.sketch_size) {
            self.low.insert(hash);
        }
        for hash in self.counts.keys().copied().collect::<Vec<_>>() {
            if !self.low.contains(&hash) {
                self.high.insert(hash);
            }
        }
    }

    fn low_hashes(&self) -> impl Iterator<Item = HashValue> + '_ {
        self.low.iter().copied()
    }

    fn insert_hash(
        &mut self,
        hash: HashValue,
        pos: Offset,
        open: &mut HashMap<HashValue, Offset>,
        intervals: &mut Vec<MinmerInterval>,
        seq_id: SeqId,
    ) {
        let count = self.counts.entry(hash).or_insert(0);
        if *count > 0 {
            *count += 1;
            return;
        }
        *count = 1;

        if self.low.len() < self.sketch_size {
            self.low.insert(hash);
            open_interval(hash, pos, open);
            return;
        }

        if let Some(&max_low) = self.low.iter().next_back() {
            if hash < max_low {
                self.low.remove(&max_low);
                self.high.insert(max_low);
                close_interval(max_low, pos, open, intervals, seq_id);
                self.low.insert(hash);
                open_interval(hash, pos, open);
            } else {
                self.high.insert(hash);
            }
        } else {
            self.low.insert(hash);
            open_interval(hash, pos, open);
        }
    }

    fn remove_hash(
        &mut self,
        hash: HashValue,
        pos: Offset,
        open: &mut HashMap<HashValue, Offset>,
        intervals: &mut Vec<MinmerInterval>,
        seq_id: SeqId,
    ) {
        let Some(count) = self.counts.get_mut(&hash) else {
            return;
        };
        *count -= 1;
        if *count > 0 {
            return;
        }
        self.counts.remove(&hash);

        if self.low.remove(&hash) {
            close_interval(hash, pos, open, intervals, seq_id);
            if let Some(next_high) = self.high.iter().next().copied() {
                self.high.remove(&next_high);
                self.low.insert(next_high);
                open_interval(next_high, pos, open);
            }
        } else {
            self.high.remove(&hash);
        }
    }
}

fn open_interval(hash: HashValue, start: Offset, open: &mut HashMap<HashValue, Offset>) {
    open.entry(hash).or_insert(start);
}

fn close_interval(
    hash: HashValue,
    end: Offset,
    open: &mut HashMap<HashValue, Offset>,
    intervals: &mut Vec<MinmerInterval>,
    seq_id: SeqId,
) {
    if let Some(start) = open.remove(&hash) {
        if start < end {
            intervals.push(MinmerInterval {
                hash,
                seq_id,
                start,
                end,
            });
        }
    }
}

fn uppercase_ascii(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .map(|&base| {
            if base.is_ascii_lowercase() {
                base.to_ascii_uppercase()
            } else {
                base
            }
        })
        .collect()
}

fn acgt_runs(seq: &[u8]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start = 0usize;
    while start < seq.len() {
        while start < seq.len() && !is_acgt(seq[start]) {
            start += 1;
        }
        if start >= seq.len() {
            break;
        }
        let mut end = start;
        while end < seq.len() && is_acgt(seq[end]) {
            end += 1;
        }
        runs.push((start, end));
        start = end;
    }
    runs
}

fn canonical_murmur_hash_at(
    seq_upper: &[u8],
    seq_rev: &[u8],
    pos: usize,
    k: usize,
) -> Result<Option<HashValue>> {
    let hash_fwd = murmurhash3_x64_128_low32(&seq_upper[pos..pos + k], 42)? as HashValue;
    let rc_start = seq_upper.len() - pos - k;
    let hash_bwd = murmurhash3_x64_128_low32(&seq_rev[rc_start..rc_start + k], 42)? as HashValue;
    if hash_bwd == hash_fwd {
        Ok(None)
    } else {
        Ok(Some(hash_fwd.min(hash_bwd)))
    }
}

fn fastani_reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&base| match base {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            other => other,
        })
        .collect()
}

fn is_acgt(base: u8) -> bool {
    matches!(base, b'A' | b'C' | b'G' | b'T')
}

fn minimizer_token(canonical_kmer_value: u64, k: usize, tab_hasher: &TabulationHasher) -> u64 {
    let key = canonical_kmer_value ^ ((k as u64) << 56) ^ 0xD1B5_4A32_D192_ED03;
    tab_hasher.hash(key)
}

pub(crate) fn deterministic_tabulation_hasher(seed: u64, mode: TabulationMode) -> TabulationHasher {
    match mode {
        TabulationMode::Twisted => TabulationHasher::Twisted(deterministic_tab64_twisted(seed)),
        TabulationMode::Simple => TabulationHasher::Simple(deterministic_tab64_simple(seed)),
    }
}

fn deterministic_tab64_twisted(seed: u64) -> Tab64Twisted {
    let mut state = seed ^ 0xA076_1D64_78BD_642F;
    let mut table = [[0u128; 256]; 8];
    for row in &mut table {
        for value in row {
            let hi = splitmix64_next(&mut state) as u128;
            let lo = splitmix64_next(&mut state) as u128;
            *value = (hi << 64) | lo;
        }
    }
    Tab64Twisted::with_table(table)
}

fn deterministic_tab64_simple(seed: u64) -> Tab64Simple {
    let mut state = seed ^ 0xA076_1D64_78BD_642F;
    let mut table = [[0u64; 256]; 8];
    for row in &mut table {
        for value in row {
            *value = splitmix64_next(&mut state);
        }
    }
    Tab64Simple::with_table(table)
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    splitmix64_permute(*state)
}

#[cfg(test)]
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    splitmix64_permute(x)
}

pub(crate) fn splitmix64_permute(x: u64) -> u64 {
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn murmurhash3_x64_128_low32(key: &[u8], seed: u32) -> Result<u32> {
    Ok(murmur3_x64_128(&mut Cursor::new(key), seed)? as u32)
}
