use std::time::Instant;

use anyhow::Result;

use crate::candidate_window::L1Candidate;
use crate::simd_minimizer::{Minimizer, MinimizerMode};
use crate::{AniConfig, HashValue, Offset, QuerySketch, ReferenceIndex, SeqId};

#[derive(Debug, Clone)]
pub(crate) struct MappingResult {
    pub(crate) query_seq_id: usize,
    pub(crate) query_len: usize,
    pub(crate) ref_seq_id: SeqId,
    pub(crate) ref_start: Offset,
    pub(crate) ref_end: Offset,
    pub(crate) identity: f64,
    pub(crate) identity_upper_bound: f64,
    pub(crate) conserved_sketches: usize,
    pub(crate) sketch_size: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct L2Stats {
    pub(crate) windows: u64,
    pub(crate) ref_sketches: u64,
    pub(crate) coord_count: usize,
    pub(crate) reference_minimizers: usize,
    pub(crate) ref_hash_ns: u128,
    pub(crate) ref_sketch_ns: u128,
    pub(crate) distance_ns: u128,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct IndexedMinimizer {
    pub(crate) wpos: Offset,
    pub(crate) coord_idx: usize,
}

/// Reusable buffers for one L2 candidate.
///
/// Nothing in here carries across candidates, so one instance is threaded down
/// from `map_fragment` and reset per candidate rather than rebuilt.
#[derive(Debug, Default)]
pub(crate) struct L2Scratch {
    coords: Vec<HashValue>,
    indexed: Vec<IndexedMinimizer>,
    ref_by_hash: Vec<(HashValue, u32)>,
    query_slots: Vec<u32>,
    mapper: BitsetBottomSketchSlideMapper,
}

impl L2Scratch {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

// Maintains the classic bottom-k union sketch exactly as an L2
// reference window slides
#[derive(Debug, Default)]
pub(crate) struct BitsetBottomSketchSlideMapper {
    query_present: Vec<u8>,
    ref_count: Vec<u32>,
    union_bits: SummaryBitSet,
    pivot_idx: usize,
    shared: usize,
}

impl BitsetBottomSketchSlideMapper {
    /// Point the mapper at a new coordinate universe, reusing the existing
    /// allocations. `query_slots` holds the coordinate index of each query
    /// hash, as produced by [`build_indexed_minimizers_into`]; it must be
    /// non-empty and ascending. Every element in use is overwritten, so no
    /// state leaks between candidates.
    pub(crate) fn reset(&mut self, coord_len: usize, query_slots: &[u32]) {
        debug_assert!(!query_slots.is_empty());
        debug_assert!(query_slots.windows(2).all(|w| w[0] < w[1]));

        self.query_present.clear();
        self.query_present.resize(coord_len, 0);
        self.ref_count.clear();
        self.ref_count.resize(coord_len, 0);
        self.union_bits.reset(coord_len);
        self.shared = 0;

        for &slot in query_slots {
            self.query_present[slot as usize] = 1;
            self.union_bits.set(slot as usize);
        }

        // The pivot starts at the largest query hash, i.e. the last slot.
        self.pivot_idx = query_slots[query_slots.len() - 1] as usize;
    }

    /// Construct a mapper for a single universe. Test-facing convenience; the
    /// mapping pipeline reuses one mapper via [`Self::reset`].
    #[cfg(test)]
    pub(crate) fn for_universe(query_hashes: &[HashValue], coords: &[HashValue]) -> Self {
        let slots = query_hashes
            .iter()
            .map(|hash| {
                coords
                    .binary_search(hash)
                    .expect("query hash must be in coordinate universe") as u32
            })
            .collect::<Vec<_>>();
        let mut mapper = Self::default();
        mapper.reset(coords.len(), &slots);
        mapper
    }

    #[inline]
    pub(crate) fn shared(&self) -> usize {
        self.shared
    }

    #[inline]
    fn insert_ref_range(&mut self, minimizers: &[IndexedMinimizer]) {
        for minimizer in minimizers {
            self.insert_ref(minimizer.coord_idx);
        }
    }

    #[inline]
    pub(crate) fn insert_ref(&mut self, idx: usize) {
        if self.ref_count[idx] > 0 {
            self.ref_count[idx] += 1;
            return;
        }

        self.ref_count[idx] = 1;
        if self.query_present[idx] != 0 {
            if idx <= self.pivot_idx {
                self.shared += 1;
            }
        } else {
            self.union_bits.set(idx);
            if idx <= self.pivot_idx {
                if self.is_shared_index(self.pivot_idx) {
                    self.shared = self.shared.saturating_sub(1);
                }
                self.pivot_idx = self
                    .union_bits
                    .prev_set_before(self.pivot_idx)
                    .expect("bottom-k pivot must have a predecessor after insert");
            }
        }
    }

    #[inline]
    pub(crate) fn delete_ref(&mut self, idx: usize) {
        let Some(next_count) = self.ref_count[idx].checked_sub(1) else {
            return;
        };
        self.ref_count[idx] = next_count;
        if next_count > 0 {
            return;
        }

        if self.query_present[idx] != 0 {
            if idx <= self.pivot_idx {
                self.shared = self.shared.saturating_sub(1);
            }
        } else {
            self.union_bits.clear(idx);
            if idx <= self.pivot_idx {
                self.pivot_idx = self
                    .union_bits
                    .next_set_after(self.pivot_idx)
                    .expect("bottom-k pivot must have a successor after delete");
                if self.is_shared_index(self.pivot_idx) {
                    self.shared += 1;
                }
            }
        }
    }

    #[inline]
    fn is_shared_index(&self, idx: usize) -> bool {
        self.query_present[idx] != 0 && self.ref_count[idx] > 0
    }
}

#[derive(Debug, Default)]
pub(crate) struct SummaryBitSet {
    words: Vec<u64>,
    summary: Vec<u64>,
}

impl SummaryBitSet {
    #[cfg(test)]
    pub(crate) fn new(len: usize) -> Self {
        let mut bits = Self::default();
        bits.reset(len);
        bits
    }

    /// Resize to `len` bits, all clear, reusing the existing allocation.
    pub(crate) fn reset(&mut self, len: usize) {
        let word_count = len.div_ceil(64);
        let summary_count = word_count.div_ceil(64);
        self.words.clear();
        self.words.resize(word_count, 0);
        self.summary.clear();
        self.summary.resize(summary_count, 0);
    }

    #[inline]
    pub(crate) fn set(&mut self, index: usize) {
        let word_idx = index >> 6;
        let mask = 1u64 << (index & 63);
        let old = self.words[word_idx];
        self.words[word_idx] = old | mask;
        if old == 0 {
            self.summary[word_idx >> 6] |= 1u64 << (word_idx & 63);
        }
    }

    #[inline]
    pub(crate) fn clear(&mut self, index: usize) {
        let word_idx = index >> 6;
        let mask = 1u64 << (index & 63);
        self.words[word_idx] &= !mask;
        if self.words[word_idx] == 0 {
            self.summary[word_idx >> 6] &= !(1u64 << (word_idx & 63));
        }
    }

    #[inline]
    pub(crate) fn prev_set_before(&self, index: usize) -> Option<usize> {
        if index == 0 {
            return None;
        }

        let target = index - 1;
        let mut word_idx = target >> 6;
        let bit = target & 63;
        let mask = if bit == 63 {
            u64::MAX
        } else {
            (1u64 << (bit + 1)) - 1
        };
        let word = self.words[word_idx] & mask;
        if word != 0 {
            return Some((word_idx << 6) + highest_set_bit(word));
        }

        if word_idx == 0 {
            return None;
        }
        word_idx -= 1;
        let mut summary_idx = word_idx >> 6;
        let summary_bit = word_idx & 63;
        let summary_mask = if summary_bit == 63 {
            u64::MAX
        } else {
            (1u64 << (summary_bit + 1)) - 1
        };
        let mut summary_word = self.summary[summary_idx] & summary_mask;

        loop {
            if summary_word != 0 {
                let nonzero_word_idx = (summary_idx << 6) + highest_set_bit(summary_word);
                let word = self.words[nonzero_word_idx];
                return Some((nonzero_word_idx << 6) + highest_set_bit(word));
            }
            if summary_idx == 0 {
                return None;
            }
            summary_idx -= 1;
            summary_word = self.summary[summary_idx];
        }
    }

    #[inline]
    pub(crate) fn next_set_after(&self, index: usize) -> Option<usize> {
        let target = index + 1;
        let mut word_idx = target >> 6;
        if word_idx >= self.words.len() {
            return None;
        }

        let bit = target & 63;
        let word = self.words[word_idx] & (u64::MAX << bit);
        if word != 0 {
            return Some((word_idx << 6) + word.trailing_zeros() as usize);
        }

        word_idx += 1;
        if word_idx >= self.words.len() {
            return None;
        }

        let mut summary_idx = word_idx >> 6;
        let summary_bit = word_idx & 63;
        let mut summary_word = self.summary[summary_idx] & (u64::MAX << summary_bit);

        loop {
            if summary_word != 0 {
                let nonzero_word_idx = (summary_idx << 6) + summary_word.trailing_zeros() as usize;
                let word = self.words[nonzero_word_idx];
                return Some((nonzero_word_idx << 6) + word.trailing_zeros() as usize);
            }
            summary_idx += 1;
            if summary_idx >= self.summary.len() {
                return None;
            }
            summary_word = self.summary[summary_idx];
        }
    }
}

#[inline]
fn highest_set_bit(word: u64) -> usize {
    63 - word.leading_zeros() as usize
}

/// Build the local coordinate universe and index the reference minimizers into
/// it, writing into caller-owned buffers so allocations are reused across
/// candidates. Every buffer is fully overwritten.
///
/// `query_hashes` must already be sorted and deduplicated — `map_fragment`
/// guarantees this. That lets the universe be produced by a merge rather than
/// a sort of everything, and lets each element's final index be handed out
/// during that merge. The previous version sorted the combined universe and
/// then binary-searched it once per reference minimizer and once per query
/// hash; profiling showed those ~710 searches per candidate were the dominant
/// cost of L2 setup.
///
/// `query_slots` receives the coordinate index of each query hash in
/// `query_hashes` order, so the slide mapper need not search for them either.
///
/// `coords` is identical to sorting the union directly: both are the ascending
/// unique union of the same multiset.
pub(crate) fn build_indexed_minimizers_into(
    query_hashes: &[HashValue],
    reference_universe: &[Minimizer],
    coords: &mut Vec<HashValue>,
    indexed: &mut Vec<IndexedMinimizer>,
    ref_by_hash: &mut Vec<(HashValue, u32)>,
    query_slots: &mut Vec<u32>,
) {
    debug_assert!(
        query_hashes.windows(2).all(|w| w[0] < w[1]),
        "query hashes must be sorted and deduplicated"
    );

    ref_by_hash.clear();
    ref_by_hash.reserve(reference_universe.len());
    ref_by_hash.extend(
        reference_universe
            .iter()
            .enumerate()
            .map(|(i, m)| (m.hash, i as u32)),
    );
    ref_by_hash.sort_unstable();

    indexed.clear();
    indexed.resize(
        reference_universe.len(),
        IndexedMinimizer {
            wpos: 0,
            coord_idx: 0,
        },
    );
    for (slot, minimizer) in indexed.iter_mut().zip(reference_universe) {
        slot.wpos = minimizer.wpos;
    }

    coords.clear();
    coords.reserve(query_hashes.len() + reference_universe.len());
    query_slots.clear();
    query_slots.reserve(query_hashes.len());

    let mut qi = 0usize;
    let mut ri = 0usize;
    while qi < query_hashes.len() || ri < ref_by_hash.len() {
        let take_query = ri >= ref_by_hash.len()
            || (qi < query_hashes.len() && query_hashes[qi] <= ref_by_hash[ri].0);
        let hash = if take_query {
            query_hashes[qi]
        } else {
            ref_by_hash[ri].0
        };

        let coord_idx = coords.len();
        coords.push(hash);
        if take_query {
            query_slots.push(coord_idx as u32);
            qi += 1;
        }
        // Every reference minimizer carrying this hash maps to the same slot.
        while ri < ref_by_hash.len() && ref_by_hash[ri].0 == hash {
            indexed[ref_by_hash[ri].1 as usize].coord_idx = coord_idx;
            ri += 1;
        }
    }
}

#[cfg(test)]
pub(crate) fn build_indexed_minimizers(
    query_hashes: &[HashValue],
    reference_universe: &[Minimizer],
) -> (Vec<HashValue>, Vec<IndexedMinimizer>) {
    let mut coords = Vec::new();
    let mut indexed = Vec::new();
    let mut ref_by_hash = Vec::new();
    let mut query_slots = Vec::new();
    build_indexed_minimizers_into(
        query_hashes,
        reference_universe,
        &mut coords,
        &mut indexed,
        &mut ref_by_hash,
        &mut query_slots,
    );
    (coords, indexed)
}

pub(crate) fn do_l2_mapping(
    query: &QuerySketch,
    candidate: L1Candidate,
    reference: &ReferenceIndex,
    config: &AniConfig,
    window_size: usize,
    scratch: &mut L2Scratch,
) -> Result<(Option<MappingResult>, L2Stats)> {
    if config.minimizer_mode == MinimizerMode::ScalarMinmer {
        return do_l2_mapping_minmer_intervals(query, candidate, reference, config);
    }
    do_l2_mapping_bitset_exact(query, candidate, reference, config, window_size, scratch)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MinmerL2Event {
    pos: Offset,
    is_start: bool,
    coord_idx: usize,
}

fn do_l2_mapping_minmer_intervals(
    query: &QuerySketch,
    candidate: L1Candidate,
    reference: &ReferenceIndex,
    config: &AniConfig,
) -> Result<(Option<MappingResult>, L2Stats)> {
    let mut stats = L2Stats::default();
    let Some(window_kmers) = query.len.checked_sub(config.kmer_size).map(|v| v + 1) else {
        return Ok((None, stats));
    };
    if window_kmers == 0
        || query.unique_hashes.is_empty()
        || candidate.range_end < candidate.range_start
    {
        return Ok((None, stats));
    }

    let contig_range = reference.contig_minmer_bounds(candidate.seq_id);
    if contig_range.is_empty() {
        return Ok((None, stats));
    }

    let intervals = &reference.minmer_intervals[contig_range];
    let candidate_start = candidate.range_start;
    let candidate_end_exclusive = candidate.range_end.saturating_add(1);
    let lower_start = candidate_start.saturating_sub(window_kmers);
    let scan_start = intervals.partition_point(|iv| iv.start < lower_start);

    let mut overlapping = Vec::new();
    for interval in &intervals[scan_start..] {
        if interval.start >= candidate_end_exclusive {
            break;
        }
        if interval.end <= candidate_start {
            continue;
        }
        overlapping.push(*interval);
    }

    if overlapping.is_empty() {
        return Ok((None, stats));
    }

    let mut coords = Vec::with_capacity(query.unique_hashes.len() + overlapping.len());
    coords.extend_from_slice(&query.unique_hashes);
    coords.extend(overlapping.iter().map(|iv| iv.hash));
    coords.sort_unstable();
    coords.dedup();
    stats.coord_count = coords.len();
    stats.reference_minimizers = overlapping.len();
    stats.windows = (candidate_end_exclusive - candidate_start) as u64;

    // The minmer path is the FastANI-parity experiment, not the hot path, so
    // it keeps its own short-lived mapper rather than sharing the scratch.
    let mut slide_map = BitsetBottomSketchSlideMapper::default();
    let query_slots = query
        .unique_hashes
        .iter()
        .map(|hash| {
            coords
                .binary_search(hash)
                .expect("query hash must be in coordinate universe") as u32
        })
        .collect::<Vec<_>>();
    slide_map.reset(coords.len(), &query_slots);
    let mut events = Vec::new();
    for interval in overlapping {
        let coord_idx = coords
            .binary_search(&interval.hash)
            .expect("minmer interval hash must be in local coordinate universe");
        if interval.start <= candidate_start && interval.end > candidate_start {
            slide_map.insert_ref(coord_idx);
        } else if interval.start > candidate_start && interval.start < candidate_end_exclusive {
            events.push(MinmerL2Event {
                pos: interval.start,
                is_start: true,
                coord_idx,
            });
        }
        if interval.end > candidate_start && interval.end < candidate_end_exclusive {
            events.push(MinmerL2Event {
                pos: interval.end,
                is_start: false,
                coord_idx,
            });
        }
    }
    events.sort_unstable();

    let mut best_shared = 0usize;
    let mut first_best_pos: Option<Offset> = None;
    let mut last_best_pos: Option<Offset> = None;
    update_minmer_best(
        &slide_map,
        candidate_start,
        next_event_or_end(&events, 0, candidate_end_exclusive).saturating_sub(1),
        &mut best_shared,
        &mut first_best_pos,
        &mut last_best_pos,
    );

    let mut idx = 0usize;
    while idx < events.len() {
        let pos = events[idx].pos;
        while idx < events.len() && events[idx].pos == pos {
            if events[idx].is_start {
                slide_map.insert_ref(events[idx].coord_idx);
            } else {
                slide_map.delete_ref(events[idx].coord_idx);
            }
            idx += 1;
        }
        let span_end = next_event_or_end(&events, idx, candidate_end_exclusive).saturating_sub(1);
        update_minmer_best(
            &slide_map,
            pos,
            span_end,
            &mut best_shared,
            &mut first_best_pos,
            &mut last_best_pos,
        );
    }

    if first_best_pos.is_none() {
        return Ok((None, stats));
    }

    let distance_start = Instant::now();
    let distance = query
        .distance_table
        .get(best_shared)
        .expect("best shared count must fit query distance table");
    let identity = distance.identity;
    let identity_upper_bound = distance.identity_upper_bound;
    stats.distance_ns += distance_start.elapsed().as_nanos();

    if identity_upper_bound < config.min_identity {
        return Ok((None, stats));
    }

    let first = first_best_pos.unwrap();
    let last = last_best_pos.unwrap_or(first);
    let mean_start = (first + last) / 2;

    Ok((
        Some(MappingResult {
            query_seq_id: query.fragment_id,
            query_len: query.len,
            ref_seq_id: candidate.seq_id,
            ref_start: mean_start,
            ref_end: mean_start + query.len - 1,
            identity,
            identity_upper_bound,
            conserved_sketches: best_shared,
            sketch_size: query.unique_hashes.len(),
        }),
        stats,
    ))
}

fn next_event_or_end(events: &[MinmerL2Event], idx: usize, end: Offset) -> Offset {
    events.get(idx).map(|event| event.pos).unwrap_or(end)
}

fn update_minmer_best(
    slide_map: &BitsetBottomSketchSlideMapper,
    span_start: Offset,
    span_end: Offset,
    best_shared: &mut usize,
    first_best_pos: &mut Option<Offset>,
    last_best_pos: &mut Option<Offset>,
) {
    if span_end < span_start {
        return;
    }
    let shared = slide_map.shared();
    if shared > *best_shared {
        *best_shared = shared;
        *first_best_pos = Some(span_start);
        *last_best_pos = Some(span_end);
    } else if shared == *best_shared {
        if first_best_pos.is_none() {
            *first_best_pos = Some(span_start);
        }
        *last_best_pos = Some(span_end);
    }
}

fn do_l2_mapping_bitset_exact(
    query: &QuerySketch,
    candidate: L1Candidate,
    reference: &ReferenceIndex,
    config: &AniConfig,
    window_size: usize,
    scratch: &mut L2Scratch,
) -> Result<(Option<MappingResult>, L2Stats)> {
    let mut stats = L2Stats::default();
    let count_minimizer_windows = query
        .len
        .checked_sub(window_size.saturating_sub(1))
        .and_then(|v| v.checked_sub(config.kmer_size.saturating_sub(1)));
    let Some(count_minimizer_windows) = count_minimizer_windows else {
        return Ok((None, stats));
    };
    if count_minimizer_windows == 0 || query.unique_hashes.is_empty() {
        return Ok((None, stats));
    }

    let contig_range = reference.contig_minimizer_bounds(candidate.seq_id);
    if contig_range.is_empty() {
        return Ok((None, stats));
    }

    let sw_beg_abs = reference.lower_bound(candidate.seq_id, candidate.range_start);
    if sw_beg_abs >= contig_range.end {
        return Ok((None, stats));
    }

    let sw_pos_abs = reference.minimizers[sw_beg_abs].wpos;
    let sw_end_abs = reference.lower_bound(
        candidate.seq_id,
        sw_pos_abs.saturating_add(count_minimizer_windows),
    );
    let last_end_abs = reference.lower_bound(
        candidate.seq_id,
        candidate.range_end.saturating_add(query.len),
    );
    let coord_end_abs = last_end_abs.min(contig_range.end);
    if coord_end_abs <= sw_beg_abs {
        return Ok((None, stats));
    }

    // Destructure so the coordinate universe can be read while the mapper is
    // borrowed mutably -- they are disjoint fields of the same scratch.
    let L2Scratch {
        coords,
        indexed: local_minimizers,
        ref_by_hash,
        query_slots,
        mapper: slide_map,
    } = scratch;
    build_indexed_minimizers_into(
        &query.unique_hashes,
        &reference.minimizers[sw_beg_abs..coord_end_abs],
        coords,
        local_minimizers,
        ref_by_hash,
        query_slots,
    );
    stats.coord_count = coords.len();
    stats.reference_minimizers = local_minimizers.len();

    let mut sw_beg = 0usize;
    let mut sw_end = sw_end_abs
        .saturating_sub(sw_beg_abs)
        .min(local_minimizers.len());
    let last_end = last_end_abs
        .saturating_sub(sw_beg_abs)
        .min(local_minimizers.len());
    let mut sw_pos = local_minimizers[sw_beg].wpos;

    slide_map.reset(coords.len(), query_slots);
    slide_map.insert_ref_range(&local_minimizers[sw_beg..sw_end.min(local_minimizers.len())]);
    let mut prev_beg = sw_beg;
    let mut prev_end = sw_end;

    let mut best_shared = 0usize;
    let mut first_best_pos: Option<Offset> = None;
    let mut last_best_pos: Option<Offset> = None;

    while sw_end < last_end && sw_beg < local_minimizers.len() && sw_pos <= candidate.range_end {
        if prev_beg != sw_beg {
            slide_map.delete_ref(local_minimizers[prev_beg].coord_idx);
        }
        if prev_end != sw_end {
            slide_map.insert_ref(local_minimizers[prev_end].coord_idx);
        }

        stats.windows += 1;
        sw_end = sw_end.min(local_minimizers.len());
        let shared = slide_map.shared();

        if shared > best_shared {
            best_shared = shared;
            first_best_pos = Some(local_minimizers[sw_beg].wpos);
            last_best_pos = first_best_pos;
        } else if shared == best_shared {
            last_best_pos = Some(local_minimizers[sw_beg].wpos);
        }

        let begin_pos = sw_pos;
        let last_pos = sw_pos + count_minimizer_windows - 1;
        let next_beg_delta = if sw_beg + 1 < local_minimizers.len() {
            local_minimizers[sw_beg + 1].wpos.saturating_sub(begin_pos)
        } else {
            usize::MAX
        };
        let next_end_delta = if sw_end < local_minimizers.len() {
            local_minimizers[sw_end].wpos.saturating_sub(last_pos)
        } else {
            usize::MAX
        };
        let advance_by = next_beg_delta.min(next_end_delta);
        if advance_by == 0 || advance_by == usize::MAX {
            break;
        }

        prev_beg = sw_beg;
        prev_end = sw_end;
        if advance_by == next_beg_delta {
            sw_beg += 1;
        }
        if advance_by == next_end_delta {
            sw_end += 1;
        }
        sw_pos += advance_by;
    }

    if first_best_pos.is_none() {
        return Ok((None, stats));
    }

    let distance_start = Instant::now();
    let sketch_size = query.unique_hashes.len();
    let distance = query
        .distance_table
        .get(best_shared)
        .expect("best shared count must fit query distance table");
    let identity = distance.identity;
    let identity_upper_bound = distance.identity_upper_bound;
    stats.distance_ns += distance_start.elapsed().as_nanos();

    if identity_upper_bound < config.min_identity {
        return Ok((None, stats));
    }

    let first = first_best_pos.unwrap();
    let last = last_best_pos.unwrap_or(first);
    let mean_start = (first + last) / 2;

    Ok((
        Some(MappingResult {
            query_seq_id: query.fragment_id,
            query_len: query.len,
            ref_seq_id: candidate.seq_id,
            ref_start: mean_start,
            ref_end: mean_start + query.len - 1,
            identity,
            identity_upper_bound,
            conserved_sketches: best_shared,
            sketch_size,
        }),
        stats,
    ))
}
