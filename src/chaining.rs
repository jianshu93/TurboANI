use crate::{
    FastAniConfig, L1Candidate, L1Stats, Offset, QuerySketch, ReferenceIndex, SeqId,
    estimate_minimum_hits_relaxed,
};

const DIAG_CLUSTER_BIN: usize = 1000;
const DIAG_CLUSTER_BAND: usize = 500;
const CHAIN_CANDIDATE_BAND: usize = 1000;
const CHAIN_MIN_BOUND: i64 = 100;
const CHAIN_RAMP_UP_FACTOR: i64 = 4;
const CHAIN_MAX_REVISIONS: usize = 32;
const INF: i64 = i64::MAX / 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SeedAnchorHit {
    seq_id: SeqId,
    ref_start: Offset,
    query_start: Offset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DiagonalClusterHit {
    seq_id: u32,
    diag_bin: i32,
    query_start: u32,
    start_est: u32,
}

impl DiagonalClusterHit {
    fn new(seq_id: SeqId, diag_bin: isize, query_start: Offset, start_est: Offset) -> Self {
        debug_assert!(u32::try_from(seq_id).is_ok());
        debug_assert!(i32::try_from(diag_bin).is_ok());
        debug_assert!(u32::try_from(query_start).is_ok());
        debug_assert!(u32::try_from(start_est).is_ok());
        Self {
            seq_id: seq_id as u32,
            diag_bin: diag_bin as i32,
            query_start: query_start as u32,
            start_est: start_est as u32,
        }
    }

    fn key(self) -> (u32, i32) {
        (self.seq_id, self.diag_bin)
    }
}

struct DiagonalGroupScratch {
    seen_query_stamp: Vec<u32>,
    stamp: u32,
}

impl DiagonalGroupScratch {
    fn new(query_len: usize) -> Self {
        Self {
            seen_query_stamp: vec![0; query_len.saturating_add(1)],
            stamp: 0,
        }
    }

    fn next_stamp(&mut self) -> u32 {
        if self.stamp == u32::MAX {
            self.seen_query_stamp.fill(0);
            self.stamp = 1;
        } else {
            self.stamp += 1;
        }
        self.stamp
    }

    fn has_unique_query_support(
        &mut self,
        group: &[DiagonalClusterHit],
        min_unique_hits: usize,
    ) -> bool {
        if group.len() < min_unique_hits {
            return false;
        }

        let stamp = self.next_stamp();
        let mut unique = 0usize;
        for hit in group {
            let idx = hit.query_start as usize;
            debug_assert!(idx < self.seen_query_stamp.len());
            if self.seen_query_stamp[idx] != stamp {
                self.seen_query_stamp[idx] = stamp;
                unique += 1;
                if unique >= min_unique_hits {
                    return true;
                }
            }
        }
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AnchorHit {
    ref_start: Offset,
    query_start: Offset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChainOrientation {
    Forward,
    Reverse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Anchor {
    ref_start: i64,
    query_start: i64,
    len: i64,
    hit: Option<AnchorHit>,
}

impl Anchor {
    fn ref_end(self) -> i64 {
        self.ref_start + self.len - 1
    }

    fn query_end(self) -> i64 {
        self.query_start + self.len - 1
    }

    fn diagonal(self) -> i64 {
        self.ref_start - self.query_start
    }
}

#[derive(Debug, Clone, Copy)]
struct ChainSetup {
    candidate_band: usize,
    min_unique_hits: usize,
}

impl ChainSetup {
    fn new(min_unique_hits: usize) -> Self {
        Self {
            candidate_band: CHAIN_CANDIDATE_BAND,
            min_unique_hits,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Event {
    pos: i64,
    is_start: bool,
    anchor_idx: usize,
}

pub(crate) fn do_l1_mapping_diagonal_clustered(
    query: &QuerySketch,
    reference: &ReferenceIndex,
    config: &FastAniConfig,
) -> (Vec<L1Candidate>, L1Stats) {
    let min_unique_hits = estimate_minimum_hits_relaxed(
        query.unique_hashes.len(),
        config.kmer_size,
        config.min_identity,
    )
    .max(1);

    let mut hits = Vec::new();
    for seed in &query.unique_seeds {
        if let Some(ref_hits) = reference.lookup.get(seed.hash) {
            if ref_hits.len() < reference.freq_threshold {
                hits.reserve(ref_hits.len());
                for &hit in ref_hits {
                    let wpos = hit.wpos();
                    let diagonal = wpos as isize - seed.qpos as isize;
                    hits.push(DiagonalClusterHit::new(
                        hit.seq_id(),
                        div_floor_isize(diagonal, DIAG_CLUSTER_BIN as isize),
                        seed.qpos,
                        wpos.saturating_sub(seed.qpos),
                    ));
                }
            }
        }
    }

    let stats = L1Stats {
        seed_hits: hits.len(),
    };
    if hits.len() < min_unique_hits {
        return (Vec::new(), stats);
    }

    hits.sort_unstable_by_key(|hit| hit.key());

    let mut candidates = Vec::new();
    let mut scratch = DiagonalGroupScratch::new(query.len);
    let mut group_start = 0usize;
    while group_start < hits.len() {
        let key = hits[group_start].key();
        let mut group_end = group_start + 1;
        while group_end < hits.len() && hits[group_end].key() == key {
            group_end += 1;
        }

        let group = &mut hits[group_start..group_end];
        if scratch.has_unique_query_support(group, min_unique_hits) {
            let start = median_start_est(group);
            candidates.push(L1Candidate {
                seq_id: key.0 as usize,
                range_start: start.saturating_sub(DIAG_CLUSTER_BAND),
                range_end: start.saturating_add(DIAG_CLUSTER_BAND),
            });
        }

        group_start = group_end;
    }

    merge_candidates(&mut candidates);
    (candidates, stats)
}

// Optimal chaining algorithm from: Rizzo, N., Cáceres, M. and Mäkinen, V., 2025, August. Practical colinear chaining on sequences revisited. In International Symposium on Bioinformatics Research and Applications (pp. 203-216). Singapore: Springer Nature Singapore.
pub(crate) fn do_l1_mapping_chained(
    query: &QuerySketch,
    reference: &ReferenceIndex,
    config: &FastAniConfig,
) -> (Vec<L1Candidate>, L1Stats) {
    let min_unique_hits = estimate_minimum_hits_relaxed(
        query.unique_hashes.len(),
        config.kmer_size,
        config.min_identity,
    )
    .max(1);
    let setup = ChainSetup::new(min_unique_hits);

    let mut hits = Vec::new();
    for seed in &query.unique_seeds {
        if let Some(ref_hits) = reference.lookup.get(seed.hash) {
            if ref_hits.len() < reference.freq_threshold {
                hits.reserve(ref_hits.len());
                for &hit in ref_hits {
                    hits.push(SeedAnchorHit {
                        seq_id: hit.seq_id(),
                        ref_start: hit.wpos(),
                        query_start: seed.qpos,
                    });
                }
            }
        }
    }

    let stats = L1Stats {
        seed_hits: hits.len(),
    };
    if hits.len() < setup.min_unique_hits {
        return (Vec::new(), stats);
    }

    hits.sort_unstable_by_key(|hit| (hit.seq_id, hit.ref_start, hit.query_start));

    let mut candidates = Vec::new();
    let mut group_start = 0usize;
    while group_start < hits.len() {
        let seq_id = hits[group_start].seq_id;
        let mut group_end = group_start + 1;
        while group_end < hits.len() && hits[group_end].seq_id == seq_id {
            group_end += 1;
        }

        let group = &hits[group_start..group_end];
        if group.len() >= setup.min_unique_hits {
            let contig_len = reference.contigs[seq_id].len;
            let forward_chain = chainx_semiglobal_supported_hits(
                group,
                query.len,
                contig_len,
                config.kmer_size,
                setup.min_unique_hits,
                ChainOrientation::Forward,
            );
            append_chain_candidates(seq_id, &forward_chain, &setup, &mut candidates);

            let reverse_chain = chainx_semiglobal_supported_hits(
                group,
                query.len,
                contig_len,
                config.kmer_size,
                setup.min_unique_hits,
                ChainOrientation::Reverse,
            );
            append_chain_candidates(seq_id, &reverse_chain, &setup, &mut candidates);
        }

        group_start = group_end;
    }

    merge_candidates(&mut candidates);
    (candidates, stats)
}

fn append_chain_candidates(
    seq_id: SeqId,
    chain: &[AnchorHit],
    setup: &ChainSetup,
    candidates: &mut Vec<L1Candidate>,
) {
    if unique_query_positions(chain) < setup.min_unique_hits {
        return;
    }

    let mut starts = chain
        .iter()
        .map(|hit| hit.ref_start.saturating_sub(hit.query_start))
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();

    for start in starts {
        candidates.push(L1Candidate {
            seq_id,
            range_start: start.saturating_sub(setup.candidate_band),
            range_end: start.saturating_add(setup.candidate_band),
        });
    }
}

fn unique_query_positions(chain: &[AnchorHit]) -> usize {
    let mut positions = chain.iter().map(|hit| hit.query_start).collect::<Vec<_>>();
    positions.sort_unstable();
    positions.dedup();
    positions.len()
}

fn median_start_est(group: &mut [DiagonalClusterHit]) -> Offset {
    let mid = group.len() / 2;
    let (_, median, _) = group.select_nth_unstable_by_key(mid, |hit| hit.start_est);
    median.start_est as usize
}

fn chainx_semiglobal_supported_hits(
    hits: &[SeedAnchorHit],
    query_len: usize,
    ref_len: usize,
    anchor_len: usize,
    min_chain_hits: usize,
    orientation: ChainOrientation,
) -> Vec<AnchorHit> {
    if hits.is_empty() {
        return Vec::new();
    }

    let anchors = build_anchors(hits, query_len, ref_len, anchor_len, orientation);
    if anchors.len() <= 2 {
        return Vec::new();
    }

    let events = sorted_events(&anchors);
    let (diagonal_idx, diagonal_values) = diagonal_buckets(&anchors);
    let initial_bound = initial_distance_bound(&anchors, query_len);
    let end_idx = anchors.len() - 1;

    let mut bound = initial_bound;
    for _ in 0..=CHAIN_MAX_REVISIONS {
        let mut costs = vec![INF; anchors.len()];
        let mut chain_lens = vec![0usize; anchors.len()];
        let mut predecessors = vec![None; anchors.len()];
        costs[0] = 0;

        sweep_chainx_semiglobal(
            &anchors,
            &events,
            &diagonal_idx,
            &diagonal_values,
            bound,
            &mut costs,
            &mut chain_lens,
            &mut predecessors,
        );

        if costs[end_idx] <= bound || bound > query_len.max(ref_len) as i64 * 4 {
            let supported =
                chain_supported_hits(&anchors, &costs, &chain_lens, &predecessors, min_chain_hits);
            if supported.is_empty() {
                return traceback_chain(&anchors, &predecessors);
            }
            return supported;
        }

        // Recompute with a looser admissible gap/diagonal bound, matching the
        // ChainX-opt paper implementation's expanding-bound strategy.
        bound = bound.saturating_mul(CHAIN_RAMP_UP_FACTOR);
        if bound >= INF / CHAIN_RAMP_UP_FACTOR {
            return traceback_chain(&anchors, &predecessors);
        }
    }

    Vec::new()
}

#[allow(clippy::too_many_arguments)]
fn sweep_chainx_semiglobal(
    anchors: &[Anchor],
    events: &[Event],
    diagonal_idx: &[usize],
    diagonal_values: &[i64],
    bound: i64,
    costs: &mut [i64],
    chain_lens: &mut [usize],
    predecessors: &mut [Option<usize>],
) {
    let end_idx = anchors.len() - 1;
    let mut active_anchor = vec![None; diagonal_values.len()];
    let mut inner_loop_start = 0usize;

    for (event_pos, event) in events.iter().copied().enumerate() {
        if event.anchor_idx == 0 {
            continue;
        }

        if !event.is_start {
            let diag = diagonal_idx[event.anchor_idx];
            if active_anchor[diag] == Some(event.anchor_idx) {
                active_anchor[diag] = None;
            }
            continue;
        }

        let anchor_idx = event.anchor_idx;
        let anchor = anchors[anchor_idx];
        let diag = diagonal_idx[anchor_idx];
        let curr_diagonal = anchor.diagonal();
        let adds_anchor = anchor_idx != end_idx;
        let mut best_cost = INF;
        let mut best_len = 0usize;
        let mut best_pred = None;

        if let Some(prev) = active_anchor[diag] {
            consider_transition(
                prev,
                0,
                adds_anchor,
                costs,
                chain_lens,
                &mut best_cost,
                &mut best_len,
                &mut best_pred,
            );
        }
        active_anchor[diag] = Some(anchor_idx);

        while inner_loop_start < event_pos {
            let previous = events[inner_loop_start];
            if previous.is_start {
                inner_loop_start += 1;
                continue;
            }

            let previous_anchor = anchors[previous.anchor_idx];
            if anchor.ref_start - previous_anchor.ref_end() - 1 > bound {
                inner_loop_start += 1;
            } else {
                break;
            }
        }

        let first_query_gap = anchor.query_start - anchors[0].query_end() - 1;
        consider_transition(
            0,
            first_query_gap.max(0),
            adds_anchor,
            costs,
            chain_lens,
            &mut best_cost,
            &mut best_len,
            &mut best_pred,
        );

        let gap_scan_start = if anchor_idx == end_idx {
            0
        } else {
            inner_loop_start
        };

        for scan_pos in (gap_scan_start..event_pos).rev() {
            let previous = events[scan_pos];
            if previous.is_start {
                continue;
            }

            let prev_idx = previous.anchor_idx;
            let prev_anchor = anchors[prev_idx];
            if !strongly_precedes(prev_anchor, anchor) {
                continue;
            }

            let mut gap1 = (anchor.ref_start - prev_anchor.ref_end() - 1).max(0);
            let gap2 = (anchor.query_start - prev_anchor.query_end() - 1).max(0);
            if anchor_idx == end_idx {
                gap1 = 0;
            }
            let gap = gap1.max(gap2);
            let overlap1 = (prev_anchor.ref_end() - anchor.ref_start + 1).max(0);
            let overlap2 = (prev_anchor.query_end() - anchor.query_start + 1).max(0);
            let overlap_delta = (overlap1 - overlap2).abs();

            consider_transition(
                prev_idx,
                gap + overlap_delta,
                adds_anchor,
                costs,
                chain_lens,
                &mut best_cost,
                &mut best_len,
                &mut best_pred,
            );
        }

        for dd in (diag + 1)..diagonal_values.len() {
            let diagonal_distance = (curr_diagonal - diagonal_values[dd]).abs();
            if diagonal_distance > bound {
                break;
            }
            if let Some(prev) = active_anchor[dd] {
                consider_transition(
                    prev,
                    diagonal_distance,
                    adds_anchor,
                    costs,
                    chain_lens,
                    &mut best_cost,
                    &mut best_len,
                    &mut best_pred,
                );
            }
        }

        for dd in (0..diag).rev() {
            let diagonal_distance = (curr_diagonal - diagonal_values[dd]).abs();
            if diagonal_distance > bound {
                break;
            }
            if let Some(prev) = active_anchor[dd] {
                consider_transition(
                    prev,
                    diagonal_distance,
                    adds_anchor,
                    costs,
                    chain_lens,
                    &mut best_cost,
                    &mut best_len,
                    &mut best_pred,
                );
            }
        }

        costs[anchor_idx] = best_cost;
        chain_lens[anchor_idx] = best_len;
        predecessors[anchor_idx] = best_pred;
    }
}

#[allow(clippy::too_many_arguments)]
fn consider_transition(
    prev_idx: usize,
    edge_cost: i64,
    adds_anchor: bool,
    costs: &[i64],
    chain_lens: &[usize],
    best_cost: &mut i64,
    best_len: &mut usize,
    best_pred: &mut Option<usize>,
) {
    if costs[prev_idx] >= INF {
        return;
    }

    let candidate_cost = costs[prev_idx].saturating_add(edge_cost);
    let candidate_len = chain_lens[prev_idx] + usize::from(adds_anchor);
    if candidate_cost < *best_cost || (candidate_cost == *best_cost && candidate_len > *best_len) {
        *best_cost = candidate_cost;
        *best_len = candidate_len;
        *best_pred = Some(prev_idx);
    }
}

fn strongly_precedes(prev: Anchor, next: Anchor) -> bool {
    prev.ref_start < next.ref_start
        && prev.ref_end() < next.ref_end()
        && prev.query_start < next.query_start
        && prev.query_end() < next.query_end()
}

fn build_anchors(
    hits: &[SeedAnchorHit],
    query_len: usize,
    ref_len: usize,
    anchor_len: usize,
    orientation: ChainOrientation,
) -> Vec<Anchor> {
    let mut anchors = Vec::with_capacity(hits.len() + 2);
    anchors.push(Anchor {
        ref_start: -1,
        query_start: -1,
        len: 1,
        hit: None,
    });

    for hit in hits {
        let query_start = oriented_query_start(hit.query_start, query_len, anchor_len, orientation);
        anchors.push(Anchor {
            ref_start: hit.ref_start as i64,
            query_start: query_start as i64,
            len: anchor_len as i64,
            hit: Some(AnchorHit {
                ref_start: hit.ref_start,
                query_start,
            }),
        });
    }

    anchors.push(Anchor {
        ref_start: ref_len as i64,
        query_start: query_len as i64,
        len: 1,
        hit: None,
    });

    anchors.sort_unstable_by_key(|anchor| {
        (
            anchor.ref_start,
            anchor.query_start,
            anchor.len,
            anchor.hit.is_none(),
        )
    });
    anchors
}

fn oriented_query_start(
    query_start: Offset,
    query_len: usize,
    anchor_len: usize,
    orientation: ChainOrientation,
) -> Offset {
    match orientation {
        ChainOrientation::Forward => query_start,
        ChainOrientation::Reverse => query_len
            .saturating_sub(query_start)
            .saturating_sub(anchor_len),
    }
}

fn sorted_events(anchors: &[Anchor]) -> Vec<Event> {
    let mut events = Vec::with_capacity(anchors.len() * 2);
    for (anchor_idx, anchor) in anchors.iter().copied().enumerate() {
        events.push(Event {
            pos: anchor.ref_start,
            is_start: true,
            anchor_idx,
        });
        events.push(Event {
            pos: anchor.ref_end(),
            is_start: false,
            anchor_idx,
        });
    }

    events.sort_unstable_by(|a, b| {
        a.pos
            .cmp(&b.pos)
            .then_with(|| b.is_start.cmp(&a.is_start))
            .then_with(|| a.anchor_idx.cmp(&b.anchor_idx))
    });
    events
}

fn diagonal_buckets(anchors: &[Anchor]) -> (Vec<usize>, Vec<i64>) {
    let mut diagonal_order = (0..anchors.len()).collect::<Vec<_>>();
    diagonal_order.sort_unstable_by_key(|&idx| anchors[idx].diagonal());

    let mut diagonal_idx = vec![0usize; anchors.len()];
    let mut diagonal_values = Vec::new();
    let mut previous = None;

    for idx in diagonal_order {
        let diagonal = anchors[idx].diagonal();
        if previous != Some(diagonal) {
            diagonal_values.push(diagonal);
            previous = Some(diagonal);
        }
        diagonal_idx[idx] = diagonal_values.len() - 1;
    }

    (diagonal_idx, diagonal_values)
}

fn initial_distance_bound(anchors: &[Anchor], query_len: usize) -> i64 {
    let covered = asymmetric_query_coverage(anchors).min(query_len as i64);
    let inverse_coverage = (query_len as i64).saturating_sub(covered);
    CHAIN_MIN_BOUND.max(((inverse_coverage as f64) * 1.1).floor() as i64)
}

fn asymmetric_query_coverage(anchors: &[Anchor]) -> i64 {
    let mut real = anchors
        .iter()
        .filter(|anchor| anchor.hit.is_some())
        .copied()
        .collect::<Vec<_>>();
    real.sort_unstable_by_key(|anchor| (anchor.query_start, anchor.query_end()));

    let mut covered = 0i64;
    let mut consumed = 0i64;
    for anchor in real {
        let start = anchor.query_start.max(0);
        let end = anchor.query_end();
        if consumed - 1 >= end {
            continue;
        }
        covered += end - start.max(consumed) + 1;
        consumed = end + 1;
    }
    covered
}

fn traceback_chain(anchors: &[Anchor], predecessors: &[Option<usize>]) -> Vec<AnchorHit> {
    let mut chain = Vec::new();
    let mut current = anchors.len() - 1;
    let mut guard = 0usize;

    while let Some(prev) = predecessors[current] {
        if let Some(hit) = anchors[current].hit {
            chain.push(hit);
        }
        if prev == 0 {
            break;
        }
        current = prev;
        guard += 1;
        if guard > anchors.len() {
            return Vec::new();
        }
    }

    chain.reverse();
    chain
}

fn chain_supported_hits(
    anchors: &[Anchor],
    costs: &[i64],
    chain_lens: &[usize],
    predecessors: &[Option<usize>],
    min_chain_hits: usize,
) -> Vec<AnchorHit> {
    let mut keep = vec![false; anchors.len()];
    for idx in 1..anchors.len().saturating_sub(1) {
        if costs[idx] >= INF || chain_lens[idx] < min_chain_hits {
            continue;
        }

        let mut current = idx;
        let mut guard = 0usize;
        loop {
            if anchors[current].hit.is_some() {
                keep[current] = true;
            }

            let Some(prev) = predecessors[current] else {
                break;
            };
            if prev == 0 {
                break;
            }
            current = prev;
            guard += 1;
            if guard > anchors.len() {
                break;
            }
        }
    }

    anchors
        .iter()
        .zip(keep)
        .filter_map(|(anchor, keep)| keep.then_some(anchor.hit).flatten())
        .collect()
}

fn merge_candidates(candidates: &mut Vec<L1Candidate>) {
    candidates.sort_by_key(|candidate| (candidate.seq_id, candidate.range_start));
    let mut merged: Vec<L1Candidate> = Vec::with_capacity(candidates.len());
    for candidate in candidates.drain(..) {
        if let Some(prev) = merged.last_mut() {
            if prev.seq_id == candidate.seq_id
                && prev.range_end.saturating_add(1) >= candidate.range_start
            {
                prev.range_end = prev.range_end.max(candidate.range_end);
                continue;
            }
        }
        merged.push(candidate);
    }
    *candidates = merged;
}

fn div_floor_isize(value: isize, divisor: isize) -> isize {
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder != 0 && ((remainder > 0) != (divisor > 0)) {
        quotient - 1
    } else {
        quotient
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chainx_semiglobal_follows_a_colinear_diagonal() {
        let hits = vec![
            hit(100, 0),
            hit(150, 50),
            hit(200, 100),
            hit(250, 150),
            hit(5_000, 80),
        ];

        let chain =
            chainx_semiglobal_supported_hits(&hits, 500, 10_000, 16, 3, ChainOrientation::Forward);
        let starts = chain
            .iter()
            .map(|hit| hit.ref_start.saturating_sub(hit.query_start))
            .collect::<Vec<_>>();

        assert!(starts.iter().filter(|&&start| start == 100).count() >= 4);
    }

    #[test]
    fn chainx_semiglobal_can_use_reverse_query_coordinates() {
        let hits = vec![
            hit(100, 400),
            hit(150, 350),
            hit(200, 300),
            hit(250, 250),
            hit(5_000, 80),
        ];

        let chain =
            chainx_semiglobal_supported_hits(&hits, 500, 10_000, 16, 3, ChainOrientation::Reverse);
        let starts = chain
            .iter()
            .map(|hit| hit.ref_start.saturating_sub(hit.query_start))
            .collect::<Vec<_>>();

        assert!(starts.iter().filter(|&&start| start == 16).count() >= 4);
    }

    #[test]
    fn chain_candidates_require_enough_unique_query_positions() {
        let setup = ChainSetup {
            candidate_band: CHAIN_CANDIDATE_BAND,
            min_unique_hits: 3,
        };
        let mut candidates = Vec::new();
        append_chain_candidates(
            0,
            &[
                AnchorHit {
                    ref_start: 100,
                    query_start: 10,
                },
                AnchorHit {
                    ref_start: 120,
                    query_start: 10,
                },
            ],
            &setup,
            &mut candidates,
        );
        assert!(candidates.is_empty());

        append_chain_candidates(
            0,
            &[
                AnchorHit {
                    ref_start: 100,
                    query_start: 0,
                },
                AnchorHit {
                    ref_start: 120,
                    query_start: 20,
                },
                AnchorHit {
                    ref_start: 140,
                    query_start: 40,
                },
            ],
            &setup,
            &mut candidates,
        );
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].range_start, 0);
        assert_eq!(candidates[0].range_end, 1100);
    }

    #[test]
    fn diagonal_cluster_requires_unique_query_positions() {
        let mut scratch = DiagonalGroupScratch::new(500);
        let repeated = vec![
            diagonal_hit(0, 100),
            diagonal_hit(0, 101),
            diagonal_hit(10, 110),
        ];
        assert!(!scratch.has_unique_query_support(&repeated, 3));

        let supported = vec![
            diagonal_hit(0, 100),
            diagonal_hit(10, 110),
            diagonal_hit(25, 125),
        ];
        assert!(scratch.has_unique_query_support(&supported, 3));
    }

    #[test]
    fn diagonal_cluster_hit_is_packed() {
        assert_eq!(std::mem::size_of::<DiagonalClusterHit>(), 16);
    }

    #[test]
    fn diagonal_cluster_median_uses_selection() {
        let mut group = vec![
            diagonal_hit(0, 500),
            diagonal_hit(10, 110),
            diagonal_hit(25, 125),
            diagonal_hit(30, 430),
            diagonal_hit(40, 140),
        ];

        assert_eq!(median_start_est(&mut group), 100);
    }

    fn hit(ref_start: Offset, query_start: Offset) -> SeedAnchorHit {
        SeedAnchorHit {
            seq_id: 0,
            ref_start,
            query_start,
        }
    }

    fn diagonal_hit(query_start: Offset, ref_start: Offset) -> DiagonalClusterHit {
        DiagonalClusterHit::new(0, 0, query_start, ref_start.saturating_sub(query_start))
    }
}
