use anyhow::Result;

use crate::compute_identity::estimate_minimum_hits_relaxed_with_model;
use crate::simd_minimizer::MinimizerMode;
use crate::{AniConfig, Offset, QuerySketch, ReferenceIndex, SeqId, u32_checked};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct MinimizerHit {
    seq_id: SeqId,
    wpos: Offset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedMinimizerHit {
    pub(crate) seq_id: u32,
    pub(crate) wpos: u32,
}

impl PackedMinimizerHit {
    pub(crate) fn new(seq_id: SeqId, wpos: Offset) -> Result<Self> {
        Ok(Self {
            seq_id: u32_checked(seq_id, "reference contig id")?,
            wpos: u32_checked(wpos, "reference minimizer position")?,
        })
    }

    pub(crate) fn seq_id(self) -> SeqId {
        self.seq_id as usize
    }

    pub(crate) fn wpos(self) -> Offset {
        self.wpos as usize
    }

    pub(crate) fn unpack(self) -> MinimizerHit {
        MinimizerHit {
            seq_id: self.seq_id(),
            wpos: self.wpos(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct L1Candidate {
    pub(crate) seq_id: SeqId,
    pub(crate) range_start: Offset,
    pub(crate) range_end: Offset,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct L1Stats {
    pub(crate) seed_hits: usize,
}

pub(crate) fn do_l1_mapping(
    query: &QuerySketch,
    reference: &ReferenceIndex,
    config: &AniConfig,
) -> (Vec<L1Candidate>, L1Stats) {
    if config.minimizer_mode == MinimizerMode::Scalar {
        return do_l1_mapping_fastani_exact(query, reference, config);
    }
    if config.minimizer_mode == MinimizerMode::ScalarMinmer {
        return do_l1_mapping_minmer_intervals(query, reference, config);
    }

    if config.chain {
        crate::chaining::do_l1_mapping_diagonal_then_chained(query, reference, config)
    } else {
        crate::chaining::do_l1_mapping_diagonal_then_rammap(query, reference, config)
    }
}

fn do_l1_mapping_fastani_exact(
    query: &QuerySketch,
    reference: &ReferenceIndex,
    config: &AniConfig,
) -> (Vec<L1Candidate>, L1Stats) {
    let mut seed_hits = Vec::new();

    for &hash in &query.unique_hashes {
        if let Some(hits) = reference.lookup.get(hash) {
            if hits.len() < reference.freq_threshold {
                seed_hits.extend(hits.iter().copied().map(PackedMinimizerHit::unpack));
            }
        }
    }

    let minimum_hits = estimate_minimum_hits_relaxed_with_model(
        query.unique_hashes.len(),
        config.kmer_size,
        config.min_identity,
        config.distance_model,
    )
    .max(1);

    let stats = L1Stats {
        seed_hits: seed_hits.len(),
    };
    (
        compute_l1_candidate_regions(&mut seed_hits, minimum_hits, query.len),
        stats,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MinmerEvent {
    seq_id: SeqId,
    pos: Offset,
    delta: i32,
}

fn do_l1_mapping_minmer_intervals(
    query: &QuerySketch,
    reference: &ReferenceIndex,
    config: &AniConfig,
) -> (Vec<L1Candidate>, L1Stats) {
    let Some(lookup) = reference.minmer_lookup.as_ref() else {
        return (Vec::new(), L1Stats::default());
    };

    let mut events = Vec::new();
    let mut seed_hits = 0usize;
    for &hash in &query.unique_hashes {
        if let Some(intervals) = lookup.get(hash) {
            if intervals.len() >= reference.freq_threshold {
                continue;
            }
            seed_hits += intervals.len();
            for interval in intervals {
                if interval.start() >= interval.end() {
                    continue;
                }
                events.push(MinmerEvent {
                    seq_id: interval.seq_id(),
                    pos: interval.start(),
                    delta: 1,
                });
                events.push(MinmerEvent {
                    seq_id: interval.seq_id(),
                    pos: interval.end(),
                    delta: -1,
                });
            }
        }
    }

    let minimum_hits = estimate_minimum_hits_relaxed_with_model(
        query.unique_hashes.len(),
        config.kmer_size,
        config.min_identity,
        config.distance_model,
    )
    .max(1);

    let stats = L1Stats { seed_hits };
    if events.is_empty() {
        return (Vec::new(), stats);
    }

    events.sort_unstable();
    let mut candidates: Vec<L1Candidate> = Vec::new();
    let mut active = 0i32;
    let mut idx = 0usize;
    while idx < events.len() {
        let seq_id = events[idx].seq_id;
        let pos = events[idx].pos;
        while idx < events.len() && events[idx].seq_id == seq_id && events[idx].pos == pos {
            active += events[idx].delta;
            idx += 1;
        }

        let next_pos = if idx < events.len() && events[idx].seq_id == seq_id {
            events[idx].pos
        } else {
            active = 0;
            continue;
        };

        if active >= minimum_hits as i32 && next_pos > pos {
            let candidate = L1Candidate {
                seq_id,
                range_start: pos,
                range_end: next_pos - 1,
            };
            if let Some(prev) = candidates.last_mut() {
                if prev.seq_id == candidate.seq_id
                    && prev.range_end.saturating_add(1) >= candidate.range_start
                {
                    prev.range_end = prev.range_end.max(candidate.range_end);
                    continue;
                }
            }
            candidates.push(candidate);
        }
    }

    (candidates, stats)
}

fn compute_l1_candidate_regions(
    seed_hits: &mut [MinimizerHit],
    minimum_hits: usize,
    query_len: usize,
) -> Vec<L1Candidate> {
    seed_hits.sort_unstable();
    if seed_hits.len() < minimum_hits {
        return Vec::new();
    }

    let mut candidates: Vec<L1Candidate> = Vec::new();
    for start in 0..=(seed_hits.len() - minimum_hits) {
        let first = seed_hits[start];
        let last = seed_hits[start + minimum_hits - 1];
        if first.seq_id == last.seq_id && last.wpos.saturating_sub(first.wpos) < query_len {
            let range_start = last.wpos.saturating_sub(query_len.saturating_sub(1));
            let candidate = L1Candidate {
                seq_id: first.seq_id,
                range_start,
                range_end: first.wpos,
            };

            if let Some(prev) = candidates.last_mut() {
                if prev.seq_id == candidate.seq_id && prev.range_end >= candidate.range_start {
                    prev.range_end = prev.range_end.max(candidate.range_end);
                    continue;
                }
            }

            candidates.push(candidate);
        }
    }

    candidates
}
