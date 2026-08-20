use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::sliding_mapper::MappingResult;
use crate::{AniConfig, AniResult, Offset, QueryFileData, ReferenceIndex, SeqId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceModel {
    Poisson,
    Binomial,
}

impl DistanceModel {
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Poisson),
            1 => Some(Self::Binomial),
            _ => None,
        }
    }

    pub fn code(self) -> u8 {
        match self {
            Self::Poisson => 0,
            Self::Binomial => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Poisson => "poisson",
            Self::Binomial => "binomial",
        }
    }

    pub fn jaccard_to_distance(self, jaccard: f64, k: usize) -> f64 {
        if jaccard <= 0.0 {
            1.0
        } else if jaccard >= 1.0 {
            0.0
        } else {
            let shared_kmer_probability = (2.0 * jaccard) / (1.0 + jaccard);
            match self {
                Self::Poisson => (-1.0 / k as f64) * shared_kmer_probability.ln(),
                Self::Binomial => 1.0 - shared_kmer_probability.powf(1.0 / k as f64),
            }
        }
    }

    pub fn distance_to_jaccard(self, distance: f64, k: usize) -> f64 {
        let distance = distance.clamp(0.0, 1.0);
        let shared_kmer_probability = match self {
            Self::Poisson => (-(k as f64) * distance).exp(),
            Self::Binomial => (1.0 - distance).powf(k as f64),
        };
        shared_kmer_probability / (2.0 - shared_kmer_probability)
    }
}

impl Default for DistanceModel {
    fn default() -> Self {
        Self::Poisson
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DistanceEstimate {
    pub(crate) identity: f64,
    pub(crate) identity_upper_bound: f64,
}

#[derive(Debug)]
pub(crate) struct DistanceTableCache {
    kmer_size: usize,
    model: DistanceModel,
    tables: Vec<OnceLock<Arc<[DistanceEstimate]>>>,
    overflow: Mutex<HashMap<usize, Arc<[DistanceEstimate]>>>,
}

impl DistanceTableCache {
    pub(crate) fn new(kmer_size: usize, max_sketch_size: usize, model: DistanceModel) -> Self {
        Self {
            kmer_size,
            model,
            tables: std::iter::repeat_with(OnceLock::new)
                .take(max_sketch_size + 1)
                .collect(),
            overflow: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn table_for(&self, sketch_size: usize) -> Arc<[DistanceEstimate]> {
        if let Some(cell) = self.tables.get(sketch_size) {
            return Arc::clone(cell.get_or_init(|| {
                Arc::<[DistanceEstimate]>::from(build_distance_table(
                    sketch_size,
                    self.kmer_size,
                    self.model,
                ))
            }));
        }

        let mut overflow = self
            .overflow
            .lock()
            .expect("distance table cache mutex poisoned");
        if let Some(table) = overflow.get(&sketch_size) {
            return Arc::clone(table);
        }

        let table = Arc::<[DistanceEstimate]>::from(build_distance_table(
            sketch_size,
            self.kmer_size,
            self.model,
        ));
        overflow.insert(sketch_size, Arc::clone(&table));
        table
    }
}

pub(crate) fn build_distance_table(
    sketch_size: usize,
    kmer_size: usize,
    model: DistanceModel,
) -> Vec<DistanceEstimate> {
    if sketch_size == 0 {
        return Vec::new();
    }

    let mut table = Vec::with_capacity(sketch_size + 1);
    for shared in 0..=sketch_size {
        let best_jaccard = shared as f64 / sketch_size as f64;
        let mash_dist = model.jaccard_to_distance(best_jaccard, kmer_size);
        let mash_dist_lower_bound =
            md_lower_bound_with_model(mash_dist, sketch_size, kmer_size, 0.9, model);
        table.push(DistanceEstimate {
            identity: 100.0 * (1.0 - mash_dist),
            identity_upper_bound: 100.0 * (1.0 - mash_dist_lower_bound),
        });
    }
    table
}

#[derive(Debug, Clone)]

pub(crate) struct ShortMapping {
    pub(crate) ref_seq_id: SeqId,
    pub(crate) genome_id: usize,
    pub(crate) query_seq_id: usize,
    pub(crate) ref_start: Offset,
    pub(crate) query_start: Offset,
    pub(crate) map_ref_pos_bin: Offset,
    pub(crate) identity: f64,
}

pub(crate) fn compute_ani_results(
    query: &QueryFileData,
    reference: &ReferenceIndex,
    mappings: Vec<MappingResult>,
    config: &AniConfig,
) -> Vec<AniResult> {
    let mut short = mappings
        .into_iter()
        .map(|m| {
            let genome_id = reference.contigs[m.ref_seq_id].genome_id;
            ShortMapping {
                ref_seq_id: m.ref_seq_id,
                genome_id,
                query_seq_id: m.query_seq_id,
                ref_start: m.ref_start,
                query_start: 0,
                map_ref_pos_bin: m.ref_start / config.fragment_len.saturating_sub(20).max(1),
                identity: m.identity,
            }
        })
        .collect::<Vec<_>>();

    short.sort_by(|a, b| {
        (a.genome_id, a.query_seq_id)
            .cmp(&(b.genome_id, b.query_seq_id))
            .then_with(|| cmp_f64(a.identity, b.identity))
            .then_with(|| a.ref_seq_id.cmp(&b.ref_seq_id))
            .then_with(|| a.ref_start.cmp(&b.ref_start))
    });

    let mut one_way: Vec<ShortMapping> = Vec::new();
    for m in short {
        if let Some(last) = one_way.last_mut() {
            if last.genome_id == m.genome_id && last.query_seq_id == m.query_seq_id {
                *last = m;
                continue;
            }
        }
        one_way.push(m);
    }

    one_way.sort_by(|a, b| {
        (a.ref_seq_id, a.map_ref_pos_bin)
            .cmp(&(b.ref_seq_id, b.map_ref_pos_bin))
            .then_with(|| cmp_f64(a.identity, b.identity))
    });

    let mut two_way: Vec<ShortMapping> = Vec::new();
    for m in one_way {
        if let Some(last) = two_way.last_mut() {
            if last.ref_seq_id == m.ref_seq_id && last.map_ref_pos_bin == m.map_ref_pos_bin {
                *last = m;
                continue;
            }
        }
        two_way.push(m);
    }

    two_way.sort_by_key(|m| m.genome_id);

    let mut results = Vec::new();
    let mut i = 0usize;
    while i < two_way.len() {
        let genome_id = two_way[i].genome_id;
        let mut j = i;
        let mut sum_identity = 0.0;
        while j < two_way.len() && two_way[j].genome_id == genome_id {
            sum_identity += two_way[j].identity;
            j += 1;
        }

        let mapped = j - i;
        let min_genome_len = query.genome_len.min(reference.genomes[genome_id].length);
        let shared_len = mapped * config.fragment_len;
        if shared_len as f64 >= min_genome_len as f64 * config.min_fraction {
            results.push(AniResult {
                query: query.path.clone(),
                reference: reference.genomes[genome_id].path.clone(),
                ani: sum_identity / mapped as f64,
                mapped_fragments: mapped,
                total_query_fragments: query.fragments.len(),
            });
        }

        i = j;
    }

    results
}

pub(crate) fn cmp_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

pub fn j2md(jaccard: f64, k: usize) -> f64 {
    DistanceModel::Poisson.jaccard_to_distance(jaccard, k)
}

pub fn md2j(distance: f64, k: usize) -> f64 {
    DistanceModel::Poisson.distance_to_jaccard(distance, k)
}

pub fn md_lower_bound(distance: f64, sketch_size: usize, k: usize, ci: f64) -> f64 {
    md_lower_bound_with_model(distance, sketch_size, k, ci, DistanceModel::Poisson)
}

pub fn md_lower_bound_with_model(
    distance: f64,
    sketch_size: usize,
    k: usize,
    ci: f64,
    model: DistanceModel,
) -> f64 {
    if sketch_size == 0 {
        return 1.0;
    }

    let q2 = (1.0 - ci) / 2.0;
    let p = model.distance_to_jaccard(distance, k).clamp(0.0, 1.0);
    let mut x = ((sketch_size as f64 * p).ceil() as usize).max(1);

    while x <= sketch_size {
        let sf = binomial_sf(x, sketch_size, p);
        if sf < q2 {
            x = x.saturating_sub(1);
            break;
        }
        x += 1;
    }

    x = x.clamp(1, sketch_size);
    model.jaccard_to_distance(x as f64 / sketch_size as f64, k)
}

pub fn estimate_minimum_hits(sketch_size: usize, k: usize, percent_identity: f64) -> usize {
    estimate_minimum_hits_with_model(sketch_size, k, percent_identity, DistanceModel::Poisson)
}

pub fn estimate_minimum_hits_with_model(
    sketch_size: usize,
    k: usize,
    percent_identity: f64,
    model: DistanceModel,
) -> usize {
    let mash_dist = 1.0 - percent_identity / 100.0;
    let jaccard = model.distance_to_jaccard(mash_dist, k);
    (sketch_size as f64 * jaccard).ceil() as usize
}

pub fn estimate_minimum_hits_relaxed(sketch_size: usize, k: usize, percent_identity: f64) -> usize {
    estimate_minimum_hits_relaxed_with_model(
        sketch_size,
        k,
        percent_identity,
        DistanceModel::Poisson,
    )
}

pub fn estimate_minimum_hits_relaxed_with_model(
    sketch_size: usize,
    k: usize,
    percent_identity: f64,
    model: DistanceModel,
) -> usize {
    estimate_minimum_hits_relaxed_with_confidence_and_model(
        sketch_size,
        k,
        percent_identity,
        0.9,
        model,
    )
}

pub fn estimate_minimum_hits_relaxed_with_confidence_and_model(
    sketch_size: usize,
    k: usize,
    percent_identity: f64,
    confidence_interval: f64,
    model: DistanceModel,
) -> usize {
    if sketch_size == 0 {
        return 0;
    }

    let strict = estimate_minimum_hits_with_model(sketch_size, k, percent_identity, model);
    let mut relaxed = strict;
    for i in (0..=strict).rev() {
        let jaccard = i as f64 / sketch_size as f64;
        let d = model.jaccard_to_distance(jaccard, k);
        let d_lower = md_lower_bound_with_model(d, sketch_size, k, confidence_interval, model);
        let upper_identity = 100.0 * (1.0 - d_lower);
        if upper_identity >= percent_identity {
            relaxed = i;
        } else {
            break;
        }
    }
    relaxed
}

pub fn estimate_pvalue(
    sketch_size: usize,
    k: usize,
    alphabet_size: usize,
    identity: f64,
    query_len: usize,
    reference_len: u64,
) -> f64 {
    estimate_pvalue_with_model(
        sketch_size,
        k,
        alphabet_size,
        identity,
        query_len,
        reference_len,
        DistanceModel::Poisson,
    )
}

pub fn estimate_pvalue_with_model(
    sketch_size: usize,
    k: usize,
    alphabet_size: usize,
    identity: f64,
    query_len: usize,
    reference_len: u64,
    model: DistanceModel,
) -> f64 {
    estimate_pvalue_with_confidence_and_model(
        sketch_size,
        k,
        alphabet_size,
        identity,
        query_len,
        reference_len,
        0.9,
        model,
    )
}

pub fn estimate_pvalue_with_confidence_and_model(
    sketch_size: usize,
    k: usize,
    alphabet_size: usize,
    identity: f64,
    query_len: usize,
    reference_len: u64,
    confidence_interval: f64,
    model: DistanceModel,
) -> f64 {
    let kmer_space = (alphabet_size as f64).powi(k as i32);
    let px = 1.0 / (1.0 + kmer_space / query_len as f64);
    let py = px;
    let random_jaccard = px * py / (px + py - px * py);
    let x = estimate_minimum_hits_relaxed_with_confidence_and_model(
        sketch_size,
        k,
        identity,
        confidence_interval,
        model,
    );
    let sf = if x == 0 {
        1.0
    } else {
        binomial_sf(x, sketch_size, random_jaccard)
    };
    reference_len as f64 * sf
}

pub fn recommended_window_size(
    pvalue_cutoff: f64,
    k: usize,
    alphabet_size: usize,
    identity: f64,
    query_len: usize,
    reference_len: u64,
) -> usize {
    recommended_window_size_with_model(
        pvalue_cutoff,
        k,
        alphabet_size,
        identity,
        query_len,
        reference_len,
        DistanceModel::Poisson,
    )
}

pub fn recommended_window_size_with_model(
    pvalue_cutoff: f64,
    k: usize,
    alphabet_size: usize,
    identity: f64,
    query_len: usize,
    reference_len: u64,
    model: DistanceModel,
) -> usize {
    let mut potential = vec![1usize, 2, 5];
    potential.extend((10..query_len).step_by(10));

    let optimal_sketch_size = potential
        .into_iter()
        .find(|&s| {
            estimate_pvalue_with_model(
                s,
                k,
                alphabet_size,
                identity,
                query_len,
                reference_len,
                model,
            ) <= pvalue_cutoff
        })
        .unwrap_or(query_len);

    ((2 * query_len) / optimal_sketch_size).clamp(1, query_len)
}

pub fn recommended_mashmap3_sketch_size_with_model(
    pvalue_cutoff: f64,
    k: usize,
    alphabet_size: usize,
    identity: f64,
    segment_len: usize,
    reference_len: u64,
    model: DistanceModel,
) -> usize {
    let length_query = segment_len.saturating_sub(k).max(1);
    for sketch_size in (10..length_query).step_by(10) {
        if estimate_pvalue_with_confidence_and_model(
            sketch_size,
            k,
            alphabet_size,
            identity,
            length_query,
            reference_len,
            0.95,
            model,
        ) <= pvalue_cutoff
        {
            return sketch_size;
        }
    }
    length_query
}

fn binomial_sf(x: usize, n: usize, p: f64) -> f64 {
    if x == 0 {
        return 1.0;
    }
    if x > n || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }

    let ln_p = p.ln();
    let ln_q = (1.0 - p).ln();
    let mut log_pmf = n as f64 * ln_q;
    let mut max_log = f64::NEG_INFINITY;
    let mut logs = Vec::with_capacity(n - x + 1);

    for i in 0..=n {
        if i >= x {
            max_log = max_log.max(log_pmf);
            logs.push(log_pmf);
        }

        if i < n {
            log_pmf += ((n - i) as f64).ln() - ((i + 1) as f64).ln() + ln_p - ln_q;
        }
    }

    let sum = logs
        .into_iter()
        .map(|value| (value - max_log).exp())
        .sum::<f64>();
    (max_log.exp() * sum).clamp(0.0, 1.0)
}
