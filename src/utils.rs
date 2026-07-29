use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use indicatif::ProgressBar;
use needletail::{Sequence, parse_fastx_file};
use rayon::prelude::*;
use tab_hash::Tab64Twisted;

use crate::candidate_window::{PackedMinimizerHit, do_l1_mapping};
use crate::compute_identity::{
    DistanceEstimate, DistanceTableCache, compute_ani_results, recommended_window_size,
};
use crate::simd_minimizer::{
    Minimizer, MinimizerMode, QuerySeed, deterministic_tab64_twisted, sequence_minimizers,
    simd_compatible_window_size, splitmix64_permute,
};
use crate::sliding_mapper::{MappingResult, do_l2_mapping};

#[cfg(test)]
use crate::compute_identity::{build_distance_table, j2md, md_lower_bound, md2j};
#[cfg(test)]
use crate::simd_minimizer::splitmix64;
#[cfg(test)]
use crate::sliding_mapper::{
    BitsetBottomSketchSlideMapper, SummaryBitSet, build_indexed_minimizers,
};

pub(crate) type HashValue = u64;
pub(crate) type SeqId = usize;
pub(crate) type Offset = usize;

pub struct AniConfig {
    pub kmer_size: usize,
    pub fragment_len: usize,
    pub min_identity: f64,
    pub min_fraction: f64,
    pub p_value: f64,
    pub reference_size: u64,
    pub window_size: Option<usize>,
    pub ignore_top_percent: f64,
    pub tab_hash_seed: u64,
    pub minimizer_mode: MinimizerMode,
    pub chain: bool,
    pub diag_cluster_bin: usize,
    pub diag_cluster_band: usize,
    pub show_progress: bool,
}

impl Default for AniConfig {
    fn default() -> Self {
        Self {
            kmer_size: 16,
            fragment_len: 3000,
            min_identity: 80.0,
            min_fraction: 0.2,
            p_value: 1e-3,
            reference_size: 5_000_000,
            window_size: None,
            ignore_top_percent: 0.0,
            tab_hash_seed: 42,
            minimizer_mode: MinimizerMode::Simd,
            chain: false,
            diag_cluster_bin: 1000,
            diag_cluster_band: 500,
            show_progress: false,
        }
    }
}

impl AniConfig {
    pub fn resolved_window_size(&self) -> usize {
        let window_size = self.window_size.unwrap_or_else(|| {
            recommended_window_size(
                self.p_value,
                self.kmer_size,
                4,
                self.min_identity,
                self.fragment_len,
                self.reference_size,
            )
        });
        match self.minimizer_mode {
            MinimizerMode::Simd => simd_compatible_window_size(self.kmer_size, window_size),
            MinimizerMode::FastAni => window_size,
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.kmer_size > 0, "k-mer size must be positive");
        anyhow::ensure!(
            self.kmer_size <= 32,
            "k-mer size must be <= 32 for u64 k-mer values"
        );
        anyhow::ensure!(
            self.fragment_len > self.kmer_size,
            "fragment length must exceed k-mer size"
        );
        anyhow::ensure!(
            (0.0..=100.0).contains(&self.min_identity),
            "minimum identity must be in [0, 100]"
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.min_fraction),
            "minimum fraction must be in [0, 1]"
        );
        anyhow::ensure!(
            (0.0..=100.0).contains(&self.ignore_top_percent),
            "ignoreTopPercent must be in [0, 100]"
        );
        anyhow::ensure!(self.diag_cluster_bin > 0, "diagBin must be positive");
        anyhow::ensure!(self.diag_cluster_band > 0, "diagBand must be positive");
        let w = self.resolved_window_size();
        anyhow::ensure!(w > 0, "minimizer window size must be positive");
        anyhow::ensure!(
            self.fragment_len >= self.kmer_size + w - 1,
            "fragment length must be at least k + w - 1"
        );
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AniResult {
    pub query: PathBuf,
    pub reference: PathBuf,
    pub ani: f64,
    pub mapped_fragments: usize,
    pub total_query_fragments: usize,
}

#[derive(Debug, Clone)]
pub struct RunOutput {
    pub results: Vec<AniResult>,
    pub timing: TimingReport,
}

#[derive(Debug, Clone, Default)]
pub struct TimingReport {
    pub total_wall_ns: u128,
    pub reference: ReferenceTiming,
    pub queries: Vec<QueryTiming>,
    pub aggregate: MappingCounters,
}

#[derive(Debug, Clone, Default)]
pub struct ReferenceTiming {
    pub total_wall_ns: u128,
    pub read_wall_ns: u128,
    pub assemble_wall_ns: u128,
    pub sort_wall_ns: u128,
    pub lookup_wall_ns: u128,
    pub genomes: usize,
    pub contigs: usize,
    pub minimizers: usize,
    pub lookup_keys: usize,
    pub freq_threshold: usize,
}

#[derive(Debug, Clone)]
pub struct QueryTiming {
    pub path: PathBuf,
    pub genome_len: usize,
    pub fragments: usize,
    pub mappings: usize,
    pub results: usize,
    pub total_wall_ns: u128,
    pub read_wall_ns: u128,
    pub map_wall_ns: u128,
    pub ani_wall_ns: u128,
    pub counters: MappingCounters,
}

#[derive(Debug, Clone, Default)]
pub struct MappingCounters {
    pub fragments: u64,
    pub query_minimizers: u64,
    pub seed_hits: u64,
    pub l1_candidates: u64,
    pub l2_candidates: u64,
    pub l2_windows: u64,
    pub l2_ref_sketches: u64,
    pub mappings: u64,
    pub query_minimizer_ns: u128,
    pub query_sketch_ns: u128,
    pub l1_ns: u128,
    pub l2_ns: u128,
    pub l2_ref_hash_ns: u128,
    pub l2_ref_sketch_ns: u128,
    pub l2_distance_ns: u128,
    pub max_l1_candidates_per_fragment: u64,
    pub max_l2_coord_count: u64,
    pub max_l2_ref_minimizers: u64,
    pub max_l2_windows_per_candidate: u64,
    pub l2_coord_le_512: u64,
    pub l2_coord_le_2048: u64,
    pub l2_coord_le_8192: u64,
    pub l2_coord_gt_8192: u64,
}

impl MappingCounters {
    fn add(&mut self, other: &Self) {
        self.fragments += other.fragments;
        self.query_minimizers += other.query_minimizers;
        self.seed_hits += other.seed_hits;
        self.l1_candidates += other.l1_candidates;
        self.l2_candidates += other.l2_candidates;
        self.l2_windows += other.l2_windows;
        self.l2_ref_sketches += other.l2_ref_sketches;
        self.mappings += other.mappings;
        self.query_minimizer_ns += other.query_minimizer_ns;
        self.query_sketch_ns += other.query_sketch_ns;
        self.l1_ns += other.l1_ns;
        self.l2_ns += other.l2_ns;
        self.l2_ref_hash_ns += other.l2_ref_hash_ns;
        self.l2_ref_sketch_ns += other.l2_ref_sketch_ns;
        self.l2_distance_ns += other.l2_distance_ns;
        self.max_l1_candidates_per_fragment = self
            .max_l1_candidates_per_fragment
            .max(other.max_l1_candidates_per_fragment);
        self.max_l2_coord_count = self.max_l2_coord_count.max(other.max_l2_coord_count);
        self.max_l2_ref_minimizers = self.max_l2_ref_minimizers.max(other.max_l2_ref_minimizers);
        self.max_l2_windows_per_candidate = self
            .max_l2_windows_per_candidate
            .max(other.max_l2_windows_per_candidate);
        self.l2_coord_le_512 += other.l2_coord_le_512;
        self.l2_coord_le_2048 += other.l2_coord_le_2048;
        self.l2_coord_le_8192 += other.l2_coord_le_8192;
        self.l2_coord_gt_8192 += other.l2_coord_gt_8192;
    }
}

const EMPTY_LOOKUP_SLOT: u32 = u32::MAX;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ContigInfo {
    pub(crate) name: String,
    pub(crate) len: usize,
    pub(crate) genome_id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]

struct LookupRange {
    hash: HashValue,
    start: u32,
    len: u32,
}

#[derive(Debug)]
pub(crate) struct CompactLookupIndex {
    ranges: Vec<LookupRange>,
    hits: Vec<PackedMinimizerHit>,
    range_slots: Vec<u32>,
}

#[derive(Debug, Clone)]
pub(crate) struct GenomeInfo {
    pub(crate) path: PathBuf,
    pub(crate) length: usize,
}

#[derive(Debug)]
struct ReferenceGenomeBuild {
    genome_id: usize,
    genome: GenomeInfo,
    contigs: Vec<ReferenceContigBuild>,
}

#[derive(Debug)]
struct ReferenceContigBuild {
    name: String,
    len: usize,
    minimizers: Vec<Minimizer>,
}

#[derive(Debug)]
pub(crate) struct ReferenceIndex {
    pub(crate) genomes: Vec<GenomeInfo>,
    pub(crate) contigs: Vec<ContigInfo>,
    pub(crate) minimizers: Vec<Minimizer>,
    pub(crate) contig_ranges: Vec<std::ops::Range<usize>>,
    pub(crate) lookup: CompactLookupIndex,
    pub(crate) freq_threshold: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct QueryFragment {
    pub(crate) seq_id: usize,
    pub(crate) global_start: usize,
    pub(crate) seq: Vec<u8>,
}

#[derive(Debug, Clone)]
pub(crate) struct QueryFileData {
    pub(crate) path: PathBuf,
    pub(crate) genome_len: usize,
    pub(crate) fragments: Vec<QueryFragment>,
}

pub(crate) struct QuerySketch {
    pub(crate) fragment_id: usize,
    pub(crate) len: usize,
    pub(crate) unique_hashes: Vec<HashValue>,
    pub(crate) unique_seeds: Vec<QuerySeed>,
    pub(crate) distance_table: Arc<[DistanceEstimate]>,
}

impl CompactLookupIndex {
    fn from_hash_sorted_minimizers(minimizers: &[Minimizer]) -> Result<Self> {
        let mut ranges = Vec::new();
        let mut hits = Vec::with_capacity(minimizers.len());

        let mut group_start = 0usize;
        while group_start < minimizers.len() {
            let hash = minimizers[group_start].hash;
            let range_start = hits.len();
            let mut group_end = group_start + 1;
            while group_end < minimizers.len() && minimizers[group_end].hash == hash {
                group_end += 1;
            }

            for minimizer in &minimizers[group_start..group_end] {
                hits.push(PackedMinimizerHit::new(minimizer.seq_id, minimizer.wpos)?);
            }

            ranges.push(LookupRange {
                hash,
                start: u32_checked(range_start, "lookup hit range start")?,
                len: u32_checked(group_end - group_start, "lookup hit range length")?,
            });
            group_start = group_end;
        }

        let range_slots = Self::build_range_slots(&ranges)?;
        Ok(Self {
            ranges,
            hits,
            range_slots,
        })
    }

    fn len(&self) -> usize {
        self.ranges.len()
    }

    pub(crate) fn get(&self, hash: HashValue) -> Option<&[PackedMinimizerHit]> {
        if self.range_slots.is_empty() {
            return None;
        }

        let mask = self.range_slots.len() - 1;
        let mut slot = lookup_slot(hash, mask);
        loop {
            let range_idx = self.range_slots[slot];
            if range_idx == EMPTY_LOOKUP_SLOT {
                return None;
            }

            let range = self.ranges[range_idx as usize];
            if range.hash == hash {
                let start = range.start as usize;
                let end = start + range.len as usize;
                return Some(&self.hits[start..end]);
            }
            slot = (slot + 1) & mask;
        }
    }

    fn frequency_threshold(&self, ignore_top_percent: f64) -> usize {
        if ignore_top_percent <= 0.0 || self.ranges.is_empty() {
            return usize::MAX;
        }

        let mut counts = self
            .ranges
            .iter()
            .map(|range| range.len as usize)
            .collect::<Vec<_>>();
        counts.sort_unstable_by(|a, b| b.cmp(a));
        let to_ignore = ((counts.len() as f64) * ignore_top_percent / 100.0).floor() as usize;
        if to_ignore == 0 {
            usize::MAX
        } else {
            counts[to_ignore.saturating_sub(1)]
        }
    }

    fn build_range_slots(ranges: &[LookupRange]) -> Result<Vec<u32>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        anyhow::ensure!(
            ranges.len() < EMPTY_LOOKUP_SLOT as usize,
            "too many unique minimizer hashes for compact lookup index"
        );

        let min_slots = ((ranges.len() * 10).div_ceil(7)).max(1);
        let slot_count = min_slots.next_power_of_two();
        let mut slots = vec![EMPTY_LOOKUP_SLOT; slot_count];
        let mask = slot_count - 1;

        for (range_idx, range) in ranges.iter().enumerate() {
            let mut slot = lookup_slot(range.hash, mask);
            loop {
                if slots[slot] == EMPTY_LOOKUP_SLOT {
                    slots[slot] = range_idx as u32;
                    break;
                }
                slot = (slot + 1) & mask;
            }
        }

        Ok(slots)
    }
}

pub fn compare_paths(
    query_paths: &[PathBuf],
    ref_paths: &[PathBuf],
    config: &AniConfig,
) -> Result<Vec<AniResult>> {
    Ok(compare_paths_with_timing(query_paths, ref_paths, config)?.results)
}

pub fn compare_paths_with_timing(
    query_paths: &[PathBuf],
    ref_paths: &[PathBuf],
    config: &AniConfig,
) -> Result<RunOutput> {
    compare_paths_with_timing_inner(query_paths, ref_paths, config, None)
}

fn compare_paths_with_timing_inner(
    query_paths: &[PathBuf],
    ref_paths: &[PathBuf],
    config: &AniConfig,
    shared_pair_progress: Option<&ProgressBar>,
) -> Result<RunOutput> {
    config.validate()?;
    let total_start = Instant::now();
    let window_size = config.resolved_window_size();
    let tab_hasher = deterministic_tab64_twisted(config.tab_hash_seed);
    let reference_progress = progress_bar(
        config.show_progress,
        usize_to_u64_saturating(ref_paths.len()),
        format!("building reference index for {} genomes", ref_paths.len()),
    );
    let (reference, reference_timing) = ReferenceIndex::build(
        ref_paths,
        config,
        window_size,
        &tab_hasher,
        &reference_progress,
    )?;
    finish_progress(
        &reference_progress,
        format!("indexed {} reference genomes", ref_paths.len()),
    );
    let distance_cache = DistanceTableCache::new(config.kmer_size, config.fragment_len);

    let pair_total = usize_to_u64_saturating(query_paths.len())
        .saturating_mul(usize_to_u64_saturating(ref_paths.len()));
    let pair_step = usize_to_u64_saturating(ref_paths.len());
    let pair_progress = shared_pair_progress.cloned().unwrap_or_else(|| {
        progress_bar(
            config.show_progress,
            pair_total,
            format!(
                "mapping {} query genomes against {} reference genomes",
                query_paths.len(),
                ref_paths.len()
            ),
        )
    });
    let per_query = query_paths
        .par_iter()
        .map(|path| {
            let result = (|| {
                let query_total_start = Instant::now();
                let read_start = Instant::now();
                let query = read_query_file(path, config)?;
                let read_wall_ns = read_start.elapsed().as_nanos();

                let map_start = Instant::now();
                let (mappings, counters) =
                    map_query_file(&query, &reference, config, window_size, &distance_cache)?;
                let map_wall_ns = map_start.elapsed().as_nanos();

                let ani_start = Instant::now();
                let mapping_count = mappings.len();
                let results = compute_ani_results(&query, &reference, mappings, config);
                let ani_wall_ns = ani_start.elapsed().as_nanos();

                let timing = QueryTiming {
                    path: query.path.clone(),
                    genome_len: query.genome_len,
                    fragments: query.fragments.len(),
                    mappings: mapping_count,
                    results: results.len(),
                    total_wall_ns: query_total_start.elapsed().as_nanos(),
                    read_wall_ns,
                    map_wall_ns,
                    ani_wall_ns,
                    counters,
                };

                Ok((results, timing))
            })();
            pair_progress.inc(pair_step);
            result
        })
        .collect::<Result<Vec<_>>>()?;
    if shared_pair_progress.is_none() {
        finish_progress(&pair_progress, format!("mapped {pair_total} ANI pairs"));
    }

    let mut all_results = Vec::new();
    let mut query_timings = Vec::new();
    let mut aggregate = MappingCounters::default();
    for (results, timing) in per_query {
        aggregate.add(&timing.counters);
        all_results.extend(results);
        query_timings.push(timing);
    }

    sort_ani_results(&mut all_results);

    query_timings.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(RunOutput {
        results: all_results,
        timing: TimingReport {
            total_wall_ns: total_start.elapsed().as_nanos(),
            reference: reference_timing,
            queries: query_timings,
            aggregate,
        },
    })
}

pub fn compare_paths_split_with_timing(
    query_paths: &[PathBuf],
    ref_paths: &[PathBuf],
    config: &AniConfig,
    split_count: usize,
) -> Result<RunOutput> {
    config.validate()?;
    anyhow::ensure!(split_count >= 2, "--split must be at least 2");
    anyhow::ensure!(
        split_count < ref_paths.len(),
        "--split chunk count must be smaller than the number of reference genomes"
    );
    anyhow::ensure!(
        config.ignore_top_percent <= f64::EPSILON,
        "--split is currently exact only when --ignoreTopPercent is 0.0"
    );

    let split_size = ref_paths.len().div_ceil(split_count).max(1);
    let total_start = Instant::now();
    let mut all_results = Vec::new();
    let mut reference_timing = ReferenceTiming::default();
    let mut aggregate = MappingCounters::default();
    let mut query_timing_by_path: HashMap<PathBuf, QueryTiming> = HashMap::new();
    let pair_total = usize_to_u64_saturating(query_paths.len())
        .saturating_mul(usize_to_u64_saturating(ref_paths.len()));
    let pair_progress = progress_bar(
        config.show_progress,
        pair_total,
        format!("mapping {pair_total} total ANI pairs across {split_count} reference chunks"),
    );

    for (chunk_idx, ref_chunk) in ref_paths.chunks(split_size).enumerate() {
        log::debug!(
            "phase=start_split_reference_chunk chunk={} split_count={} refs={} total_refs={}",
            chunk_idx + 1,
            split_count,
            ref_chunk.len(),
            ref_paths.len()
        );
        let chunk_run =
            compare_paths_with_timing_inner(query_paths, ref_chunk, config, Some(&pair_progress))?;
        log::debug!(
            "phase=end_split_reference_chunk chunk={} split_count={} refs={} elapsed={:.6}s results={}",
            chunk_idx + 1,
            split_count,
            ref_chunk.len(),
            seconds(chunk_run.timing.total_wall_ns),
            chunk_run.results.len()
        );

        add_reference_timing(&mut reference_timing, &chunk_run.timing.reference);
        aggregate.add(&chunk_run.timing.aggregate);
        all_results.extend(chunk_run.results);

        for query_timing in chunk_run.timing.queries {
            match query_timing_by_path.entry(query_timing.path.clone()) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    add_query_timing(entry.get_mut(), query_timing);
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(query_timing);
                }
            }
        }
    }
    finish_progress(&pair_progress, format!("mapped {pair_total} ANI pairs"));

    sort_ani_results(&mut all_results);
    let mut query_timings = query_timing_by_path.into_values().collect::<Vec<_>>();
    query_timings.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(RunOutput {
        results: all_results,
        timing: TimingReport {
            total_wall_ns: total_start.elapsed().as_nanos(),
            reference: reference_timing,
            queries: query_timings,
            aggregate,
        },
    })
}

fn sort_ani_results(results: &mut [AniResult]) {
    results.sort_by(|a, b| {
        b.ani
            .partial_cmp(&a.ani)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.query.cmp(&b.query))
            .then_with(|| a.reference.cmp(&b.reference))
    });
}

fn add_reference_timing(total: &mut ReferenceTiming, chunk: &ReferenceTiming) {
    total.total_wall_ns += chunk.total_wall_ns;
    total.read_wall_ns += chunk.read_wall_ns;
    total.assemble_wall_ns += chunk.assemble_wall_ns;
    total.sort_wall_ns += chunk.sort_wall_ns;
    total.lookup_wall_ns += chunk.lookup_wall_ns;
    total.genomes += chunk.genomes;
    total.contigs += chunk.contigs;
    total.minimizers += chunk.minimizers;
    total.lookup_keys += chunk.lookup_keys;
    total.freq_threshold = total.freq_threshold.max(chunk.freq_threshold);
}

fn add_query_timing(total: &mut QueryTiming, chunk: QueryTiming) {
    total.mappings += chunk.mappings;
    total.results += chunk.results;
    total.total_wall_ns += chunk.total_wall_ns;
    total.read_wall_ns += chunk.read_wall_ns;
    total.map_wall_ns += chunk.map_wall_ns;
    total.ani_wall_ns += chunk.ani_wall_ns;
    total.counters.add(&chunk.counters);
}

fn progress_bar(enabled: bool, len: u64, _message: String) -> ProgressBar {
    if !enabled {
        return ProgressBar::hidden();
    }

    ProgressBar::new(len)
}

fn finish_progress(progress: &ProgressBar, message: String) {
    if progress.is_hidden() {
        return;
    }
    progress.finish_with_message(message);
}

fn usize_to_u64_saturating(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

pub fn read_path_list(path: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let text = fs::read_to_string(path.as_ref())
        .with_context(|| format!("failed to read list {}", path.as_ref().display()))?;
    let paths = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    anyhow::ensure!(!paths.is_empty(), "path list is empty");
    Ok(paths)
}

pub fn write_results(path: impl AsRef<Path>, results: &[AniResult]) -> Result<()> {
    let file = fs::File::create(path.as_ref())
        .with_context(|| format!("failed to create {}", path.as_ref().display()))?;
    let mut out = BufWriter::new(file);
    for result in results {
        writeln!(
            out,
            "{}\t{}\t{:.6}\t{}\t{}",
            result.query.display(),
            result.reference.display(),
            result.ani,
            result.mapped_fragments,
            result.total_query_fragments
        )?;
    }
    Ok(())
}

pub fn write_phylip_matrix(
    output_path: impl AsRef<Path>,
    query_paths: &[PathBuf],
    ref_paths: &[PathBuf],
    results: &[AniResult],
) -> Result<PathBuf> {
    let mut genome_to_index = HashMap::new();
    let mut genomes = Vec::new();
    for path in query_paths.iter().chain(ref_paths) {
        if !genome_to_index.contains_key(path) {
            genome_to_index.insert(path.clone(), genomes.len());
            genomes.push(path.clone());
        }
    }

    let n = genomes.len();
    let mut matrix = vec![0.0f64; n * n];
    for result in results {
        let Some(&query_idx) = genome_to_index.get(&result.query) else {
            continue;
        };
        let Some(&ref_idx) = genome_to_index.get(&result.reference) else {
            continue;
        };
        if query_idx == ref_idx {
            continue;
        }

        let (row, col) = if query_idx > ref_idx {
            (query_idx, ref_idx)
        } else {
            (ref_idx, query_idx)
        };
        let cell = &mut matrix[row * n + col];
        if *cell > 0.0 {
            *cell = (*cell + result.ani) / 2.0;
        } else {
            *cell = result.ani;
        }
    }

    let matrix_path = fastani_matrix_path(output_path.as_ref());
    let file = fs::File::create(&matrix_path)
        .with_context(|| format!("failed to create {}", matrix_path.display()))?;
    let mut out = BufWriter::new(file);

    writeln!(out, "{n}")?;
    for (i, genome) in genomes.iter().enumerate() {
        write!(out, "{}", genome.display())?;
        for j in 0..i {
            let value = matrix[i * n + j];
            if value > 0.0 {
                write!(out, "\t{value:.6}")?;
            } else {
                write!(out, "\tNA")?;
            }
        }
        writeln!(out)?;
    }

    Ok(matrix_path)
}

fn fastani_matrix_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".matrix");
    PathBuf::from(name)
}

pub fn write_timing_report(path: impl AsRef<Path>, report: &TimingReport) -> Result<()> {
    let file = fs::File::create(path.as_ref())
        .with_context(|| format!("failed to create {}", path.as_ref().display()))?;
    let mut out = BufWriter::new(file);

    writeln!(out, "section\tname\tmetric\tvalue")?;
    write_metric(
        &mut out,
        "run",
        "all",
        "total_sec",
        seconds(report.total_wall_ns),
    )?;
    write_metric(&mut out, "run", "all", "total_ns", report.total_wall_ns)?;

    let r = &report.reference;
    for (metric, value) in [
        ("total_sec", seconds(r.total_wall_ns)),
        ("read_sec", seconds(r.read_wall_ns)),
        ("assemble_sec", seconds(r.assemble_wall_ns)),
        ("sort_sec", seconds(r.sort_wall_ns)),
        ("lookup_sec", seconds(r.lookup_wall_ns)),
    ] {
        write_metric(&mut out, "reference", "all", metric, value)?;
    }
    write_metric(&mut out, "reference", "all", "genomes", r.genomes)?;
    write_metric(&mut out, "reference", "all", "contigs", r.contigs)?;
    write_metric(&mut out, "reference", "all", "minimizers", r.minimizers)?;
    write_metric(&mut out, "reference", "all", "lookup_keys", r.lookup_keys)?;
    write_metric(
        &mut out,
        "reference",
        "all",
        "freq_threshold",
        r.freq_threshold,
    )?;

    write_counter_metrics(&mut out, "aggregate", "all", &report.aggregate)?;
    for query in &report.queries {
        let name = query.path.display().to_string();
        write_metric(
            &mut out,
            "query",
            &name,
            "total_sec",
            seconds(query.total_wall_ns),
        )?;
        write_metric(
            &mut out,
            "query",
            &name,
            "read_sec",
            seconds(query.read_wall_ns),
        )?;
        write_metric(
            &mut out,
            "query",
            &name,
            "map_sec",
            seconds(query.map_wall_ns),
        )?;
        write_metric(
            &mut out,
            "query",
            &name,
            "ani_sec",
            seconds(query.ani_wall_ns),
        )?;
        write_metric(&mut out, "query", &name, "genome_len", query.genome_len)?;
        write_metric(&mut out, "query", &name, "fragments", query.fragments)?;
        write_metric(&mut out, "query", &name, "mappings", query.mappings)?;
        write_metric(&mut out, "query", &name, "results", query.results)?;
        write_counter_metrics(&mut out, "query_counter", &name, &query.counters)?;
    }

    Ok(())
}

pub fn format_timing_summary(report: &TimingReport) -> String {
    let mut text = String::new();
    let _ = writeln!(
        text,
        "timing total={:.6}s reference={:.6}s query_map_wall_sum={:.6}s",
        seconds(report.total_wall_ns),
        seconds(report.reference.total_wall_ns),
        seconds(report.queries.iter().map(|q| q.map_wall_ns).sum::<u128>())
    );
    let c = &report.aggregate;
    let _ = writeln!(
        text,
        "timing fragments={} query_minimizers={} seed_hits={} l1_candidates={} l2_candidates={} l2_windows={} l2_ref_sketches={} mappings={}",
        c.fragments,
        c.query_minimizers,
        c.seed_hits,
        c.l1_candidates,
        c.l2_candidates,
        c.l2_windows,
        c.l2_ref_sketches,
        c.mappings
    );
    let _ = writeln!(
        text,
        "timing cpu-stage-sum query_minimizers={:.6}s query_sketch={:.6}s l1={:.6}s l2={:.6}s l2_ref_hash={:.6}s l2_ref_sketch={:.6}s l2_distance={:.6}s",
        seconds(c.query_minimizer_ns),
        seconds(c.query_sketch_ns),
        seconds(c.l1_ns),
        seconds(c.l2_ns),
        seconds(c.l2_ref_hash_ns),
        seconds(c.l2_ref_sketch_ns),
        seconds(c.l2_distance_ns),
    );
    let _ = writeln!(
        text,
        "timing l2-shape max_candidates_per_fragment={} max_coord_count={} max_ref_minimizers={} max_windows_per_candidate={} coord_le_512={} coord_513_2048={} coord_2049_8192={} coord_gt_8192={}",
        c.max_l1_candidates_per_fragment,
        c.max_l2_coord_count,
        c.max_l2_ref_minimizers,
        c.max_l2_windows_per_candidate,
        c.l2_coord_le_512,
        c.l2_coord_le_2048,
        c.l2_coord_le_8192,
        c.l2_coord_gt_8192,
    );
    text
}

fn write_metric<T: std::fmt::Display>(
    out: &mut BufWriter<fs::File>,
    section: &str,
    name: &str,
    metric: &str,
    value: T,
) -> Result<()> {
    writeln!(out, "{section}\t{name}\t{metric}\t{value}")?;
    Ok(())
}

fn write_counter_metrics(
    out: &mut BufWriter<fs::File>,
    section: &str,
    name: &str,
    c: &MappingCounters,
) -> Result<()> {
    write_metric(out, section, name, "fragments", c.fragments)?;
    write_metric(out, section, name, "query_minimizers", c.query_minimizers)?;
    write_metric(out, section, name, "seed_hits", c.seed_hits)?;
    write_metric(out, section, name, "l1_candidates", c.l1_candidates)?;
    write_metric(out, section, name, "l2_candidates", c.l2_candidates)?;
    write_metric(out, section, name, "l2_windows", c.l2_windows)?;
    write_metric(out, section, name, "l2_ref_sketches", c.l2_ref_sketches)?;
    write_metric(out, section, name, "mappings", c.mappings)?;
    write_metric(
        out,
        section,
        name,
        "query_minimizer_sec",
        seconds(c.query_minimizer_ns),
    )?;
    write_metric(
        out,
        section,
        name,
        "query_sketch_sec",
        seconds(c.query_sketch_ns),
    )?;
    write_metric(out, section, name, "l1_sec", seconds(c.l1_ns))?;
    write_metric(out, section, name, "l2_sec", seconds(c.l2_ns))?;
    write_metric(
        out,
        section,
        name,
        "l2_ref_hash_sec",
        seconds(c.l2_ref_hash_ns),
    )?;
    write_metric(
        out,
        section,
        name,
        "l2_ref_sketch_sec",
        seconds(c.l2_ref_sketch_ns),
    )?;
    write_metric(
        out,
        section,
        name,
        "l2_distance_sec",
        seconds(c.l2_distance_ns),
    )?;
    write_metric(
        out,
        section,
        name,
        "max_l1_candidates_per_fragment",
        c.max_l1_candidates_per_fragment,
    )?;
    write_metric(
        out,
        section,
        name,
        "max_l2_coord_count",
        c.max_l2_coord_count,
    )?;
    write_metric(
        out,
        section,
        name,
        "max_l2_ref_minimizers",
        c.max_l2_ref_minimizers,
    )?;
    write_metric(
        out,
        section,
        name,
        "max_l2_windows_per_candidate",
        c.max_l2_windows_per_candidate,
    )?;
    write_metric(out, section, name, "l2_coord_le_512", c.l2_coord_le_512)?;
    write_metric(out, section, name, "l2_coord_le_2048", c.l2_coord_le_2048)?;
    write_metric(out, section, name, "l2_coord_le_8192", c.l2_coord_le_8192)?;
    write_metric(out, section, name, "l2_coord_gt_8192", c.l2_coord_gt_8192)?;
    Ok(())
}

fn seconds(ns: u128) -> f64 {
    ns as f64 / 1_000_000_000.0
}

pub(crate) fn u32_checked(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{label} exceeds u32::MAX"))
}

fn lookup_slot(hash: HashValue, mask: usize) -> usize {
    (splitmix64_permute(hash) as usize) & mask
}

impl ReferenceIndex {
    pub(crate) fn build(
        paths: &[PathBuf],
        config: &AniConfig,
        window_size: usize,
        tab_hasher: &Tab64Twisted,
        progress: &ProgressBar,
    ) -> Result<(Self, ReferenceTiming)> {
        anyhow::ensure!(!paths.is_empty(), "at least one reference path is required");
        let total_start = Instant::now();

        let read_start = Instant::now();
        let mut builds = paths
            .par_iter()
            .enumerate()
            .map(|(genome_id, path)| {
                let result =
                    read_reference_genome(path, genome_id, config, window_size, tab_hasher);
                progress.inc(1);
                result
            })
            .collect::<Result<Vec<_>>>()?;
        let read_wall_ns = read_start.elapsed().as_nanos();

        let assemble_start = Instant::now();
        builds.sort_by_key(|build| build.genome_id);

        let mut genomes = Vec::with_capacity(paths.len());
        let mut contigs = Vec::new();
        let mut minimizers = Vec::new();

        for build in builds {
            genomes.push(build.genome);
            for contig in build.contigs {
                let seq_id = contigs.len();
                contigs.push(ContigInfo {
                    name: contig.name,
                    len: contig.len,
                    genome_id: build.genome_id,
                });
                minimizers.extend(contig.minimizers.into_iter().map(|mut minimizer| {
                    minimizer.seq_id = seq_id;
                    minimizer
                }));
            }
        }
        let assemble_wall_ns = assemble_start.elapsed().as_nanos();

        let sort_start = Instant::now();
        minimizers.sort_unstable_by_key(|m| m.hash);
        let sort_hash_wall_ns = sort_start.elapsed().as_nanos();

        let lookup_start = Instant::now();
        let lookup = CompactLookupIndex::from_hash_sorted_minimizers(&minimizers)?;
        let freq_threshold = lookup.frequency_threshold(config.ignore_top_percent);
        let lookup_wall_ns = lookup_start.elapsed().as_nanos();

        let sort_position_start = Instant::now();
        minimizers.sort_unstable_by_key(|m| (m.seq_id, m.wpos));
        let mut contig_ranges = vec![0..0; contigs.len()];
        let mut start = 0usize;
        while start < minimizers.len() {
            let seq_id = minimizers[start].seq_id;
            let mut end = start + 1;
            while end < minimizers.len() && minimizers[end].seq_id == seq_id {
                end += 1;
            }
            contig_ranges[seq_id] = start..end;
            start = end;
        }
        let sort_wall_ns = sort_hash_wall_ns + sort_position_start.elapsed().as_nanos();

        let timing = ReferenceTiming {
            total_wall_ns: total_start.elapsed().as_nanos(),
            read_wall_ns,
            assemble_wall_ns,
            sort_wall_ns,
            lookup_wall_ns,
            genomes: genomes.len(),
            contigs: contigs.len(),
            minimizers: minimizers.len(),
            lookup_keys: lookup.len(),
            freq_threshold,
        };

        Ok((
            Self {
                genomes,
                contigs,
                minimizers,
                contig_ranges,
                lookup,
                freq_threshold,
            },
            timing,
        ))
    }

    pub(crate) fn lower_bound(&self, seq_id: SeqId, wpos: Offset) -> usize {
        let range = &self.contig_ranges[seq_id];
        let slice = &self.minimizers[range.clone()];
        range.start + slice.partition_point(|m| m.wpos < wpos)
    }

    pub(crate) fn contig_minimizer_bounds(&self, seq_id: SeqId) -> std::ops::Range<usize> {
        self.contig_ranges[seq_id].clone()
    }
}

fn read_reference_genome(
    path: &Path,
    genome_id: usize,
    config: &AniConfig,
    window_size: usize,
    tab_hasher: &Tab64Twisted,
) -> Result<ReferenceGenomeBuild> {
    let mut reader = parse_fastx_file(path)
        .with_context(|| format!("failed to open reference {}", path.display()))?;
    let mut genome_len = 0usize;
    let mut contigs = Vec::new();

    while let Some(record) = reader.next() {
        let record = record.with_context(|| format!("failed to parse {}", path.display()))?;
        let name = String::from_utf8_lossy(record.id()).into_owned();
        let seq = record.normalize(false);
        genome_len += seq.len();
        let minimizers = sequence_minimizers(seq.as_ref(), config, window_size, 0, tab_hasher)?;

        contigs.push(ReferenceContigBuild {
            name,
            len: seq.len(),
            minimizers,
        });
    }

    Ok(ReferenceGenomeBuild {
        genome_id,
        genome: GenomeInfo {
            path: path.to_path_buf(),
            length: genome_len,
        },
        contigs,
    })
}
pub(crate) fn read_query_file(path: &Path, config: &AniConfig) -> Result<QueryFileData> {
    let mut reader = parse_fastx_file(path)
        .with_context(|| format!("failed to open query {}", path.display()))?;
    let mut fragments = Vec::new();
    let mut genome_len = 0usize;
    let mut next_fragment_id = 0usize;

    while let Some(record) = reader.next() {
        let record = record.with_context(|| format!("failed to parse {}", path.display()))?;
        let seq = record.normalize(false);
        let contig_global_start = genome_len;
        genome_len += seq.len();

        if seq.len() < config.fragment_len {
            continue;
        }

        let fragment_count = seq.len() / config.fragment_len;
        for i in 0..fragment_count {
            let start = i * config.fragment_len;
            let end = start + config.fragment_len;
            fragments.push(QueryFragment {
                seq_id: next_fragment_id,
                global_start: contig_global_start + start,
                seq: seq[start..end].to_vec(),
            });
            next_fragment_id += 1;
        }
    }

    Ok(QueryFileData {
        path: path.to_path_buf(),
        genome_len,
        fragments,
    })
}

pub(crate) fn map_query_file(
    query: &QueryFileData,
    reference: &ReferenceIndex,
    config: &AniConfig,
    window_size: usize,
    distance_cache: &DistanceTableCache,
) -> Result<(Vec<MappingResult>, MappingCounters)> {
    let tab_hasher = deterministic_tab64_twisted(config.tab_hash_seed);
    let per_fragment = query
        .fragments
        .par_iter()
        .map(|fragment| {
            map_fragment(
                fragment,
                reference,
                config,
                window_size,
                &tab_hasher,
                distance_cache,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let mut mappings = Vec::new();
    let mut counters = MappingCounters::default();
    for (fragment_mappings, fragment_counters) in per_fragment {
        counters.add(&fragment_counters);
        mappings.extend(fragment_mappings);
    }

    Ok((mappings, counters))
}

fn map_fragment(
    fragment: &QueryFragment,
    reference: &ReferenceIndex,
    config: &AniConfig,
    window_size: usize,
    tab_hasher: &Tab64Twisted,
    distance_cache: &DistanceTableCache,
) -> Result<(Vec<MappingResult>, MappingCounters)> {
    let mut counters = MappingCounters {
        fragments: 1,
        ..MappingCounters::default()
    };

    let minimizer_start = Instant::now();
    let mut minimizers = sequence_minimizers(&fragment.seq, config, window_size, 0, tab_hasher)?;
    counters.query_minimizer_ns += minimizer_start.elapsed().as_nanos();

    minimizers.sort_by_key(|m| m.hash);
    minimizers.dedup_by_key(|m| m.hash);
    counters.query_minimizers += minimizers.len() as u64;

    if minimizers.is_empty() {
        return Ok((Vec::new(), counters));
    }

    let sketch_start = Instant::now();
    let distance_table = distance_cache.table_for(minimizers.len());
    counters.query_sketch_ns += sketch_start.elapsed().as_nanos();

    let query_sketch = QuerySketch {
        fragment_id: fragment.seq_id,
        len: fragment.seq.len(),
        unique_hashes: minimizers.iter().map(|m| m.hash).collect(),
        unique_seeds: minimizers
            .iter()
            .map(|m| QuerySeed {
                hash: m.hash,
                qpos: m.wpos,
            })
            .collect(),
        distance_table,
    };

    let l1_start = Instant::now();
    let (l1_candidates, l1_stats) = do_l1_mapping(&query_sketch, reference, config);
    counters.l1_ns += l1_start.elapsed().as_nanos();
    counters.seed_hits += l1_stats.seed_hits as u64;
    counters.l1_candidates += l1_candidates.len() as u64;
    counters.l2_candidates += l1_candidates.len() as u64;
    counters.max_l1_candidates_per_fragment = l1_candidates.len() as u64;

    let mut mappings = Vec::new();
    for candidate in l1_candidates {
        let l2_start = Instant::now();
        let (mapping, l2_stats) =
            do_l2_mapping(&query_sketch, candidate, reference, config, window_size)?;
        counters.l2_ns += l2_start.elapsed().as_nanos();
        counters.l2_windows += l2_stats.windows;
        counters.l2_ref_sketches += l2_stats.ref_sketches;
        counters.l2_ref_hash_ns += l2_stats.ref_hash_ns;
        counters.l2_ref_sketch_ns += l2_stats.ref_sketch_ns;
        counters.l2_distance_ns += l2_stats.distance_ns;
        counters.max_l2_coord_count = counters.max_l2_coord_count.max(l2_stats.coord_count as u64);
        counters.max_l2_ref_minimizers = counters
            .max_l2_ref_minimizers
            .max(l2_stats.reference_minimizers as u64);
        counters.max_l2_windows_per_candidate =
            counters.max_l2_windows_per_candidate.max(l2_stats.windows);
        match l2_stats.coord_count {
            0..=512 => counters.l2_coord_le_512 += 1,
            513..=2048 => counters.l2_coord_le_2048 += 1,
            2049..=8192 => counters.l2_coord_le_8192 += 1,
            _ => counters.l2_coord_gt_8192 += 1,
        }

        if let Some(mapping) = mapping {
            mappings.push(mapping);
        }
    }
    counters.mappings += mappings.len() as u64;

    Ok((mappings, counters))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn mash_distance_roundtrip() {
        let d = 0.05;
        let j = md2j(d, 16);
        let roundtrip = j2md(j, 16);
        assert!((d - roundtrip).abs() < 1e-12);
    }

    #[test]
    fn distance_table_matches_direct_formula() {
        let sketch_size = 257;
        let table = build_distance_table(sketch_size, 16);
        for &shared in &[0usize, 1, 17, 128, 256, 257] {
            let best_jaccard = shared as f64 / sketch_size as f64;
            let mash_dist = j2md(best_jaccard, 16);
            let mash_dist_lower_bound = md_lower_bound(mash_dist, sketch_size, 16, 0.9);
            assert_eq!(
                table[shared].identity.to_bits(),
                (100.0 * (1.0 - mash_dist)).to_bits()
            );
            assert_eq!(
                table[shared].identity_upper_bound.to_bits(),
                (100.0 * (1.0 - mash_dist_lower_bound)).to_bits()
            );
        }
    }

    #[test]
    fn simd_window_size_resolves_to_odd_canonical_span() {
        let default_k_config = AniConfig {
            min_identity: 78.0,
            minimizer_mode: MinimizerMode::Simd,
            ..AniConfig::default()
        };
        assert_eq!(
            recommended_window_size(
                default_k_config.p_value,
                default_k_config.kmer_size,
                4,
                default_k_config.min_identity,
                default_k_config.fragment_len,
                default_k_config.reference_size,
            ),
            17
        );
        assert_eq!(default_k_config.resolved_window_size(), 16);
        assert_eq!(
            (default_k_config.kmer_size + default_k_config.resolved_window_size() - 1) % 2,
            1
        );

        let odd_k_config = AniConfig {
            kmer_size: 15,
            window_size: Some(16),
            minimizer_mode: MinimizerMode::Simd,
            ..AniConfig::default()
        };
        assert_eq!(odd_k_config.resolved_window_size(), 15);
        assert_eq!(
            (odd_k_config.kmer_size + odd_k_config.resolved_window_size() - 1) % 2,
            1
        );

        let minimum_window_config = AniConfig {
            kmer_size: 16,
            window_size: Some(1),
            minimizer_mode: MinimizerMode::Simd,
            ..AniConfig::default()
        };
        assert_eq!(minimum_window_config.resolved_window_size(), 2);
        assert_eq!(
            (minimum_window_config.kmer_size + minimum_window_config.resolved_window_size() - 1)
                % 2,
            1
        );
    }

    #[test]
    fn fastani_mode_does_not_adjust_window_size_for_simd() {
        let config = AniConfig {
            kmer_size: 16,
            window_size: Some(17),
            minimizer_mode: MinimizerMode::FastAni,
            ..AniConfig::default()
        };
        assert_eq!(config.resolved_window_size(), 17);
    }

    #[test]
    fn identical_genomes_map() -> Result<()> {
        let dir = tempdir()?;
        let query = dir.path().join("query.fa");
        let reference = dir.path().join("ref.fa");
        let seq = deterministic_dna(6000);
        fs::write(&query, format!(">q\n{}\n", seq))?;
        fs::write(&reference, format!(">r\n{}\n", seq))?;

        let config = AniConfig {
            kmer_size: 8,
            fragment_len: 1000,
            min_identity: 70.0,
            min_fraction: 0.0,
            window_size: Some(10),
            ..AniConfig::default()
        };

        let results = compare_paths(&[query.clone()], &[reference.clone()], &config)?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].mapped_fragments, 6);
        assert!(results[0].ani > 99.0, "ANI was {}", results[0].ani);
        Ok(())
    }

    #[test]
    fn identical_genomes_map_with_chained_l1() -> Result<()> {
        let dir = tempdir()?;
        let query = dir.path().join("query.fa");
        let reference = dir.path().join("ref.fa");
        let seq = deterministic_dna(6000);
        fs::write(&query, format!(">q\n{}\n", seq))?;
        fs::write(&reference, format!(">r\n{}\n", seq))?;

        let config = AniConfig {
            kmer_size: 8,
            fragment_len: 1000,
            min_identity: 70.0,
            min_fraction: 0.0,
            window_size: Some(10),
            chain: true,
            ..AniConfig::default()
        };

        let results = compare_paths(&[query.clone()], &[reference.clone()], &config)?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].mapped_fragments, 6);
        assert!(results[0].ani > 99.0, "ANI was {}", results[0].ani);
        Ok(())
    }

    #[test]
    fn phylip_matrix_averages_reciprocal_results() -> Result<()> {
        let dir = tempdir()?;
        let out = dir.path().join("ani.out");
        let a = PathBuf::from("A.fa");
        let b = PathBuf::from("B.fa");
        let c = PathBuf::from("C.fa");
        let results = vec![
            AniResult {
                query: a.clone(),
                reference: b.clone(),
                ani: 97.0,
                mapped_fragments: 10,
                total_query_fragments: 12,
            },
            AniResult {
                query: b.clone(),
                reference: a.clone(),
                ani: 98.0,
                mapped_fragments: 11,
                total_query_fragments: 13,
            },
            AniResult {
                query: a.clone(),
                reference: c.clone(),
                ani: 90.0,
                mapped_fragments: 7,
                total_query_fragments: 12,
            },
            AniResult {
                query: a.clone(),
                reference: a.clone(),
                ani: 100.0,
                mapped_fragments: 12,
                total_query_fragments: 12,
            },
        ];

        let matrix_path = write_phylip_matrix(
            &out,
            &[a.clone(), b.clone()],
            &[a.clone(), b.clone(), c.clone()],
            &results,
        )?;
        let matrix = fs::read_to_string(matrix_path)?;

        assert_eq!(matrix, "3\nA.fa\nB.fa\t97.500000\nC.fa\t90.000000\tNA\n");
        Ok(())
    }

    #[test]
    fn summary_bitset_finds_previous_and_next_words() {
        let mut bits = SummaryBitSet::new(200);
        bits.set(3);
        bits.set(70);
        bits.set(130);

        assert_eq!(bits.prev_set_before(130), Some(70));
        assert_eq!(bits.prev_set_before(70), Some(3));
        assert_eq!(bits.next_set_after(3), Some(70));
        assert_eq!(bits.next_set_after(70), Some(130));

        bits.clear(70);
        assert_eq!(bits.prev_set_before(130), Some(3));
        assert_eq!(bits.next_set_after(3), Some(130));
    }

    #[test]
    fn bitset_slide_mapper_counts_duplicate_reference_hashes() {
        let query_hashes = vec![10, 20, 30];
        let ref_universe = vec![
            Minimizer {
                hash: 20,
                seq_id: 0,
                wpos: 1,
            },
            Minimizer {
                hash: 20,
                seq_id: 0,
                wpos: 2,
            },
        ];
        let (coords, indexed) = build_indexed_minimizers(&query_hashes, &ref_universe);
        let mut mapper = BitsetBottomSketchSlideMapper::new(&query_hashes, &coords);

        mapper.insert_ref(indexed[0].coord_idx);
        assert_eq!(mapper.shared(), 1);
        mapper.insert_ref(indexed[1].coord_idx);
        assert_eq!(mapper.shared(), 1);
        mapper.delete_ref(indexed[0].coord_idx);
        assert_eq!(mapper.shared(), 1);
        mapper.delete_ref(indexed[1].coord_idx);
        assert_eq!(mapper.shared(), 0);
    }

    #[test]
    fn l2_does_not_score_window_after_exclusive_end() -> Result<()> {
        let query = QuerySketch {
            fragment_id: 0,
            len: 10,
            unique_hashes: vec![10, 20, 30],
            unique_seeds: Vec::new(),
            distance_table: Arc::from(build_distance_table(3, 1)),
        };
        let reference = ReferenceIndex {
            genomes: vec![GenomeInfo {
                path: PathBuf::from("ref.fa"),
                length: 100,
            }],
            contigs: vec![ContigInfo {
                name: "ref".to_string(),
                len: 100,
                genome_id: 0,
            }],
            minimizers: vec![
                Minimizer {
                    hash: 10,
                    seq_id: 0,
                    wpos: 0,
                },
                Minimizer {
                    hash: 20,
                    seq_id: 0,
                    wpos: 5,
                },
                Minimizer {
                    hash: 30,
                    seq_id: 0,
                    wpos: 11,
                },
            ],
            contig_ranges: vec![0..3],
            lookup: CompactLookupIndex {
                ranges: Vec::new(),
                hits: Vec::new(),
                range_slots: Vec::new(),
            },
            freq_threshold: usize::MAX,
        };
        let config = AniConfig {
            kmer_size: 1,
            fragment_len: 10,
            min_identity: 0.0,
            window_size: Some(1),
            ..AniConfig::default()
        };
        let candidate = crate::candidate_window::L1Candidate {
            seq_id: 0,
            range_start: 0,
            range_end: 1,
        };

        let (mapping, stats) = do_l2_mapping(&query, candidate, &reference, &config, 1)?;

        assert_eq!(stats.windows, 0);
        assert!(mapping.is_none());
        Ok(())
    }

    fn deterministic_dna(len: usize) -> String {
        let mut x = 0x1234_5678_9abc_def0u64;
        let mut out = String::with_capacity(len);
        for _ in 0..len {
            x = splitmix64(x);
            out.push(match x & 3 {
                0 => 'A',
                1 => 'C',
                2 => 'G',
                _ => 'T',
            });
        }
        out
    }
}
