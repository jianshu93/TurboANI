use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result};
use needletail::{parse_fastx_file, Sequence};
use plotters::coord::Shift;
use plotters::prelude::*;
use rayon::prelude::*;
use simd_minimizers::packed_seq::{PackedSeqVec, SeqVec};
use tab_hash::Tab64Twisted;

mod chaining;

type HashValue = u64;
type SeqId = usize;
type Offset = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimizerMode {
    Simd,
    FastAni,
}

impl MinimizerMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Simd => "simd",
            Self::FastAni => "fastani",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FastAniConfig {
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
}

impl Default for FastAniConfig {
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
        }
    }
}

impl FastAniConfig {
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

    fn validate(&self) -> Result<()> {
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
        let w = self.resolved_window_size();
        anyhow::ensure!(w > 0, "minimizer window size must be positive");
        anyhow::ensure!(
            self.fragment_len >= self.kmer_size + w - 1,
            "fragment length must be at least k + w - 1"
        );
        Ok(())
    }
}

fn simd_compatible_window_size(k: usize, w: usize) -> usize {
    if (k + w - 1) % 2 == 1 {
        w
    } else if w > 1 {
        w - 1
    } else {
        w + 1
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
struct ContigInfo {
    name: String,
    len: usize,
    genome_id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Minimizer {
    hash: HashValue,
    seq_id: SeqId,
    wpos: Offset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MinimizerHit {
    seq_id: SeqId,
    wpos: Offset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PackedMinimizerHit {
    seq_id: u32,
    wpos: u32,
}

impl PackedMinimizerHit {
    fn new(seq_id: SeqId, wpos: Offset) -> Result<Self> {
        Ok(Self {
            seq_id: u32_checked(seq_id, "reference contig id")?,
            wpos: u32_checked(wpos, "reference minimizer position")?,
        })
    }

    fn seq_id(self) -> SeqId {
        self.seq_id as usize
    }

    fn wpos(self) -> Offset {
        self.wpos as usize
    }

    fn unpack(self) -> MinimizerHit {
        MinimizerHit {
            seq_id: self.seq_id(),
            wpos: self.wpos(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LookupRange {
    hash: HashValue,
    start: u32,
    len: u32,
}

#[derive(Debug)]
struct CompactLookupIndex {
    ranges: Vec<LookupRange>,
    hits: Vec<PackedMinimizerHit>,
    range_slots: Vec<u32>,
}

#[derive(Debug, Clone)]
struct GenomeInfo {
    path: PathBuf,
    length: usize,
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
struct ReferenceIndex {
    genomes: Vec<GenomeInfo>,
    contigs: Vec<ContigInfo>,
    minimizers: Vec<Minimizer>,
    contig_ranges: Vec<std::ops::Range<usize>>,
    lookup: CompactLookupIndex,
    freq_threshold: usize,
}

#[derive(Debug, Clone)]
struct QueryFragment {
    seq_id: usize,
    global_start: usize,
    seq: Vec<u8>,
}

#[derive(Debug, Clone)]
struct QueryFileData {
    path: PathBuf,
    genome_len: usize,
    fragments: Vec<QueryFragment>,
}

#[derive(Debug, Clone, Copy)]
struct L1Candidate {
    seq_id: SeqId,
    range_start: Offset,
    range_end: Offset,
}

#[derive(Debug, Clone)]
struct QuerySketch {
    fragment_id: usize,
    len: usize,
    unique_hashes: Vec<HashValue>,
    unique_seeds: Vec<QuerySeed>,
    distance_table: Arc<[DistanceEstimate]>,
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

    fn get(&self, hash: HashValue) -> Option<&[PackedMinimizerHit]> {
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

#[derive(Debug, Clone, Copy)]
struct QuerySeed {
    hash: HashValue,
    qpos: Offset,
}

#[derive(Debug, Clone, Copy)]
struct DistanceEstimate {
    identity: f64,
    identity_upper_bound: f64,
}

#[derive(Debug)]
struct DistanceTableCache {
    kmer_size: usize,
    tables: Vec<OnceLock<Arc<[DistanceEstimate]>>>,
    overflow: Mutex<HashMap<usize, Arc<[DistanceEstimate]>>>,
}

impl DistanceTableCache {
    fn new(kmer_size: usize, max_sketch_size: usize) -> Self {
        Self {
            kmer_size,
            tables: std::iter::repeat_with(OnceLock::new)
                .take(max_sketch_size + 1)
                .collect(),
            overflow: Mutex::new(HashMap::new()),
        }
    }

    fn table_for(&self, sketch_size: usize) -> Arc<[DistanceEstimate]> {
        if let Some(cell) = self.tables.get(sketch_size) {
            return Arc::clone(cell.get_or_init(|| {
                Arc::<[DistanceEstimate]>::from(build_distance_table(sketch_size, self.kmer_size))
            }));
        }

        let mut overflow = self
            .overflow
            .lock()
            .expect("distance table cache mutex poisoned");
        if let Some(table) = overflow.get(&sketch_size) {
            return Arc::clone(table);
        }

        let table =
            Arc::<[DistanceEstimate]>::from(build_distance_table(sketch_size, self.kmer_size));
        overflow.insert(sketch_size, Arc::clone(&table));
        table
    }
}

fn build_distance_table(sketch_size: usize, kmer_size: usize) -> Vec<DistanceEstimate> {
    if sketch_size == 0 {
        return Vec::new();
    }

    let mut table = Vec::with_capacity(sketch_size + 1);
    for shared in 0..=sketch_size {
        let best_jaccard = shared as f64 / sketch_size as f64;
        let mash_dist = j2md(best_jaccard, kmer_size);
        let mash_dist_lower_bound = md_lower_bound(mash_dist, sketch_size, kmer_size, 0.9);
        table.push(DistanceEstimate {
            identity: 100.0 * (1.0 - mash_dist),
            identity_upper_bound: 100.0 * (1.0 - mash_dist_lower_bound),
        });
    }
    table
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MappingResult {
    query_seq_id: usize,
    query_len: usize,
    ref_seq_id: SeqId,
    ref_start: Offset,
    ref_end: Offset,
    identity: f64,
    identity_upper_bound: f64,
    conserved_sketches: usize,
    sketch_size: usize,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ShortMapping {
    ref_seq_id: SeqId,
    genome_id: usize,
    query_seq_id: usize,
    ref_start: Offset,
    query_start: Offset,
    map_ref_pos_bin: Offset,
    identity: f64,
}

pub fn compare_paths(
    query_paths: &[PathBuf],
    ref_paths: &[PathBuf],
    config: &FastAniConfig,
) -> Result<Vec<AniResult>> {
    Ok(compare_paths_with_timing(query_paths, ref_paths, config)?.results)
}

pub fn compare_paths_with_timing(
    query_paths: &[PathBuf],
    ref_paths: &[PathBuf],
    config: &FastAniConfig,
) -> Result<RunOutput> {
    config.validate()?;
    let total_start = Instant::now();
    let window_size = config.resolved_window_size();
    let tab_hasher = deterministic_tab64_twisted(config.tab_hash_seed);
    let (reference, reference_timing) =
        ReferenceIndex::build(ref_paths, config, window_size, &tab_hasher)?;
    let distance_cache = DistanceTableCache::new(config.kmer_size, config.fragment_len);

    let per_query = query_paths
        .par_iter()
        .map(|path| {
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
        })
        .collect::<Result<Vec<_>>>()?;

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
    config: &FastAniConfig,
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

    for (chunk_idx, ref_chunk) in ref_paths.chunks(split_size).enumerate() {
        log::debug!(
            "phase=start_split_reference_chunk chunk={} split_count={} refs={} total_refs={}",
            chunk_idx + 1,
            split_count,
            ref_chunk.len(),
            ref_paths.len()
        );
        let chunk_run = compare_paths_with_timing(query_paths, ref_chunk, config)?;
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

pub fn write_pair_visualization_pdf(
    query_path: impl AsRef<Path>,
    ref_path: impl AsRef<Path>,
    config: &FastAniConfig,
    output_path: impl AsRef<Path>,
) -> Result<Vec<AniResult>> {
    config.validate()?;
    let query_path = query_path.as_ref();
    let ref_path = ref_path.as_ref();
    let output_path = output_path.as_ref();
    let window_size = config.resolved_window_size();
    let tab_hasher = deterministic_tab64_twisted(config.tab_hash_seed);
    let ref_paths = vec![ref_path.to_path_buf()];
    let (reference, _) = ReferenceIndex::build(&ref_paths, config, window_size, &tab_hasher)?;
    let distance_cache = DistanceTableCache::new(config.kmer_size, config.fragment_len);
    let query = read_query_file(query_path, config)?;
    let (mappings, _) = map_query_file(&query, &reference, config, window_size, &distance_cache)?;

    // The raw L2 mappings can contain several candidate target regions for one
    // query fragment. The ANI calculation subsequently keeps one best mapping
    // per query fragment/reference genome and then one best query mapping per
    // reference-position bin. Use the same two-way filtered set for plotting so
    // the conserved view matches the mappings that contribute to ANI instead of
    // displaying every raw L2 candidate hit.
    let visualization_mappings = select_best_visualization_mappings(&mappings, &reference, config);
    let points = visualization_points(&query, &reference, &visualization_mappings);
    let results = compute_ani_results(&query, &reference, mappings, config);
    draw_pair_visualization_pdf(
        output_path,
        query_path,
        ref_path,
        &query,
        &reference,
        &results,
        &points,
    )?;
    Ok(results)
}

#[derive(Debug, Clone)]
struct VisualizationPoint {
    query_start: f64,
    query_end: f64,
    ref_start: f64,
    ref_end: f64,
    identity: f64,
}

impl VisualizationPoint {
    fn query_mid(&self) -> f64 {
        (self.query_start + self.query_end) / 2.0
    }

    fn ref_mid(&self) -> f64 {
        (self.ref_start + self.ref_end) / 2.0
    }
}

fn visualization_points(
    query: &QueryFileData,
    reference: &ReferenceIndex,
    mappings: &[MappingResult],
) -> Vec<VisualizationPoint> {
    let ref_offsets = reference_contig_offsets(reference, 0);
    let mut points = mappings
        .iter()
        .filter_map(|mapping| {
            if reference.contigs[mapping.ref_seq_id].genome_id != 0 {
                return None;
            }
            let query_fragment = query.fragments.get(mapping.query_seq_id)?;
            let ref_offset = *ref_offsets.get(mapping.ref_seq_id)?;
            let query_start = query_fragment.global_start as f64;
            let ref_start = ref_offset as f64 + mapping.ref_start as f64;
            Some(VisualizationPoint {
                query_start,
                query_end: query_start + mapping.query_len as f64,
                ref_start,
                ref_end: ref_offset as f64 + (mapping.ref_end + 1) as f64,
                identity: mapping.identity,
            })
        })
        .collect::<Vec<_>>();
    points.sort_by(|a, b| {
        cmp_f64(a.query_start, b.query_start).then_with(|| cmp_f64(a.ref_start, b.ref_start))
    });
    points
}

fn select_best_visualization_mappings(
    mappings: &[MappingResult],
    reference: &ReferenceIndex,
    config: &FastAniConfig,
) -> Vec<MappingResult> {
    #[derive(Debug, Clone, Copy)]
    struct IndexedVisualMapping {
        mapping_index: usize,
        ref_seq_id: usize,
        genome_id: usize,
        query_seq_id: usize,
        ref_start: usize,
        map_ref_pos_bin: usize,
        identity: f64,
    }

    let mut one_way_candidates = mappings
        .iter()
        .enumerate()
        .map(|(mapping_index, mapping)| IndexedVisualMapping {
            mapping_index,
            ref_seq_id: mapping.ref_seq_id,
            genome_id: reference.contigs[mapping.ref_seq_id].genome_id,
            query_seq_id: mapping.query_seq_id,
            ref_start: mapping.ref_start,
            map_ref_pos_bin: mapping.ref_start / config.fragment_len.saturating_sub(20).max(1),
            identity: mapping.identity,
        })
        .collect::<Vec<_>>();

    // This is the same ordering and "keep last" rule used by
    // compute_ani_results(): the last record in each genome/query-fragment group
    // is the best identity, with reference contig and position as tie breakers.
    one_way_candidates.sort_unstable_by(|a, b| {
        (a.genome_id, a.query_seq_id)
            .cmp(&(b.genome_id, b.query_seq_id))
            .then_with(|| cmp_f64(a.identity, b.identity))
            .then_with(|| a.ref_seq_id.cmp(&b.ref_seq_id))
            .then_with(|| a.ref_start.cmp(&b.ref_start))
    });

    let mut one_way = Vec::<IndexedVisualMapping>::new();
    for mapping in one_way_candidates {
        if let Some(last) = one_way.last_mut() {
            if last.genome_id == mapping.genome_id && last.query_seq_id == mapping.query_seq_id {
                *last = mapping;
                continue;
            }
        }
        one_way.push(mapping);
    }

    // Apply the same reciprocal/reference-bin uniqueness filter used by the ANI
    // reducer. This prevents many query fragments from being drawn onto the same
    // target region and makes the plotted mapped-fragment count correspond to the
    // two-way set that contributes to ANI.
    one_way.sort_unstable_by(|a, b| {
        (a.ref_seq_id, a.map_ref_pos_bin)
            .cmp(&(b.ref_seq_id, b.map_ref_pos_bin))
            .then_with(|| cmp_f64(a.identity, b.identity))
    });

    let mut two_way = Vec::<IndexedVisualMapping>::new();
    for mapping in one_way {
        if let Some(last) = two_way.last_mut() {
            if last.ref_seq_id == mapping.ref_seq_id
                && last.map_ref_pos_bin == mapping.map_ref_pos_bin
            {
                *last = mapping;
                continue;
            }
        }
        two_way.push(mapping);
    }

    two_way
        .into_iter()
        .map(|mapping| mappings[mapping.mapping_index].clone())
        .collect()
}

fn reference_contig_offsets(reference: &ReferenceIndex, genome_id: usize) -> Vec<usize> {
    let mut offsets = vec![0usize; reference.contigs.len()];
    let mut offset = 0usize;
    for (seq_id, contig) in reference.contigs.iter().enumerate() {
        if contig.genome_id == genome_id {
            offsets[seq_id] = offset;
            offset += contig.len;
        }
    }
    offsets
}

fn draw_pair_visualization_pdf(
    output_path: &Path,
    query_path: &Path,
    ref_path: &Path,
    query: &QueryFileData,
    reference: &ReferenceIndex,
    results: &[AniResult],
    points: &[VisualizationPoint],
) -> Result<()> {
    let paths = visualization_output_paths(output_path);
    let query_mbp = (query.genome_len as f64 / 1_000_000.0).max(0.001);
    let ref_mbp = (reference.genomes[0].length as f64 / 1_000_000.0).max(0.001);
    let ani_label = ani_label(results);

    let combined_svg = render_combined_visualization_svg(
        query_path, ref_path, query_mbp, ref_mbp, &ani_label, results, points,
    )?;
    write_vector_plot(&combined_svg, &paths.combined_svg, &paths.combined_pdf)?;

    let map_svg =
        render_map_visualization_svg(query_path, ref_path, query_mbp, ref_mbp, &ani_label, points)?;
    write_vector_plot(&map_svg, &paths.map_svg, &paths.map_pdf)?;

    let identity_svg = render_identity_visualization_svg(query_mbp, results, points)?;
    write_vector_plot(&identity_svg, &paths.identity_svg, &paths.identity_pdf)?;

    let conserved_svg = render_conserved_visualization_svg(
        query_path, ref_path, query_mbp, ref_mbp, &ani_label, points,
    )?;
    write_vector_plot(&conserved_svg, &paths.conserved_svg, &paths.conserved_pdf)?;

    Ok(())
}

#[derive(Debug, Clone)]
struct VisualizationOutputPaths {
    combined_pdf: PathBuf,
    combined_svg: PathBuf,
    map_pdf: PathBuf,
    map_svg: PathBuf,
    identity_pdf: PathBuf,
    identity_svg: PathBuf,
    conserved_pdf: PathBuf,
    conserved_svg: PathBuf,
}

fn visualization_output_paths(output_path: &Path) -> VisualizationOutputPaths {
    VisualizationOutputPaths {
        combined_pdf: output_path.to_path_buf(),
        combined_svg: output_path.with_extension("svg"),
        map_pdf: visualization_sidecar_path(output_path, "map", "pdf"),
        map_svg: visualization_sidecar_path(output_path, "map", "svg"),
        identity_pdf: visualization_sidecar_path(output_path, "identity", "pdf"),
        identity_svg: visualization_sidecar_path(output_path, "identity", "svg"),
        conserved_pdf: visualization_sidecar_path(output_path, "conserved", "pdf"),
        conserved_svg: visualization_sidecar_path(output_path, "conserved", "svg"),
    }
}

fn visualization_sidecar_path(output_path: &Path, suffix: &str, extension: &str) -> PathBuf {
    let stem = output_path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("visualization");
    let mut path = output_path.to_path_buf();
    path.set_file_name(format!("{stem}.{suffix}.{extension}"));
    path
}

fn write_vector_plot(svg: &str, svg_path: &Path, pdf_path: &Path) -> Result<()> {
    ensure_parent_dir(svg_path)?;
    ensure_parent_dir(pdf_path)?;
    fs::write(svg_path, svg)
        .with_context(|| format!("failed to write SVG {}", svg_path.display()))?;

    let mut options = svg2pdf::usvg::Options::default();
    options.fontdb_mut().load_system_fonts();
    let tree = svg2pdf::usvg::Tree::from_str(svg, &options)
        .map_err(|e| anyhow::anyhow!("failed to parse plot SVG before PDF conversion: {e}"))?;
    let pdf = svg2pdf::to_pdf(
        &tree,
        svg2pdf::ConversionOptions::default(),
        svg2pdf::PageOptions::default(),
    )
    .map_err(|e| anyhow::anyhow!("failed to convert SVG plot to PDF: {e}"))?;
    fs::write(pdf_path, pdf)
        .with_context(|| format!("failed to write PDF {}", pdf_path.display()))?;
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    Ok(())
}

fn render_combined_visualization_svg(
    query_path: &Path,
    ref_path: &Path,
    query_mbp: f64,
    ref_mbp: f64,
    ani_label: &str,
    results: &[AniResult],
    points: &[VisualizationPoint],
) -> Result<String> {
    let mut svg = String::new();
    {
        let root = SVGBackend::with_string(&mut svg, (500, 720)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| anyhow::anyhow!("failed to initialize SVG drawing area: {e:?}"))?;
        let areas = root.split_evenly((2, 1));
        draw_map_chart(
            &areas[0], query_path, ref_path, query_mbp, ref_mbp, ani_label, points,
        )?;
        draw_identity_chart(&areas[1], query_mbp, results, points)?;
        root.present()
            .map_err(|e| anyhow::anyhow!("failed to finalize SVG plot: {e:?}"))?;
    }
    Ok(svg)
}

fn render_map_visualization_svg(
    query_path: &Path,
    ref_path: &Path,
    query_mbp: f64,
    ref_mbp: f64,
    ani_label: &str,
    points: &[VisualizationPoint],
) -> Result<String> {
    let mut svg = String::new();
    {
        let root = SVGBackend::with_string(&mut svg, (500, 360)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| anyhow::anyhow!("failed to initialize map SVG drawing area: {e:?}"))?;
        draw_map_chart(
            &root, query_path, ref_path, query_mbp, ref_mbp, ani_label, points,
        )?;
        root.present()
            .map_err(|e| anyhow::anyhow!("failed to finalize map SVG plot: {e:?}"))?;
    }
    Ok(svg)
}

fn render_identity_visualization_svg(
    query_mbp: f64,
    results: &[AniResult],
    points: &[VisualizationPoint],
) -> Result<String> {
    let mut svg = String::new();
    {
        let root = SVGBackend::with_string(&mut svg, (500, 360)).into_drawing_area();
        root.fill(&WHITE).map_err(|e| {
            anyhow::anyhow!("failed to initialize identity SVG drawing area: {e:?}")
        })?;
        draw_identity_chart(&root, query_mbp, results, points)?;
        root.present()
            .map_err(|e| anyhow::anyhow!("failed to finalize identity SVG plot: {e:?}"))?;
    }
    Ok(svg)
}

fn render_conserved_visualization_svg(
    query_path: &Path,
    ref_path: &Path,
    query_mbp: f64,
    ref_mbp: f64,
    ani_label: &str,
    points: &[VisualizationPoint],
) -> Result<String> {
    // The conserved view uses native SVG cubic Bezier ribbons rather than
    // Plotters PathElement polylines. svg2pdf preserves these native paths in
    // the generated PDF, so the SVG and PDF have the same smooth geometry.
    const WIDTH: f64 = 1100.0;
    const HEIGHT: f64 = 520.0;
    const LEFT: f64 = 26.0;
    const RIGHT: f64 = 26.0;
    const QUERY_Y: f64 = 142.0;
    const REF_Y: f64 = 408.0;
    const TRACK_HEIGHT: f64 = 28.0;
    const MIN_RIBBON_PX: f64 = 0.55;

    let plot_width = WIDTH - LEFT - RIGHT;
    let max_mbp = query_mbp.max(ref_mbp).max(0.001);
    let x_scale = plot_width / max_mbp;
    let query_link_y = QUERY_Y + TRACK_HEIGHT * 0.5;
    let ref_link_y = REF_Y - TRACK_HEIGHT * 0.5;

    let mut svg = String::with_capacity(256 * 1024 + points.len() * 280);

    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">"#
    )?;

    writeln!(svg, r#"<rect width="100%" height="100%" fill="white"/>"#)?;

    // Higher-identity ribbons are drawn first. Lower-identity ribbons are drawn
    // afterward, but their lower opacity keeps underlying strong alignments visible.
    let mut ordered_links = points.iter().collect::<Vec<_>>();
    ordered_links.sort_unstable_by(|a, b| cmp_f64(b.identity, a.identity));

    writeln!(svg, r#"<g id="mapping-ribbons">"#)?;

    for point in ordered_links {
        let mut q0 = LEFT + (point.query_start / 1_000_000.0) * x_scale;

        let mut q1 = LEFT + (point.query_end / 1_000_000.0) * x_scale;

        let mut r0 = LEFT + (point.ref_start / 1_000_000.0) * x_scale;

        let mut r1 = LEFT + (point.ref_end / 1_000_000.0) * x_scale;

        normalize_visible_interval(&mut q0, &mut q1, MIN_RIBBON_PX);

        normalize_visible_interval(&mut r0, &mut r1, MIN_RIBBON_PX);

        // Fixed vertical control levels keep every ribbon in the same curve
        // family and prevent inconsistent bends between neighboring mappings.
        let dy = ref_link_y - query_link_y;
        let control_offset = dy * 0.43;
        let c1y = query_link_y + control_offset;
        let c2y = ref_link_y - control_offset;

        let (red, green, blue, opacity) = identity_flame_style(point.identity);

        writeln!(
            svg,
            concat!(
                r#"<path d="M {q0:.4} {qy:.4} "#,
                r#"C {q0:.4} {c1y:.4}, {r0:.4} {c2y:.4}, {r0:.4} {ry:.4} "#,
                r#"L {r1:.4} {ry:.4} "#,
                r#"C {r1:.4} {c2y:.4}, {q1:.4} {c1y:.4}, {q1:.4} {qy:.4} Z" "#,
                r#"fill="rgb({red},{green},{blue})" "#,
                r#"fill-opacity="{opacity:.3}" stroke="none"/>"#
            ),
            q0 = q0,
            q1 = q1,
            r0 = r0,
            r1 = r1,
            qy = query_link_y,
            ry = ref_link_y,
            c1y = c1y,
            c2y = c2y,
            red = red,
            green = green,
            blue = blue,
            opacity = opacity,
        )?;
    }

    writeln!(svg, "</g>")?;

    // Neutral genome tracks expose unaligned regions as gray gaps.
    let query_width = query_mbp * x_scale;
    let ref_width = ref_mbp * x_scale;

    writeln!(
        svg,
        r##"<rect x="{LEFT:.3}" y="{:.3}" width="{query_width:.3}" height="{TRACK_HEIGHT:.3}" fill="#d6dbe0"/>"##,
        QUERY_Y - TRACK_HEIGHT * 0.5,
    )?;

    writeln!(
        svg,
        r##"<rect x="{LEFT:.3}" y="{:.3}" width="{ref_width:.3}" height="{TRACK_HEIGHT:.3}" fill="#d6dbe0"/>"##,
        REF_Y - TRACK_HEIGHT * 0.5,
    )?;

    // Aligned regions use the query-position color gradient. Each linked
    // reference block receives the same query-derived color.
    writeln!(svg, r#"<g id="aligned-blocks">"#)?;

    for point in points {
        let query_mid = 0.5 * (point.query_start + point.query_end);

        let RGBColor(red, green, blue) = genome_position_color(query_mid, query_mbp * 1_000_000.0);

        let mut q0 = LEFT + (point.query_start / 1_000_000.0) * x_scale;

        let mut q1 = LEFT + (point.query_end / 1_000_000.0) * x_scale;

        let mut r0 = LEFT + (point.ref_start / 1_000_000.0) * x_scale;

        let mut r1 = LEFT + (point.ref_end / 1_000_000.0) * x_scale;

        normalize_visible_interval(&mut q0, &mut q1, MIN_RIBBON_PX);

        normalize_visible_interval(&mut r0, &mut r1, MIN_RIBBON_PX);

        writeln!(
            svg,
            r#"<rect x="{q0:.4}" y="{:.3}" width="{:.4}" height="{TRACK_HEIGHT:.3}" fill="rgb({red},{green},{blue})"/>"#,
            QUERY_Y - TRACK_HEIGHT * 0.5,
            q1 - q0,
        )?;

        writeln!(
            svg,
            r#"<rect x="{r0:.4}" y="{:.3}" width="{:.4}" height="{TRACK_HEIGHT:.3}" fill="rgb({red},{green},{blue})"/>"#,
            REF_Y - TRACK_HEIGHT * 0.5,
            r1 - r0,
        )?;
    }

    writeln!(svg, "</g>")?;

    // Genome labels and ANI summary.
    writeln!(
        svg,
        r#"<text x="{LEFT:.1}" y="92" font-family="sans-serif" font-size="18" fill="black">{}</text>"#,
        xml_escape(&display_name(query_path)),
    )?;

    writeln!(
        svg,
        r#"<text x="{LEFT:.1}" y="478" font-family="sans-serif" font-size="18" fill="black">{}</text>"#,
        xml_escape(&display_name(ref_path)),
    )?;

    writeln!(
        svg,
        r#"<text x="550" y="48" text-anchor="middle" font-family="sans-serif" font-size="17" fill="black">{}</text>"#,
        xml_escape(ani_label),
    )?;

    append_native_identity_colorbar(&mut svg, 826.0, 63.0, 170.0, 12.0)?;

    if points.is_empty() {
        writeln!(
            svg,
            r#"<text x="60" y="180" font-family="sans-serif" font-size="24" fill="black">No mapped fragments passed ANI thresholds</text>"#
        )?;
    }

    writeln!(svg, "</svg>")?;

    Ok(svg)
}

fn normalize_visible_interval(start: &mut f64, end: &mut f64, minimum_width: f64) {
    if *end < *start {
        std::mem::swap(start, end);
    }
    if *end - *start < minimum_width {
        let midpoint = 0.5 * (*start + *end);
        *start = midpoint - minimum_width * 0.5;
        *end = midpoint + minimum_width * 0.5;
    }
}

fn identity_flame_style(identity: f64) -> (u8, u8, u8, f64) {
    // Normalize the expected identity range of 80–100% to 0–1.
    let t = ((identity - 80.0) / 20.0).clamp(0.0, 1.0);

    // Nonlinear scaling increases visible separation between intermediate
    // and very high identities.
    let strength = t.powf(1.35);

    // Flame base color: #C16642.
    const FLAME_RED: f64 = 193.0;
    const FLAME_GREEN: f64 = 102.0;
    const FLAME_BLUE: f64 = 66.0;

    // Lower identities are blended substantially toward white.
    let pale_fraction = 0.78 * (1.0 - strength);

    let red = (FLAME_RED + (255.0 - FLAME_RED) * pale_fraction)
        .round()
        .clamp(0.0, 255.0) as u8;

    let green = (FLAME_GREEN + (255.0 - FLAME_GREEN) * pale_fraction)
        .round()
        .clamp(0.0, 255.0) as u8;

    let blue = (FLAME_BLUE + (255.0 - FLAME_BLUE) * pale_fraction)
        .round()
        .clamp(0.0, 255.0) as u8;

    // Opacity varies strongly with identity:
    // 80%  -> 0.16
    // 100% -> 0.92
    let opacity = 0.16 + 0.76 * strength;

    (red, green, blue, opacity)
}

fn xml_escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

fn append_native_identity_colorbar(
    svg: &mut String,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) -> Result<()> {
    writeln!(
        svg,
        concat!(
            r#"<defs>"#,
            r#"<linearGradient id="identity-gradient" "#,
            r#"x1="0%" x2="100%" y1="0%" y2="0%">"#,
            r#"<stop offset="0%" "#,
            r#"stop-color="rgb(241,218,208)" "#,
            r#"stop-opacity="0.16"/>"#,
            r#"<stop offset="50%" "#,
            r#"stop-color="rgb(218,160,137)" "#,
            r#"stop-opacity="0.46"/>"#,
            r#"<stop offset="100%" "#,
            r#"stop-color="rgb(193,102,66)" "#,
            r#"stop-opacity="0.92"/>"#,
            r#"</linearGradient>"#,
            r#"</defs>"#
        )
    )?;

    writeln!(
        svg,
        r#"<text x="{x:.1}" y="{:.1}" font-family="sans-serif" font-size="14" fill="black">Identity (%)</text>"#,
        y - 10.0,
    )?;

    writeln!(
        svg,
        r#"<rect x="{x:.1}" y="{y:.1}" width="{width:.1}" height="{height:.1}" fill="url(#identity-gradient)" stroke="black" stroke-width="1"/>"#
    )?;

    for (fraction, label) in [(0.0, "80"), (0.5, "90"), (1.0, "100")] {
        let tick_x = x + fraction * width;

        writeln!(
            svg,
            r#"<line x1="{tick_x:.1}" y1="{:.1}" x2="{tick_x:.1}" y2="{:.1}" stroke="black" stroke-width="1"/>"#,
            y + height,
            y + height + 5.0,
        )?;

        writeln!(
            svg,
            r#"<text x="{tick_x:.1}" y="{:.1}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="black">{label}</text>"#,
            y + height + 22.0,
        )?;
    }

    Ok(())
}

fn draw_map_chart(
    area: &DrawingArea<SVGBackend<'_>, Shift>,
    _query_path: &Path,
    _ref_path: &Path,
    query_mbp: f64,
    ref_mbp: f64,
    ani_label: &str,
    points: &[VisualizationPoint],
) -> Result<()> {
    let mut map_chart = ChartBuilder::on(area)
        .caption(
            format!("Query-reference alignment positions ({ani_label})"),
            ("sans-serif", 16),
        )
        .margin(24)
        .x_label_area_size(62)
        .y_label_area_size(78)
        .build_cartesian_2d(0.0..query_mbp, 0.0..ref_mbp)
        .map_err(|e| anyhow::anyhow!("failed to build map chart: {e:?}"))?;

    map_chart
        .configure_mesh()
        .x_desc("Query position (Mb)")
        .y_desc("Reference position (Mb)")
        .axis_desc_style(("sans-serif", 24))
        .label_style(("sans-serif", 22))
        .draw()?;

    // Draw short alignment segments rather than only points,
    // giving a MUMmer-like map.
    map_chart
        .draw_series(points.iter().map(|point| {
            PathElement::new(
                vec![
                    (
                        point.query_start / 1_000_000.0,
                        point.ref_start / 1_000_000.0,
                    ),
                    (point.query_end / 1_000_000.0, point.ref_end / 1_000_000.0),
                ],
                identity_color(point.identity).stroke_width(2),
            )
        }))
        .map_err(|e| anyhow::anyhow!("failed to draw map segments: {e:?}"))?;

    if points.is_empty() {
        area.draw(&Text::new(
            "No mapped fragments passed ANI thresholds",
            (60, 80),
            ("sans-serif", 24).into_font(),
        ))
        .map_err(|e| anyhow::anyhow!("failed to draw empty-plot label: {e:?}"))?;
    }

    Ok(())
}

fn smooth_mapping_curve(
    point: &VisualizationPoint,
    query_link_y: f64,
    ref_link_y: f64,
    max_mbp: f64,
) -> Vec<(f64, f64)> {
    // Plotters' PathElement is a polyline rather than a native Bezier path. To
    // keep neighboring links visually coherent, every mapping must therefore use
    // exactly the same normalized curve family instead of independently selected
    // Bezier handles.
    //
    // Parameterize vertical position linearly and horizontal position with a
    // seventh-order smootherstep:
    //
    //   s(t) = 35t^4 - 84t^5 + 70t^6 - 20t^7
    //
    // The first three derivatives vanish at both endpoints. Links therefore leave
    // and enter the genome tracks vertically, with zero endpoint curvature and no
    // visible shoulder. Because the same s(t) is used for every mapping, adjacent
    // colinear mappings remain parallel instead of opening artificial white wedges.
    const CURVE_STEPS: usize = 128;

    let scale = max_mbp.max(f64::EPSILON);
    let x0 = (0.5 * (point.query_start + point.query_end) / 1_000_000.0) / scale;
    let x1 = (0.5 * (point.ref_start + point.ref_end) / 1_000_000.0) / scale;
    let y0 = query_link_y;
    let y1 = ref_link_y;

    (0..=CURVE_STEPS)
        .map(|step| {
            let t = step as f64 / CURVE_STEPS as f64;
            let t2 = t * t;
            let t3 = t2 * t;
            let t4 = t2 * t2;
            let t5 = t4 * t;
            let t6 = t3 * t3;
            let t7 = t6 * t;

            let s = 35.0 * t4 - 84.0 * t5 + 70.0 * t6 - 20.0 * t7;
            let x = x0 + (x1 - x0) * s;
            let y = y0 + (y1 - y0) * t;

            (x * scale, y)
        })
        .collect()
}

fn draw_homology_ribbon_chart(
    area: &DrawingArea<SVGBackend<'_>, Shift>,
    query_path: &Path,
    ref_path: &Path,
    query_mbp: f64,
    ref_mbp: f64,
    ani_label: &str,
    points: &[VisualizationPoint],
) -> Result<()> {
    let max_mbp = query_mbp.max(ref_mbp).max(0.001);
    let query_y = 0.78;
    let ref_y = 0.22;
    let bar_half = 0.030;

    // No configure_mesh(): this conserved-region view intentionally has no axes,
    // ticks, plot frame, or grid. The MUMmer-like map remains a separate plot.
    let mut chart = ChartBuilder::on(area)
        .margin(20)
        .build_cartesian_2d(0.0..max_mbp, 0.0..1.0)
        .map_err(|e| anyhow::anyhow!("failed to build homology ribbon chart: {e:?}"))?;

    // Draw high-identity links first and lower-identity links last. The weaker,
    // paler mappings are therefore not hidden beneath the stronger mappings, while
    // the strong mappings remain visible through the links' transparency.
    let mut ordered_links = points.iter().collect::<Vec<_>>();
    ordered_links.sort_unstable_by(|a, b| cmp_f64(b.identity, a.identity));

    // Identity is encoded by link shade. Use a thin one-pixel centerline so dense
    // comparisons remain readable and individual rearrangements do not merge into
    // broad gray bands.
    chart
        .draw_series(ordered_links.into_iter().map(|point| {
            PathElement::new(
                smooth_mapping_curve(point, query_y - bar_half, ref_y + bar_half, max_mbp),
                ShapeStyle::from(&identity_link_color(point.identity).mix(0.60)).stroke_width(1),
            )
        }))
        .map_err(|e| anyhow::anyhow!("failed to draw homology links: {e:?}"))?;

    // Neutral backgrounds make unaligned sequence immediately visible.
    let unaligned_track = RGBColor(214, 219, 224);
    chart
        .draw_series([
            Rectangle::new(
                [(0.0, query_y - bar_half), (query_mbp, query_y + bar_half)],
                ShapeStyle::from(&unaligned_track).filled().stroke_width(1),
            ),
            Rectangle::new(
                [(0.0, ref_y - bar_half), (ref_mbp, ref_y + bar_half)],
                ShapeStyle::from(&unaligned_track).filled().stroke_width(1),
            ),
        ])
        .map_err(|e| anyhow::anyhow!("failed to draw genome tracks: {e:?}"))?;

    // pyGenomeViz-style positional coloring: color is determined by the query
    // coordinate, not by mapping index. Nearby query regions therefore form a
    // smooth viridis-like gradient. The linked reference interval receives the
    // exact same color, which makes translocations and cross-mapping visible.
    // Neutral track regions remain visible wherever no alignment is present.
    chart
        .draw_series(points.iter().map(|point| {
            let query_mid = 0.5 * (point.query_start + point.query_end);
            let color = genome_position_color(query_mid, query_mbp * 1_000_000.0);
            Rectangle::new(
                [
                    (point.query_start / 1_000_000.0, query_y - bar_half),
                    (point.query_end / 1_000_000.0, query_y + bar_half),
                ],
                color.filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("failed to color query aligned blocks: {e:?}"))?;

    chart
        .draw_series(points.iter().map(|point| {
            let query_mid = 0.5 * (point.query_start + point.query_end);
            let color = genome_position_color(query_mid, query_mbp * 1_000_000.0);
            Rectangle::new(
                [
                    (point.ref_start / 1_000_000.0, ref_y - bar_half),
                    (point.ref_end / 1_000_000.0, ref_y + bar_half),
                ],
                color.filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("failed to color reference aligned blocks: {e:?}"))?;

    chart
        .draw_series([
            Text::new(
                display_name(query_path),
                (0.0, query_y + 0.095),
                ("sans-serif", 18).into_font(),
            ),
            Text::new(
                display_name(ref_path),
                (0.0, ref_y - 0.115),
                ("sans-serif", 18).into_font(),
            ),
            Text::new(
                ani_label.to_owned(),
                (max_mbp * 0.5, 0.96),
                ("sans-serif", 17).into_font(),
            ),
        ])
        .map_err(|e| anyhow::anyhow!("failed to draw homology map labels: {e:?}"))?;

    // This is the requested line-identity heat indicator, not a separate heatmap.
    draw_identity_colorbar(area)?;

    if points.is_empty() {
        area.draw(&Text::new(
            "No mapped fragments passed ANI thresholds",
            (60, 80),
            ("sans-serif", 24).into_font(),
        ))
        .map_err(|e| anyhow::anyhow!("failed to draw empty homology-map label: {e:?}"))?;
    }

    Ok(())
}

fn draw_identity_chart(
    area: &DrawingArea<SVGBackend<'_>, Shift>,
    query_mbp: f64,
    results: &[AniResult],
    points: &[VisualizationPoint],
) -> Result<()> {
    let y_min = points
        .iter()
        .map(|point| point.identity)
        .fold(100.0f64, f64::min)
        .floor()
        .min(99.0)
        .max(75.0);
    let y_max = 100.1;
    let mut identity_chart = ChartBuilder::on(area)
        .caption("Fragment identity by query position", ("sans-serif", 16))
        .margin(24)
        .x_label_area_size(62)
        .y_label_area_size(78)
        .build_cartesian_2d(0.0..query_mbp, y_min..y_max)
        .map_err(|e| anyhow::anyhow!("failed to build identity chart: {e:?}"))?;
    identity_chart
        .configure_mesh()
        .x_desc("Query position (Mb)")
        .y_desc("Reference position (Mb)")
        .axis_desc_style(("sans-serif", 24))
        .label_style(("sans-serif", 22))
        .draw()?;
    identity_chart
        .draw_series(points.iter().map(|point| {
            Circle::new(
                (point.query_mid() / 1_000_000.0, point.identity),
                2,
                identity_color(point.identity).filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("failed to draw identity points: {e:?}"))?;

    if let Some(result) = results.first() {
        identity_chart
            .draw_series(LineSeries::new(
                vec![(0.0, result.ani), (query_mbp, result.ani)],
                &RED,
            ))
            .map_err(|e| anyhow::anyhow!("failed to draw ANI line: {e:?}"))?;
    }
    Ok(())
}

fn draw_conserved_chart(
    area: &DrawingArea<SVGBackend<'_>, Shift>,
    query_path: &Path,
    ref_path: &Path,
    query_mbp: f64,
    ref_mbp: f64,
    ani_label: &str,
    points: &[VisualizationPoint],
) -> Result<()> {
    draw_homology_ribbon_chart(
        area, query_path, ref_path, query_mbp, ref_mbp, ani_label, points,
    )
}

fn draw_identity_colorbar(area: &DrawingArea<SVGBackend<'_>, Shift>) -> Result<()> {
    let (area_width, _) = area.dim_in_pixel();
    let bar_width = 170i32;
    let bar_height = 12i32;
    let x0 = (area_width as i32 - 265).max(140);
    let y0 = 64i32;

    area.draw(&Text::new(
        "Identity (%)",
        (x0, y0 - 12),
        ("sans-serif", 14).into_font(),
    ))
    .map_err(|e| anyhow::anyhow!("failed to draw identity colorbar title: {e:?}"))?;

    for pixel in 0..bar_width {
        let identity = 80.0 + 20.0 * pixel as f64 / (bar_width - 1) as f64;
        area.draw(&Rectangle::new(
            [(x0 + pixel, y0), (x0 + pixel + 1, y0 + bar_height)],
            identity_link_color(identity).filled(),
        ))
        .map_err(|e| anyhow::anyhow!("failed to draw identity colorbar gradient: {e:?}"))?;
    }

    area.draw(&Rectangle::new(
        [(x0, y0), (x0 + bar_width, y0 + bar_height)],
        BLACK.stroke_width(1),
    ))
    .map_err(|e| anyhow::anyhow!("failed to draw identity colorbar frame: {e:?}"))?;

    for (fraction, label) in [(0.0, "80"), (0.5, "90"), (1.0, "100")] {
        let x = x0 + (fraction * bar_width as f64).round() as i32;
        area.draw(&PathElement::new(
            vec![(x, y0 + bar_height), (x, y0 + bar_height + 5)],
            BLACK,
        ))
        .map_err(|e| anyhow::anyhow!("failed to draw identity colorbar tick: {e:?}"))?;
        area.draw(&Text::new(
            label,
            (x - 8, y0 + bar_height + 22),
            ("sans-serif", 12).into_font(),
        ))
        .map_err(|e| anyhow::anyhow!("failed to draw identity colorbar label: {e:?}"))?;
    }
    Ok(())
}

fn ani_label(results: &[AniResult]) -> String {
    results
        .first()
        .map(|result| {
            format!(
                "ANI {:.4}%, mapped {}/{}",
                result.ani, result.mapped_fragments, result.total_query_fragments
            )
        })
        .unwrap_or_else(|| "no ANI result passed thresholds".to_string())
}

fn identity_color(identity: f64) -> HSLColor {
    let t = ((identity - 80.0) / 20.0).clamp(0.0, 1.0);
    HSLColor((220.0 - 180.0 * t) / 360.0, 0.75, 0.45)
}

fn identity_link_color(identity: f64) -> HSLColor {
    // Wide grayscale range: very pale at 80%, nearly black at 100%. The links are
    // still translucent, but this larger luminance span keeps identity differences
    // visible after alpha blending and in regions with overlapping mappings.
    let t = ((identity - 80.0) / 20.0).clamp(0.0, 1.0);
    HSLColor(0.0, 0.0, 0.92 - 0.86 * t)
}

fn genome_position_color(position: f64, genome_len: f64) -> RGBColor {
    // Viridis-like positional gradient used for aligned genome blocks.
    // The same query-derived color is applied to both ends of an alignment.
    // Stops approximate matplotlib/pyGenomeViz viridis without adding a dependency.
    const STOPS: &[(f64, (u8, u8, u8))] = &[
        (0.00, (68, 1, 84)),
        (0.13, (71, 44, 122)),
        (0.25, (59, 82, 139)),
        (0.38, (44, 113, 142)),
        (0.50, (33, 145, 140)),
        (0.63, (39, 173, 129)),
        (0.75, (92, 200, 99)),
        (0.88, (170, 220, 50)),
        (1.00, (253, 231, 37)),
    ];

    let t = if genome_len > 0.0 {
        (position / genome_len).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut right = 1usize;
    while right < STOPS.len() && t > STOPS[right].0 {
        right += 1;
    }
    if right >= STOPS.len() {
        let (r, g, b) = STOPS[STOPS.len() - 1].1;
        return RGBColor(r, g, b);
    }

    let left = right - 1;
    let (t0, (r0, g0, b0)) = STOPS[left];
    let (t1, (r1, g1, b1)) = STOPS[right];
    let u = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };

    let lerp = |a: u8, b: u8| -> u8 { (a as f64 + (b as f64 - a as f64) * u).round() as u8 };

    RGBColor(lerp(r0, r1), lerp(g0, g1), lerp(b0, b1))
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.display().to_string())
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

fn u32_checked(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{label} exceeds u32::MAX"))
}

fn lookup_slot(hash: HashValue, mask: usize) -> usize {
    (splitmix64_permute(hash) as usize) & mask
}

impl ReferenceIndex {
    fn build(
        paths: &[PathBuf],
        config: &FastAniConfig,
        window_size: usize,
        tab_hasher: &Tab64Twisted,
    ) -> Result<(Self, ReferenceTiming)> {
        anyhow::ensure!(!paths.is_empty(), "at least one reference path is required");
        let total_start = Instant::now();

        let read_start = Instant::now();
        let mut builds = paths
            .par_iter()
            .enumerate()
            .map(|(genome_id, path)| {
                read_reference_genome(path, genome_id, config, window_size, tab_hasher)
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

    fn lower_bound(&self, seq_id: SeqId, wpos: Offset) -> usize {
        let range = &self.contig_ranges[seq_id];
        let slice = &self.minimizers[range.clone()];
        range.start + slice.partition_point(|m| m.wpos < wpos)
    }

    fn contig_minimizer_bounds(&self, seq_id: SeqId) -> std::ops::Range<usize> {
        self.contig_ranges[seq_id].clone()
    }
}

fn read_reference_genome(
    path: &Path,
    genome_id: usize,
    config: &FastAniConfig,
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
fn read_query_file(path: &Path, config: &FastAniConfig) -> Result<QueryFileData> {
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

fn map_query_file(
    query: &QueryFileData,
    reference: &ReferenceIndex,
    config: &FastAniConfig,
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
    config: &FastAniConfig,
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

#[derive(Debug, Clone, Copy, Default)]
struct L1Stats {
    seed_hits: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct L2Stats {
    windows: u64,
    ref_sketches: u64,
    coord_count: usize,
    reference_minimizers: usize,
    ref_hash_ns: u128,
    ref_sketch_ns: u128,
    distance_ns: u128,
}

#[derive(Debug, Clone, Copy)]
struct IndexedMinimizer {
    wpos: Offset,
    coord_idx: usize,
}

// Maintains the classic FastANI/Mashmap bottom-k union sketch exactly as an L2
// reference window slides, avoiding OPH sketch construction per placement.
#[derive(Debug)]
struct BitsetBottomSketchSlideMapper {
    query_present: Vec<u8>,
    ref_count: Vec<u32>,
    union_bits: SummaryBitSet,
    pivot_idx: usize,
    shared: usize,
}

impl BitsetBottomSketchSlideMapper {
    fn new(query_hashes: &[HashValue], coords: &[HashValue]) -> Self {
        debug_assert!(!query_hashes.is_empty());
        let mut query_present = vec![0u8; coords.len()];
        let ref_count = vec![0u32; coords.len()];
        let mut union_bits = SummaryBitSet::new(coords.len());

        for &hash in query_hashes {
            let idx = coords
                .binary_search(&hash)
                .expect("query hash must be in coordinate universe");
            query_present[idx] = 1;
            union_bits.set(idx);
        }

        let pivot_idx = coords
            .binary_search(&query_hashes[query_hashes.len() - 1])
            .expect("query pivot must be in coordinate universe");

        Self {
            query_present,
            ref_count,
            union_bits,
            pivot_idx,
            shared: 0,
        }
    }

    #[inline]
    fn shared(&self) -> usize {
        self.shared
    }

    #[inline]
    fn insert_ref_range(&mut self, minimizers: &[IndexedMinimizer]) {
        for minimizer in minimizers {
            self.insert_ref(minimizer.coord_idx);
        }
    }

    #[inline]
    fn insert_ref(&mut self, idx: usize) {
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
    fn delete_ref(&mut self, idx: usize) {
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

#[derive(Debug)]
struct SummaryBitSet {
    words: Vec<u64>,
    summary: Vec<u64>,
}

impl SummaryBitSet {
    fn new(len: usize) -> Self {
        let word_count = len.div_ceil(64);
        let summary_count = word_count.div_ceil(64);
        Self {
            words: vec![0; word_count],
            summary: vec![0; summary_count],
        }
    }

    #[inline]
    fn set(&mut self, index: usize) {
        let word_idx = index >> 6;
        let mask = 1u64 << (index & 63);
        let old = self.words[word_idx];
        self.words[word_idx] = old | mask;
        if old == 0 {
            self.summary[word_idx >> 6] |= 1u64 << (word_idx & 63);
        }
    }

    #[inline]
    fn clear(&mut self, index: usize) {
        let word_idx = index >> 6;
        let mask = 1u64 << (index & 63);
        self.words[word_idx] &= !mask;
        if self.words[word_idx] == 0 {
            self.summary[word_idx >> 6] &= !(1u64 << (word_idx & 63));
        }
    }

    #[inline]
    fn prev_set_before(&self, index: usize) -> Option<usize> {
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
    fn next_set_after(&self, index: usize) -> Option<usize> {
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

fn build_indexed_minimizers(
    query_hashes: &[HashValue],
    reference_universe: &[Minimizer],
) -> (Vec<HashValue>, Vec<IndexedMinimizer>) {
    let mut coords = Vec::with_capacity(query_hashes.len() + reference_universe.len());
    coords.extend_from_slice(query_hashes);
    coords.extend(reference_universe.iter().map(|m| m.hash));
    coords.sort_unstable();
    coords.dedup();

    let indexed = reference_universe
        .iter()
        .map(|minimizer| IndexedMinimizer {
            wpos: minimizer.wpos,
            coord_idx: coords
                .binary_search(&minimizer.hash)
                .expect("reference hash must be in coordinate universe"),
        })
        .collect();

    (coords, indexed)
}

fn do_l1_mapping(
    query: &QuerySketch,
    reference: &ReferenceIndex,
    config: &FastAniConfig,
) -> (Vec<L1Candidate>, L1Stats) {
    if config.chain {
        return chaining::do_l1_mapping_chained(query, reference, config);
    }

    if config.minimizer_mode == MinimizerMode::FastAni {
        return do_l1_mapping_fastani_exact(query, reference, config);
    }

    chaining::do_l1_mapping_diagonal_clustered(query, reference, config)
}

fn do_l1_mapping_fastani_exact(
    query: &QuerySketch,
    reference: &ReferenceIndex,
    config: &FastAniConfig,
) -> (Vec<L1Candidate>, L1Stats) {
    let mut seed_hits = Vec::new();

    for &hash in &query.unique_hashes {
        if let Some(hits) = reference.lookup.get(hash) {
            if hits.len() < reference.freq_threshold {
                seed_hits.extend(hits.iter().copied().map(PackedMinimizerHit::unpack));
            }
        }
    }

    let minimum_hits = estimate_minimum_hits_relaxed(
        query.unique_hashes.len(),
        config.kmer_size,
        config.min_identity,
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

fn do_l2_mapping(
    query: &QuerySketch,
    candidate: L1Candidate,
    reference: &ReferenceIndex,
    config: &FastAniConfig,
    window_size: usize,
) -> Result<(Option<MappingResult>, L2Stats)> {
    do_l2_mapping_bitset_exact(query, candidate, reference, config, window_size)
}

fn do_l2_mapping_bitset_exact(
    query: &QuerySketch,
    candidate: L1Candidate,
    reference: &ReferenceIndex,
    config: &FastAniConfig,
    window_size: usize,
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
    let coord_end_abs = last_end_abs.saturating_add(1).min(contig_range.end);
    if coord_end_abs <= sw_beg_abs {
        return Ok((None, stats));
    }

    let (coords, local_minimizers) = build_indexed_minimizers(
        &query.unique_hashes,
        &reference.minimizers[sw_beg_abs..coord_end_abs],
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

    let mut slide_map = BitsetBottomSketchSlideMapper::new(&query.unique_hashes, &coords);
    slide_map.insert_ref_range(&local_minimizers[sw_beg..sw_end.min(local_minimizers.len())]);

    let mut best_shared = 0usize;
    let mut first_best_pos: Option<Offset> = None;
    let mut last_best_pos: Option<Offset> = None;

    while sw_beg < local_minimizers.len() && sw_beg < last_end && sw_pos <= candidate.range_end {
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

        if advance_by == next_beg_delta {
            slide_map.delete_ref(local_minimizers[sw_beg].coord_idx);
            sw_beg += 1;
        }
        if advance_by == next_end_delta && sw_end < local_minimizers.len() {
            slide_map.insert_ref(local_minimizers[sw_end].coord_idx);
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

fn compute_ani_results(
    query: &QueryFileData,
    reference: &ReferenceIndex,
    mappings: Vec<MappingResult>,
    config: &FastAniConfig,
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

fn sequence_minimizers(
    seq: &[u8],
    config: &FastAniConfig,
    w: usize,
    seq_id: SeqId,
    tab_hasher: &Tab64Twisted,
) -> Result<Vec<Minimizer>> {
    match config.minimizer_mode {
        MinimizerMode::Simd => {
            simd_sequence_minimizers(seq, config.kmer_size, w, seq_id, tab_hasher)
        }
        MinimizerMode::FastAni => fastani_sequence_minimizers(seq, config.kmer_size, w, seq_id),
    }
}

fn simd_sequence_minimizers(
    seq: &[u8],
    k: usize,
    w: usize,
    seq_id: SeqId,
    tab_hasher: &Tab64Twisted,
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
struct FastAniDequeEntry {
    hash: HashValue,
    pos: Offset,
    emitted: bool,
}

fn fastani_sequence_minimizers(
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
    let mut deque: std::collections::VecDeque<FastAniDequeEntry> =
        std::collections::VecDeque::new();
    let max_i = seq_upper.len() - k + 1;

    for i in 0..max_i {
        let hash_fwd = murmurhash3_x64_128_low32(&seq_upper[i..i + k], 42) as HashValue;
        let rc_start = seq_upper.len() - i - k;
        let hash_bwd = murmurhash3_x64_128_low32(&seq_rev[rc_start..rc_start + k], 42) as HashValue;

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

        deque.push_back(FastAniDequeEntry {
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

fn cmp_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

fn minimizer_token(canonical_kmer_value: u64, k: usize, tab_hasher: &Tab64Twisted) -> u64 {
    let key = canonical_kmer_value ^ ((k as u64) << 56) ^ 0xD1B5_4A32_D192_ED03;
    tab_hasher.hash(key)
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

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    splitmix64_permute(*state)
}

#[cfg(test)]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    splitmix64_permute(x)
}

fn splitmix64_permute(x: u64) -> u64 {
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn murmurhash3_x64_128_low32(key: &[u8], seed: u32) -> u32 {
    let len = key.len();
    let nblocks = len / 16;
    let mut h1 = seed as u64;
    let mut h2 = seed as u64;
    const C1: u64 = 0x87c3_7b91_1142_53d5;
    const C2: u64 = 0x4cf5_ad43_2745_937f;

    for block in 0..nblocks {
        let offset = block * 16;
        let mut k1 = u64::from_le_bytes(key[offset..offset + 8].try_into().unwrap());
        let mut k2 = u64::from_le_bytes(key[offset + 8..offset + 16].try_into().unwrap());

        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(31);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;

        h1 = h1.rotate_left(27);
        h1 = h1.wrapping_add(h2);
        h1 = h1.wrapping_mul(5).wrapping_add(0x52dc_e729);

        k2 = k2.wrapping_mul(C2);
        k2 = k2.rotate_left(33);
        k2 = k2.wrapping_mul(C1);
        h2 ^= k2;

        h2 = h2.rotate_left(31);
        h2 = h2.wrapping_add(h1);
        h2 = h2.wrapping_mul(5).wrapping_add(0x3849_5ab5);
    }

    let tail = &key[nblocks * 16..];
    let mut k1 = 0u64;
    let mut k2 = 0u64;

    if tail.len() >= 15 {
        k2 ^= (tail[14] as u64) << 48;
    }
    if tail.len() >= 14 {
        k2 ^= (tail[13] as u64) << 40;
    }
    if tail.len() >= 13 {
        k2 ^= (tail[12] as u64) << 32;
    }
    if tail.len() >= 12 {
        k2 ^= (tail[11] as u64) << 24;
    }
    if tail.len() >= 11 {
        k2 ^= (tail[10] as u64) << 16;
    }
    if tail.len() >= 10 {
        k2 ^= (tail[9] as u64) << 8;
    }
    if tail.len() >= 9 {
        k2 ^= tail[8] as u64;
        k2 = k2.wrapping_mul(C2);
        k2 = k2.rotate_left(33);
        k2 = k2.wrapping_mul(C1);
        h2 ^= k2;
    }

    if tail.len() >= 8 {
        k1 ^= (tail[7] as u64) << 56;
    }
    if tail.len() >= 7 {
        k1 ^= (tail[6] as u64) << 48;
    }
    if tail.len() >= 6 {
        k1 ^= (tail[5] as u64) << 40;
    }
    if tail.len() >= 5 {
        k1 ^= (tail[4] as u64) << 32;
    }
    if tail.len() >= 4 {
        k1 ^= (tail[3] as u64) << 24;
    }
    if tail.len() >= 3 {
        k1 ^= (tail[2] as u64) << 16;
    }
    if tail.len() >= 2 {
        k1 ^= (tail[1] as u64) << 8;
    }
    if !tail.is_empty() {
        k1 ^= tail[0] as u64;
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(31);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
    }

    h1 ^= len as u64;
    h2 ^= len as u64;

    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);

    h1 = fmix64(h1);
    h2 = fmix64(h2);

    h1 = h1.wrapping_add(h2);
    (h1 & 0xffff_ffff) as u32
}

fn fmix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    k ^ (k >> 33)
}

pub fn j2md(jaccard: f64, k: usize) -> f64 {
    if jaccard <= 0.0 {
        1.0
    } else if jaccard >= 1.0 {
        0.0
    } else {
        (-1.0 / k as f64) * ((2.0 * jaccard) / (1.0 + jaccard)).ln()
    }
}

pub fn md2j(distance: f64, k: usize) -> f64 {
    1.0 / (2.0 * (k as f64 * distance).exp() - 1.0)
}

pub fn md_lower_bound(distance: f64, sketch_size: usize, k: usize, ci: f64) -> f64 {
    if sketch_size == 0 {
        return 1.0;
    }

    let q2 = (1.0 - ci) / 2.0;
    let p = md2j(distance, k).clamp(0.0, 1.0);
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
    j2md(x as f64 / sketch_size as f64, k)
}

pub fn estimate_minimum_hits(sketch_size: usize, k: usize, percent_identity: f64) -> usize {
    let mash_dist = 1.0 - percent_identity / 100.0;
    let jaccard = md2j(mash_dist, k);
    (sketch_size as f64 * jaccard).ceil() as usize
}

pub fn estimate_minimum_hits_relaxed(sketch_size: usize, k: usize, percent_identity: f64) -> usize {
    if sketch_size == 0 {
        return 0;
    }

    let strict = estimate_minimum_hits(sketch_size, k, percent_identity);
    let mut relaxed = strict;
    for i in (0..=strict).rev() {
        let jaccard = i as f64 / sketch_size as f64;
        let d = j2md(jaccard, k);
        let d_lower = md_lower_bound(d, sketch_size, k, 0.9);
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
    let kmer_space = (alphabet_size as f64).powi(k as i32);
    let px = 1.0 / (1.0 + kmer_space / query_len as f64);
    let py = px;
    let random_jaccard = px * py / (px + py - px * py);
    let x = estimate_minimum_hits_relaxed(sketch_size, k, identity);
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
    let mut potential = vec![1usize, 2, 5];
    potential.extend((10..query_len).step_by(10));

    let optimal_sketch_size = potential
        .into_iter()
        .find(|&s| {
            estimate_pvalue(s, k, alphabet_size, identity, query_len, reference_len)
                <= pvalue_cutoff
        })
        .unwrap_or(query_len);

    ((2 * query_len) / optimal_sketch_size).clamp(1, query_len)
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
        let default_k_config = FastAniConfig {
            min_identity: 78.0,
            minimizer_mode: MinimizerMode::Simd,
            ..FastAniConfig::default()
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

        let odd_k_config = FastAniConfig {
            kmer_size: 15,
            window_size: Some(16),
            minimizer_mode: MinimizerMode::Simd,
            ..FastAniConfig::default()
        };
        assert_eq!(odd_k_config.resolved_window_size(), 15);
        assert_eq!(
            (odd_k_config.kmer_size + odd_k_config.resolved_window_size() - 1) % 2,
            1
        );

        let minimum_window_config = FastAniConfig {
            kmer_size: 16,
            window_size: Some(1),
            minimizer_mode: MinimizerMode::Simd,
            ..FastAniConfig::default()
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
        let config = FastAniConfig {
            kmer_size: 16,
            window_size: Some(17),
            minimizer_mode: MinimizerMode::FastAni,
            ..FastAniConfig::default()
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

        let config = FastAniConfig {
            kmer_size: 8,
            fragment_len: 1000,
            min_identity: 70.0,
            min_fraction: 0.0,
            window_size: Some(10),
            ..FastAniConfig::default()
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

        let config = FastAniConfig {
            kmer_size: 8,
            fragment_len: 1000,
            min_identity: 70.0,
            min_fraction: 0.0,
            window_size: Some(10),
            chain: true,
            ..FastAniConfig::default()
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
