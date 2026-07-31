pub mod candidate_window;
pub mod chaining;
pub mod compute_identity;
#[cfg(feature = "visual")]
pub mod plot;
pub mod simd_minimizer;
pub mod sliding_mapper;

mod utils;

pub use compute_identity::{
    DistanceModel, estimate_minimum_hits, estimate_minimum_hits_relaxed,
    estimate_minimum_hits_relaxed_with_model, estimate_minimum_hits_with_model, estimate_pvalue,
    estimate_pvalue_with_model, j2md, md_lower_bound, md_lower_bound_with_model, md2j,
    recommended_window_size, recommended_window_size_with_model,
};
#[cfg(feature = "visual")]
pub use plot::write_pair_visualization_pdf;
pub use simd_minimizer::MinimizerMode;
pub use utils::{
    AniConfig, AniResult, MappingCounters, QueryTiming, ReferenceTiming, RunOutput, TimingReport,
    compare_paths, compare_paths_split_with_timing, compare_paths_with_timing,
    format_timing_summary, read_path_list, write_phylip_matrix, write_results, write_timing_report,
};

pub(crate) use utils::{
    HashValue, Offset, QueryFileData, QuerySketch, ReferenceIndex, SeqId, u32_checked,
};
#[cfg(feature = "visual")]
pub(crate) use utils::{map_query_file, read_query_file};
