use std::{env, path::PathBuf};

use anyhow::{Context, Result};
use clap::{Arg, ArgAction, ArgGroup, Command};
use log::info;
use turboani::{
    AniConfig, DistanceModel, MinimizerMode, TabulationMode, TimingReport,
    compare_paths_split_with_timing, compare_paths_upper_triangle_split_with_timing,
    compare_paths_upper_triangle_with_timing, compare_paths_with_timing, format_timing_summary,
    read_path_list, write_phylip_matrix, write_results, write_results_upper_triangle,
};

fn main() -> Result<()> {
    println!("\n ************** initializing logger *****************\n");
    env_logger::Builder::from_default_env().init();
    log::info!("logger initialized from default environment");

    let m = Command::new("turboani")
        .version(env!("CARGO_PKG_VERSION"))
        .about(
            "Super-fast ANI with SIMD minimizers, tabulation hashing, compact lookup indexing, efficient candidate window screening \
             and bitset rolling bottom-s sketch order statistics over winnowed minimizers",
        )
        .arg(
            Arg::new("query")
                .short('q')
                .long("query")
                .help("Single query genome FASTA/FASTQ path")
                .value_name("QUERY")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("query-list")
                .long("ql")
                .alias("queryList")
                .help("Text file containing one query genome path per line; when supplied alone, reuse it as the reference list and compute the upper triangle")
                .value_name("QUERY_LIST")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .group(
            ArgGroup::new("query-input")
                .args(["query", "query-list"])
                .multiple(false),
        )
        .arg(
            Arg::new("reference")
                .short('r')
                .long("ref")
                .help("Single reference genome FASTA/FASTQ path")
                .value_name("REFERENCE")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("ref-list")
                .long("rl")
                .alias("refList")
                .help("Text file containing one reference genome path per line; when supplied alone, reuse it as the query list and compute the upper triangle")
                .value_name("REFERENCE_LIST")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .group(
            ArgGroup::new("reference-input")
                .args(["reference", "ref-list"])
                .multiple(false),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output ANI table")
                .value_name("OUTPUT")
                .required(true)
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("kmer-size")
                .short('k')
                .long("kmer")
                .help("K-mer size")
                .value_name("KMER_SIZE")
                .default_value("16")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("fragment-len")
                .long("fragLen")
                .help("Query fragment length")
                .value_name("FRAGMENT_LENGTH")
                .default_value("3000")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("min-identity")
                .long("minIdentity")
                .help("Minimum identity percentage")
                .value_name("MIN_IDENTITY")
                .default_value("80")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("min-fraction")
                .long("minFraction")
                .help("Minimum aligned fraction")
                .value_name("MIN_FRACTION")
                .default_value("0.2")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("p-value")
                .long("pValue")
                .help("P-value used to derive the minimizer window size")
                .value_name("P_VALUE")
                .default_value("0.001")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("reference-size")
                .long("referenceSize")
                .help("Reference size used to derive the minimizer window size")
                .value_name("REFERENCE_SIZE")
                .default_value("5000000")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("window-size")
                .long("windowSize")
                .help("Override the p-value-derived minimizer window size")
                .value_name("WINDOW_SIZE")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("ignore-top-percent")
                .long("ignoreTopPercent")
                .help("Ignore the most frequent minimizer tokens during L1 lookup")
                .value_name("PERCENT")
                .default_value("0.0")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("tab-seed")
                .long("tabSeed")
                .help("Deterministic tabulation-hash seed")
                .value_name("SEED")
                .default_value("42")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("simple-tabulation")
                .long("simpleTabulation")
                .help("Use simple tabulation hashing instead of the default twisted tabulation hashing")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("diag-bin")
                .long("diagBin")
                .help("Diagonal clustering bin width in bases")
                .value_name("BASES")
                .default_value("1000")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("diag-band")
                .long("diagBand")
                .help("Candidate reference half-band emitted by diagonal clustering")
                .value_name("BASES")
                .default_value("500")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("chain-max-predecessors")
                .long("chainMaxPredecessors")
                .help("Maximum previous anchors examined per anchor in heuristic chaining")
                .value_name("N")
                .default_value("96")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("chain-max-gap")
                .long("chainMaxGap")
                .help("Maximum reference/query gap allowed in heuristic chaining")
                .value_name("BASES")
                .default_value("5000")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("chain-diag-tolerance")
                .long("chainDiagTolerance")
                .help("Maximum diagonal drift allowed in heuristic chaining")
                .value_name("BASES")
                .default_value("1200")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("chainx")
                .long("chainX")
                .help(
                    "Use optimal ChainX refinement after diagonal clustering for L1 candidate \
                     screening; the default is heuristic chaining")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .help("Distance model: 0=Poisson Mash equation, 1=binomial Mash equation")
                .value_name("MODEL")
                .default_value("0")
                .value_parser(clap::value_parser!(u8)),
        )
        .arg(
            Arg::new("matrix")
                .long("matrix")
                .help(
                    "Also write a Phylip lower-triangular matrix to <output>.matrix; \
                     reciprocal ANI values are averaged",
                )
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("triangle")
                .long("triangle")
                .help(
                    "For identical ordered --ql and --rl lists, compute and write only pairs with query index < reference index and omit self comparisons; this mode is automatic when only --ql or --rl is provided",
                )
                .conflicts_with("matrix")
                .action(ArgAction::SetTrue),
        );
    let m = add_visualize_arg(m)
        .arg(
            Arg::new("split")
                .long("split")
                .help(
                    "Split the reference list into this many chunks and map all queries against \
                     one chunk at a time; exact only with --ignoreTopPercent 0.0",
                )
                .value_name("REFERENCE_CHUNKS")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("threads")
                .short('t')
                .long("threads")
                .help("Number of threads. Defaults to all logical cores")
                .value_name("THREADS")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("no-progress")
                .long("no-progress")
                .help("Disable progress bars")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    let query_path = m.get_one::<PathBuf>("query").cloned();
    let query_list_path = m.get_one::<PathBuf>("query-list").cloned();
    let reference_path = m.get_one::<PathBuf>("reference").cloned();
    let ref_list_path = m.get_one::<PathBuf>("ref-list").cloned();
    let query_list_mode = query_list_path.is_some();
    let ref_list_mode = ref_list_path.is_some();
    let output_path = m.get_one::<PathBuf>("output").unwrap();
    let kmer_size = *m.get_one::<usize>("kmer-size").unwrap();
    let fragment_len = *m.get_one::<usize>("fragment-len").unwrap();
    let min_identity = *m.get_one::<f64>("min-identity").unwrap();
    let min_fraction = *m.get_one::<f64>("min-fraction").unwrap();
    let p_value = *m.get_one::<f64>("p-value").unwrap();
    let reference_size = *m.get_one::<u64>("reference-size").unwrap();
    let window_size = m.get_one::<usize>("window-size").copied();
    let ignore_top_percent = *m.get_one::<f64>("ignore-top-percent").unwrap();
    let tab_seed = *m.get_one::<u64>("tab-seed").unwrap();
    let tabulation_mode = if m.get_flag("simple-tabulation") {
        TabulationMode::Simple
    } else {
        TabulationMode::Twisted
    };
    let chainx = m.get_flag("chainx");
    let diag_cluster_bin = *m.get_one::<usize>("diag-bin").unwrap();
    let diag_cluster_band = *m.get_one::<usize>("diag-band").unwrap();
    let chain_max_predecessors = *m.get_one::<usize>("chain-max-predecessors").unwrap();
    let chain_max_gap = *m.get_one::<usize>("chain-max-gap").unwrap();
    let chain_diag_tolerance = *m.get_one::<usize>("chain-diag-tolerance").unwrap();
    let distance_model = DistanceModel::from_code(*m.get_one::<u8>("model").unwrap())
        .context("--model must be 0 for Poisson or 1 for binomial")?;
    let matrix = m.get_flag("matrix");
    let explicit_triangle = m.get_flag("triangle");
    #[cfg(feature = "visual")]
    let visualize_path = m.get_one::<PathBuf>("visualize");
    let split_count = m.get_one::<usize>("split").copied();
    let show_progress = !m.get_flag("no-progress");
    let threads = m
        .get_one::<usize>("threads")
        .copied()
        .unwrap_or_else(num_cpus::get)
        .max(1);

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .context("initialize rayon thread pool")?;

    info!("using {} rayon threads", rayon::current_num_threads());

    let single_list_source = resolve_single_list_source(
        query_path.as_ref(),
        query_list_path.as_ref(),
        reference_path.as_ref(),
        ref_list_path.as_ref(),
    )?;
    let triangle = explicit_triangle || single_list_source.is_some();

    anyhow::ensure!(
        !(matrix && single_list_source.is_some()),
        "--matrix requires both query and reference inputs; provide the same list with both --ql and --rl to compute and average reciprocal ANI values"
    );

    let (query_paths, ref_paths) = match single_list_source {
        Some(SingleListSource::Query) => {
            let paths = read_path_list(query_list_path.as_ref().unwrap())?;
            info!(
                "only --ql was provided; automatically computing {} upper-triangle genome pairs",
                paths.len().saturating_mul(paths.len().saturating_sub(1)) / 2
            );
            (paths.clone(), paths)
        }
        Some(SingleListSource::Reference) => {
            let paths = read_path_list(ref_list_path.as_ref().unwrap())?;
            info!(
                "only --rl was provided; automatically computing {} upper-triangle genome pairs",
                paths.len().saturating_mul(paths.len().saturating_sub(1)) / 2
            );
            (paths.clone(), paths)
        }
        None => (
            input_paths(query_path, query_list_path)?,
            input_paths(reference_path, ref_list_path)?,
        ),
    };

    if triangle {
        anyhow::ensure!(
            single_list_source.is_some() || (query_list_mode && ref_list_mode),
            "--triangle requires list input: provide one of --ql/--rl, or provide both with identical ordered lists"
        );
        anyhow::ensure!(
            query_paths.len() == ref_paths.len(),
            "--triangle requires query and reference lists with the same number of genomes"
        );
        anyhow::ensure!(
            query_paths == ref_paths,
            "--triangle requires identical query and reference lists in the same order"
        );
    }

    #[cfg(feature = "visual")]
    if visualize_path.is_some() && (query_paths.len() != 1 || ref_paths.len() != 1) {
        anyhow::bail!("--visualize only supports a single -q query and a single -r reference");
    }

    let config = AniConfig {
        kmer_size,
        fragment_len,
        min_identity,
        min_fraction,
        p_value,
        reference_size,
        window_size,
        ignore_top_percent,
        tab_hash_seed: tab_seed,
        tabulation_mode,
        distance_model,
        minimizer_mode: MinimizerMode::Simd,
        chain: chainx,
        diag_cluster_bin,
        diag_cluster_band,
        chain_max_predecessors,
        chain_max_gap,
        chain_diag_tolerance,
        show_progress,
    };

    info!(
        "using distance model {} ({})",
        config.distance_model.code(),
        config.distance_model.name()
    );
    info!(
        "using {} tabulation hashing",
        config.tabulation_mode.as_str()
    );

    let run = if let Some(split_count) = split_count {
        if triangle {
            compare_paths_upper_triangle_split_with_timing(
                &query_paths,
                &ref_paths,
                &config,
                split_count,
            )?
        } else {
            compare_paths_split_with_timing(&query_paths, &ref_paths, &config, split_count)?
        }
    } else if triangle {
        compare_paths_upper_triangle_with_timing(&query_paths, &ref_paths, &config)?
    } else {
        compare_paths_with_timing(&query_paths, &ref_paths, &config)?
    };

    if triangle {
        write_results_upper_triangle(output_path, &query_paths, &ref_paths, &run.results)?;
    } else {
        write_results(output_path, &run.results)?;
    }

    if matrix {
        write_phylip_matrix(output_path, &query_paths, &ref_paths, &run.results)?;
    }

    #[cfg(feature = "visual")]
    if let Some(path) = visualize_path {
        turboani::write_pair_visualization_pdf(&query_paths[0], &ref_paths[0], &config, path)?;
    }

    log_timing_summary(&run.timing);

    Ok(())
}

#[cfg(feature = "visual")]
fn add_visualize_arg(command: Command) -> Command {
    command.arg(
        Arg::new("visualize")
            .long("visualize")
            .visible_alias("visualization")
            .help("Write a single-pair visualization PDF")
            .value_name("PDF")
            .value_parser(clap::value_parser!(PathBuf)),
    )
}

#[cfg(not(feature = "visual"))]
fn add_visualize_arg(command: Command) -> Command {
    command
}

fn log_timing_summary(report: &TimingReport) {
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }

    for line in format_timing_summary(report).lines() {
        log::debug!("{line}");
    }
}

fn input_paths(single: Option<PathBuf>, list: Option<PathBuf>) -> Result<Vec<PathBuf>> {
    if let Some(path) = single {
        Ok(vec![path])
    } else if let Some(path) = list {
        read_path_list(path)
    } else {
        anyhow::bail!("provide either a single path or a list path")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SingleListSource {
    Query,
    Reference,
}

fn resolve_single_list_source(
    query: Option<&PathBuf>,
    query_list: Option<&PathBuf>,
    reference: Option<&PathBuf>,
    ref_list: Option<&PathBuf>,
) -> Result<Option<SingleListSource>> {
    let has_query_input = query.is_some() || query_list.is_some();
    let has_reference_input = reference.is_some() || ref_list.is_some();

    match (has_query_input, has_reference_input) {
        (true, true) => Ok(None),
        (false, false) => anyhow::bail!("provide at least one query or reference input"),
        (true, false) if query_list.is_some() => Ok(Some(SingleListSource::Query)),
        (false, true) if ref_list.is_some() => Ok(Some(SingleListSource::Reference)),
        (true, false) => anyhow::bail!(
            "a single -q/--query genome requires -r/--ref or --rl; use --ql for automatic all-versus-all triangle mode"
        ),
        (false, true) => anyhow::bail!(
            "a single -r/--ref genome requires -q/--query or --ql; use --rl for automatic all-versus-all triangle mode"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_list_alone_enables_single_list_mode() -> Result<()> {
        let list = PathBuf::from("genomes.txt");
        assert_eq!(
            resolve_single_list_source(None, Some(&list), None, None)?,
            Some(SingleListSource::Query)
        );
        Ok(())
    }

    #[test]
    fn reference_list_alone_enables_single_list_mode() -> Result<()> {
        let list = PathBuf::from("genomes.txt");
        assert_eq!(
            resolve_single_list_source(None, None, None, Some(&list))?,
            Some(SingleListSource::Reference)
        );
        Ok(())
    }

    #[test]
    fn two_sided_input_does_not_force_triangle_mode() -> Result<()> {
        let query_list = PathBuf::from("queries.txt");
        let ref_list = PathBuf::from("references.txt");
        assert_eq!(
            resolve_single_list_source(None, Some(&query_list), None, Some(&ref_list))?,
            None
        );
        Ok(())
    }

    #[test]
    fn one_single_genome_still_requires_the_other_side() {
        let query = PathBuf::from("query.fa");
        let error = resolve_single_list_source(Some(&query), None, None, None).unwrap_err();
        assert!(error.to_string().contains("requires -r/--ref or --rl"));
    }
}
