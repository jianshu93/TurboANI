use std::{env, path::PathBuf};

use anyhow::{Context, Result};
use clap::{Arg, ArgAction, ArgGroup, Command};
use log::info;
use turboani::{
    FastAniConfig, MinimizerMode, TimingReport, compare_paths_split_with_timing,
    compare_paths_with_timing, format_timing_summary, read_path_list, write_pair_visualization_pdf,
    write_phylip_matrix, write_results,
};

fn main() -> Result<()> {
    println!("\n ************** initializing logger *****************\n");
    env_logger::Builder::from_default_env().init();
    log::info!("logger initialized from default environment");

    let m = Command::new("bitani")
        .version(env!("CARGO_PKG_VERSION"))
        .about(
            "Super-fast ANI with SIMD minimizers, tabulation hashing, compact lookup indexing, \
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
                .help("Text file containing one query genome path per line")
                .value_name("QUERY_LIST")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .group(
            ArgGroup::new("query-input")
                .args(["query", "query-list"])
                .required(true)
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
                .help("Text file containing one reference genome path per line")
                .value_name("REFERENCE_LIST")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .group(
            ArgGroup::new("reference-input")
                .args(["reference", "ref-list"])
                .required(true)
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
            Arg::new("chaining")
                .long("chaining")
                .help(
                    "Use optimal ChainX semiglobal colinear chaining over minimizer anchors for \
                     L1 candidate screening; the default is diagonal clustering",
                )
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("matrix")
                .long("matrix")
                .help(
                    "Also write a FastANI-style lower-triangular matrix to <output>.matrix; \
                     reciprocal ANI values are averaged",
                )
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("visualize")
                .long("visualize")
                .help("Write a single-pair visualization PDF")
                .value_name("PDF")
                .value_parser(clap::value_parser!(PathBuf)),
        )
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
        .get_matches();

    let query_path = m.get_one::<PathBuf>("query").cloned();
    let query_list_path = m.get_one::<PathBuf>("query-list").cloned();
    let reference_path = m.get_one::<PathBuf>("reference").cloned();
    let ref_list_path = m.get_one::<PathBuf>("ref-list").cloned();
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
    let chaining = m.get_flag("chaining");
    let matrix = m.get_flag("matrix");
    let visualize_path = m.get_one::<PathBuf>("visualize");
    let split_count = m.get_one::<usize>("split").copied();
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

    let query_paths = input_paths(query_path, query_list_path)?;
    let ref_paths = input_paths(reference_path, ref_list_path)?;

    if visualize_path.is_some() && (query_paths.len() != 1 || ref_paths.len() != 1) {
        anyhow::bail!("--visualize only supports a single -q query and a single -r reference");
    }

    let config = FastAniConfig {
        kmer_size,
        fragment_len,
        min_identity,
        min_fraction,
        p_value,
        reference_size,
        window_size,
        ignore_top_percent,
        tab_hash_seed: tab_seed,
        minimizer_mode: MinimizerMode::Simd,
        chain: chaining,
    };

    let run = if let Some(split_count) = split_count {
        compare_paths_split_with_timing(&query_paths, &ref_paths, &config, split_count)?
    } else {
        compare_paths_with_timing(&query_paths, &ref_paths, &config)?
    };

    write_results(output_path, &run.results)?;

    if matrix {
        write_phylip_matrix(output_path, &query_paths, &ref_paths, &run.results)?;
    }

    if let Some(path) = visualize_path {
        write_pair_visualization_pdf(&query_paths[0], &ref_paths[0], &config, path)?;
    }

    log_timing_summary(&run.timing);

    Ok(())
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
