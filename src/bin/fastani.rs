use std::{env, path::PathBuf};

use anyhow::Result;
use clap::{Arg, ArgAction, ArgGroup, Command};
use turboani::{
    AniConfig, DistanceModel, MinimizerMode, TabulationMode, TimingReport,
    compare_paths_with_timing, format_timing_summary, read_path_list, write_phylip_matrix,
    write_results,
};

#[derive(Debug)]
struct Cli {
    query: Option<PathBuf>,
    query_list: Option<PathBuf>,
    reference: Option<PathBuf>,
    ref_list: Option<PathBuf>,
    output: PathBuf,
    kmer_size: usize,
    fragment_len: usize,
    min_identity: f64,
    min_fraction: f64,
    p_value: f64,
    reference_size: u64,
    window_size: Option<usize>,
    ignore_top_percent: f64,
    matrix: bool,
    #[cfg(feature = "visual")]
    visualize: Option<PathBuf>,
    threads: usize,
}

fn main() -> Result<()> {
    let cli = parse_cli();

    if cli.threads > 1 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .ok();
    }

    let query_paths = input_paths(cli.query, cli.query_list)?;
    let ref_paths = input_paths(cli.reference, cli.ref_list)?;
    #[cfg(feature = "visual")]
    if cli.visualize.is_some() && (query_paths.len() != 1 || ref_paths.len() != 1) {
        anyhow::bail!("--visualize only supports a single -q query and a single -r reference");
    }

    let config = AniConfig {
        kmer_size: cli.kmer_size,
        fragment_len: cli.fragment_len,
        min_identity: cli.min_identity,
        min_fraction: cli.min_fraction,
        p_value: cli.p_value,
        reference_size: cli.reference_size,
        window_size: cli.window_size,
        ignore_top_percent: cli.ignore_top_percent,
        tab_hash_seed: 42,
        tabulation_mode: TabulationMode::Twisted,
        distance_model: DistanceModel::Poisson,
        minimizer_mode: MinimizerMode::FastAni,
        chain: false,
        diag_cluster_bin: 1000,
        diag_cluster_band: 500,
        show_progress: false,
    };

    let run = compare_paths_with_timing(&query_paths, &ref_paths, &config)?;
    write_results(&cli.output, &run.results)?;
    if cli.matrix {
        write_phylip_matrix(&cli.output, &query_paths, &ref_paths, &run.results)?;
    }
    #[cfg(feature = "visual")]
    if let Some(path) = cli.visualize {
        turboani::write_pair_visualization_pdf(&query_paths[0], &ref_paths[0], &config, path)?;
    }
    log_timing_summary(&run.timing);
    Ok(())
}

fn parse_cli() -> Cli {
    let m = Command::new("rust-fastani-style")
        .version(env!("CARGO_PKG_VERSION"))
        .about("FastANI-compatible Rust ANI with Murmur3 minimizers and FastANI-style L2")
        .arg(
            Arg::new("query")
                .short('q')
                .long("query")
                .help("Single query genome FASTA/FASTQ path")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("query-list")
                .long("ql")
                .alias("queryList")
                .help("Text file containing one query genome path per line")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .group(
            ArgGroup::new("query-input")
                .args(["query", "query-list"])
                .required(true),
        )
        .arg(
            Arg::new("reference")
                .short('r')
                .long("ref")
                .help("Single reference genome FASTA/FASTQ path")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("ref-list")
                .long("rl")
                .alias("refList")
                .help("Text file containing one reference genome path per line")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .group(
            ArgGroup::new("reference-input")
                .args(["reference", "ref-list"])
                .required(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output ANI table")
                .required(true)
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("kmer-size")
                .short('k')
                .long("kmer")
                .help("K-mer size")
                .value_parser(clap::value_parser!(usize))
                .default_value("16"),
        )
        .arg(
            Arg::new("fragment-len")
                .long("fragLen")
                .help("Query fragment length")
                .value_parser(clap::value_parser!(usize))
                .default_value("3000"),
        )
        .arg(
            Arg::new("min-identity")
                .long("minIdentity")
                .help("Minimum identity percentage")
                .value_parser(clap::value_parser!(f64))
                .default_value("80"),
        )
        .arg(
            Arg::new("min-fraction")
                .long("minFraction")
                .help("Minimum aligned fraction")
                .value_parser(clap::value_parser!(f64))
                .default_value("0.2"),
        )
        .arg(
            Arg::new("p-value")
                .long("pValue")
                .help("P-value used to derive the minimizer window size")
                .value_parser(clap::value_parser!(f64))
                .default_value("0.001"),
        )
        .arg(
            Arg::new("reference-size")
                .long("referenceSize")
                .help("Reference size used to derive the minimizer window size")
                .value_parser(clap::value_parser!(u64))
                .default_value("5000000"),
        )
        .arg(
            Arg::new("window-size")
                .long("windowSize")
                .help("Override the p-value-derived minimizer window size")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("ignore-top-percent")
                .long("ignoreTopPercent")
                .help("Ignore the most frequent minimizer tokens during L1 lookup")
                .value_parser(clap::value_parser!(f64))
                .default_value("0"),
        )
        .arg(
            Arg::new("matrix")
                .long("matrix")
                .help("Also write a Phylip lower-triangular matrix to <output>.matrix; reciprocal ANI values are averaged")
                .action(ArgAction::SetTrue),
        )
        ;
    let m = add_visualize_arg(m)
        .arg(
            Arg::new("threads")
                .short('t')
                .long("threads")
                .help("Number of Rayon worker threads")
                .value_parser(clap::value_parser!(usize))
                .default_value("1"),
        )
        .get_matches();

    Cli {
        query: m.get_one::<PathBuf>("query").cloned(),
        query_list: m.get_one::<PathBuf>("query-list").cloned(),
        reference: m.get_one::<PathBuf>("reference").cloned(),
        ref_list: m.get_one::<PathBuf>("ref-list").cloned(),
        output: m.get_one::<PathBuf>("output").unwrap().clone(),
        kmer_size: *m.get_one::<usize>("kmer-size").unwrap(),
        fragment_len: *m.get_one::<usize>("fragment-len").unwrap(),
        min_identity: *m.get_one::<f64>("min-identity").unwrap(),
        min_fraction: *m.get_one::<f64>("min-fraction").unwrap(),
        p_value: *m.get_one::<f64>("p-value").unwrap(),
        reference_size: *m.get_one::<u64>("reference-size").unwrap(),
        window_size: m.get_one::<usize>("window-size").copied(),
        ignore_top_percent: *m.get_one::<f64>("ignore-top-percent").unwrap(),
        matrix: m.get_flag("matrix"),
        #[cfg(feature = "visual")]
        visualize: m.get_one::<PathBuf>("visualize").cloned(),
        threads: *m.get_one::<usize>("threads").unwrap(),
    }
}

#[cfg(feature = "visual")]
fn add_visualize_arg(command: Command) -> Command {
    command.arg(
        Arg::new("visualize")
            .long("visualize")
            .visible_alias("visualization")
            .help("Write a single-pair visualization PDF")
            .value_parser(clap::value_parser!(PathBuf)),
    )
}

#[cfg(not(feature = "visual"))]
fn add_visualize_arg(command: Command) -> Command {
    command
}

fn log_timing_summary(report: &TimingReport) {
    if !timing_debug_enabled() {
        return;
    }
    for line in format_timing_summary(report).lines() {
        eprintln!("{line}");
    }
}

fn timing_debug_enabled() -> bool {
    let Ok(value) = env::var("RUST_LOG") else {
        return false;
    };
    value.split(',').any(|part| {
        let part = part.trim().to_ascii_lowercase();
        part == "debug" || part == "trace" || part.ends_with("=debug") || part.ends_with("=trace")
    })
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
