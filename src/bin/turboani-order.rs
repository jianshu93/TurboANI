use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "turboani-order",
    version,
    about = "Order and orient fragmented query contigs against a complete reference with pure-Rust rammap asm5 mapping"
)]
struct Cli {
    /// Complete reference genome FASTA/FASTQ path.
    reference: PathBuf,
    /// Fragmented query genome FASTA/FASTQ path.
    query: PathBuf,
    /// Output prefix.
    #[arg(default_value = "turboani_reordered")]
    prefix: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let output = turboani::order_query_contigs(&cli.reference, &cli.query, &cli.prefix)?;

    eprintln!("Wrote {}", output.paf.display());
    eprintln!("Wrote {}", output.placement_tsv.display());
    eprintln!("Wrote {}", output.ordered_fasta.display());
    eprintln!("Wrote {}", output.pseudochromosome_fasta.display());
    eprintln!();
    eprintln!("Suggested TurboANI visualization command:");
    eprintln!(
        "  turboani -r \"{}\" -q \"{}\" -o \"{}.ani.txt\" --visualize \"{}.pdf\"",
        cli.reference.display(),
        output.ordered_fasta.display(),
        cli.prefix.display(),
        cli.prefix.display()
    );
    Ok(())
}
