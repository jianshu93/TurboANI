use std::cmp::Reverse;
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use needletail::{Sequence, parse_fastx_file};
use rammap::{Aligner, Mapping, Preset, Strand};

const CONTIG_GAP: usize = 1000;
const UNPLACED_COORD: u64 = 1_000_000_000_000_000_000;

#[derive(Debug, Clone)]
pub struct ContigOrderOutput {
    pub paf: PathBuf,
    pub placement_tsv: PathBuf,
    pub ordered_fasta: PathBuf,
    pub pseudochromosome_fasta: PathBuf,
}

#[derive(Debug, Clone)]
struct Contig {
    name: String,
    seq: Vec<u8>,
    map_seq: Vec<u8>,
}

#[derive(Debug, Clone)]
struct Placement {
    contig: String,
    length: usize,
    ref_name: String,
    ref_start: u64,
    ref_end: u64,
    strand: char,
    aligned_length: usize,
    matches: usize,
    mapq: i32,
    status: &'static str,
}

pub fn order_query_contigs(
    reference_path: &Path,
    query_path: &Path,
    prefix: &Path,
) -> Result<ContigOrderOutput> {
    let output = ContigOrderOutput {
        paf: prefixed_path(prefix, ".query_vs_ref.paf"),
        placement_tsv: prefixed_path(prefix, ".contig_order.tsv"),
        ordered_fasta: prefixed_path(prefix, ".reordered.fasta"),
        pseudochromosome_fasta: prefixed_path(prefix, ".pseudochromosome.fasta"),
    };

    let reference = read_reference_sequences(reference_path)?;
    let query = read_query_contigs(query_path)?;
    let aligner = Aligner::from_seqs(reference, Preset::Asm5);
    let mut best = HashMap::<String, ((usize, usize, i32), Placement)>::new();
    let mut paf = BufWriter::new(
        File::create(&output.paf)
            .with_context(|| format!("failed to create rammap PAF {}", output.paf.display()))?,
    );

    for contig in &query {
        let result = aligner.map_seq(&contig.name, &contig.map_seq);
        for mapping in result
            .mappings
            .iter()
            .filter(|mapping| mapping.is_primary || mapping.is_supplementary)
        {
            write_paf_record(&mut paf, &contig.name, contig.seq.len(), mapping)?;
            let placement = placement_from_mapping(contig, mapping);
            let score = (placement.aligned_length, placement.matches, placement.mapq);
            let replace = best
                .get(&contig.name)
                .map(|(old_score, _)| score > *old_score)
                .unwrap_or(true);
            if replace {
                best.insert(contig.name.clone(), (score, placement));
            }
        }
    }
    paf.flush()
        .with_context(|| format!("failed to flush rammap PAF {}", output.paf.display()))?;

    let ordered = ordered_placements(&query, best);
    write_placement_tsv(&output.placement_tsv, &ordered)?;
    write_ordered_fasta(&output.ordered_fasta, &query, &ordered)?;
    write_pseudochromosome_fasta(&output.pseudochromosome_fasta, &query, &ordered)?;

    Ok(output)
}

fn read_reference_sequences(path: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    let mut reader =
        parse_fastx_file(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut sequences = Vec::new();
    while let Some(record) = reader.next() {
        let record = record.with_context(|| format!("failed to parse {}", path.display()))?;
        let name = String::from_utf8_lossy(record.id()).into_owned();
        sequences.push((name, record.normalize(false).into_owned()));
    }
    Ok(sequences)
}

fn read_query_contigs(path: &Path) -> Result<Vec<Contig>> {
    let mut reader =
        parse_fastx_file(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut contigs = Vec::new();
    while let Some(record) = reader.next() {
        let record = record.with_context(|| format!("failed to parse {}", path.display()))?;
        contigs.push(Contig {
            name: String::from_utf8_lossy(record.id()).into_owned(),
            seq: record.seq().into_owned(),
            map_seq: record.normalize(false).into_owned(),
        });
    }
    Ok(contigs)
}

fn placement_from_mapping(contig: &Contig, mapping: &Mapping) -> Placement {
    Placement {
        contig: contig.name.clone(),
        length: contig.seq.len(),
        ref_name: mapping.target_name.to_string(),
        ref_start: mapping.target_start as u64,
        ref_end: mapping.target_end as u64,
        strand: match mapping.strand {
            Strand::Forward => '+',
            Strand::Reverse => '-',
        },
        aligned_length: mapping.block_len,
        matches: mapping.matches,
        mapq: mapping.mapq,
        status: "placed",
    }
}

fn ordered_placements(
    contigs: &[Contig],
    mut best: HashMap<String, ((usize, usize, i32), Placement)>,
) -> Vec<Placement> {
    let mut placed = Vec::new();
    let mut unplaced = Vec::new();
    for contig in contigs {
        if let Some((_, placement)) = best.remove(&contig.name) {
            placed.push(placement);
        } else {
            unplaced.push(Placement {
                contig: contig.name.clone(),
                length: contig.seq.len(),
                ref_name: ".".to_string(),
                ref_start: UNPLACED_COORD,
                ref_end: UNPLACED_COORD,
                strand: '.',
                aligned_length: 0,
                matches: 0,
                mapq: 0,
                status: "unplaced",
            });
        }
    }

    placed.sort_unstable_by(|a, b| {
        (
            &a.ref_name,
            a.ref_start,
            a.ref_end,
            Reverse(a.length),
            &a.contig,
        )
            .cmp(&(
                &b.ref_name,
                b.ref_start,
                b.ref_end,
                Reverse(b.length),
                &b.contig,
            ))
    });
    unplaced.sort_unstable_by(|a, b| {
        (Reverse(a.length), &a.contig).cmp(&(Reverse(b.length), &b.contig))
    });
    placed.into_iter().chain(unplaced).collect()
}

fn write_placement_tsv(path: &Path, ordered: &[Placement]) -> Result<()> {
    let mut out = BufWriter::new(
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    writeln!(
        out,
        "order\tcontig\tlength\tref_name\tref_start\tref_end\tstrand\taligned_length\tmatches\tmapq\tstatus"
    )?;
    for (idx, placement) in ordered.iter().enumerate() {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            idx + 1,
            placement.contig,
            placement.length,
            placement.ref_name,
            placement.ref_start,
            placement.ref_end,
            placement.strand,
            placement.aligned_length,
            placement.matches,
            placement.mapq,
            placement.status
        )?;
    }
    Ok(())
}

fn write_ordered_fasta(path: &Path, contigs: &[Contig], ordered: &[Placement]) -> Result<()> {
    let index = contig_index(contigs);
    let mut out = BufWriter::new(
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    for (idx, placement) in ordered.iter().enumerate() {
        let contig = &contigs[index[placement.contig.as_str()]];
        let seq = oriented_seq(&contig.seq, placement.strand);
        writeln!(
            out,
            ">{:04}_{} ref={}:{}-{} strand={} status={}",
            idx + 1,
            placement.contig,
            placement.ref_name,
            placement.ref_start,
            placement.ref_end,
            placement.strand,
            placement.status
        )?;
        write_wrapped(&mut out, &seq)?;
    }
    Ok(())
}

fn write_pseudochromosome_fasta(
    path: &Path,
    contigs: &[Contig],
    ordered: &[Placement],
) -> Result<()> {
    let index = contig_index(contigs);
    let mut pseudo = Vec::new();
    for (idx, placement) in ordered.iter().enumerate() {
        if idx > 0 {
            pseudo.extend(std::iter::repeat_n(b'N', CONTIG_GAP));
        }
        let contig = &contigs[index[placement.contig.as_str()]];
        pseudo.extend(oriented_seq(&contig.seq, placement.strand));
    }

    let mut out = BufWriter::new(
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    writeln!(out, ">pseudochromosome_ordered_from_query_contigs")?;
    write_wrapped(&mut out, &pseudo)?;
    Ok(())
}

fn write_paf_record<W: Write>(
    out: &mut W,
    query_name: &str,
    query_len: usize,
    mapping: &Mapping,
) -> Result<()> {
    let strand = match mapping.strand {
        Strand::Forward => '+',
        Strand::Reverse => '-',
    };
    write!(
        out,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        query_name,
        query_len,
        mapping.query_start,
        mapping.query_end,
        strand,
        mapping.target_name,
        mapping.target_len,
        mapping.target_start,
        mapping.target_end,
        mapping.matches,
        mapping.block_len,
        mapping.mapq
    )?;
    if let Some(cigar) = &mapping.cigar {
        write!(out, "\tcg:Z:{cigar}")?;
    }
    writeln!(out)?;
    Ok(())
}

fn contig_index(contigs: &[Contig]) -> HashMap<&str, usize> {
    contigs
        .iter()
        .enumerate()
        .map(|(idx, contig)| (contig.name.as_str(), idx))
        .collect()
}

fn oriented_seq(seq: &[u8], strand: char) -> Vec<u8> {
    if strand == '-' {
        seq.iter().rev().map(|base| complement(*base)).collect()
    } else {
        seq.to_vec()
    }
}

fn complement(base: u8) -> u8 {
    match base {
        b'A' => b'T',
        b'C' => b'G',
        b'G' => b'C',
        b'T' => b'A',
        b'a' => b't',
        b'c' => b'g',
        b'g' => b'c',
        b't' => b'a',
        b'R' => b'Y',
        b'Y' => b'R',
        b'M' => b'K',
        b'K' => b'M',
        b'B' => b'V',
        b'V' => b'B',
        b'D' => b'H',
        b'H' => b'D',
        b'N' => b'N',
        b'r' => b'y',
        b'y' => b'r',
        b'm' => b'k',
        b'k' => b'm',
        b'b' => b'v',
        b'v' => b'b',
        b'd' => b'h',
        b'h' => b'd',
        b'n' => b'n',
        _ => base,
    }
}

fn write_wrapped<W: Write>(out: &mut W, seq: &[u8]) -> Result<()> {
    for chunk in seq.chunks(80) {
        out.write_all(chunk)?;
        out.write_all(b"\n")?;
    }
    Ok(())
}

fn prefixed_path(prefix: &Path, suffix: &str) -> PathBuf {
    let mut path = OsString::from(prefix.as_os_str());
    path.push(suffix);
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_complement_matches_iupac_script_mapping() {
        assert_eq!(
            oriented_seq(b"ACGTRYMKBDHVNacgtrymkbdhvn", '-'),
            b"nbdhvmkryacgtNBDHVMKRYACGT"
        );
    }

    #[test]
    fn prefix_suffix_is_appended_not_replaced() {
        assert_eq!(
            prefixed_path(Path::new("sample.out"), ".contig_order.tsv"),
            PathBuf::from("sample.out.contig_order.tsv")
        );
    }

    #[test]
    fn rammap_ordering_places_orients_and_keeps_unplaced_contigs() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let ref_path = dir.path().join("ref.fa");
        let query_path = dir.path().join("query.fa");
        let prefix = dir.path().join("ordered");

        let reference = deterministic_dna(60_000, 17);
        std::fs::write(&ref_path, format!(">refA\n{}\n", wrap_string(&reference)))?;

        let left = &reference[5_000..11_000];
        let middle = &reference[22_000..28_500];
        let reversed = oriented_seq(reference[43_000..50_000].as_bytes(), '-');
        let unplaced = deterministic_dna(2_600, 991);
        std::fs::write(
            &query_path,
            format!(
                ">ctg_mid\n{}\n>ctg_left\n{}\n>ctg_rev\n{}\n>ctg_unplaced\n{}\n",
                wrap_string(middle),
                wrap_string(left),
                wrap_string(std::str::from_utf8(&reversed)?),
                wrap_string(&unplaced)
            ),
        )?;

        let output = order_query_contigs(&ref_path, &query_path, &prefix)?;
        let placement = std::fs::read_to_string(output.placement_tsv)?;
        let rows = placement.lines().skip(1).collect::<Vec<_>>();
        assert_eq!(rows.len(), 4);
        assert!(rows[0].contains("\tctg_left\t"));
        assert!(rows[0].contains("\t+\t"));
        assert!(rows[1].contains("\tctg_mid\t"));
        assert!(rows[2].contains("\tctg_rev\t"));
        assert!(rows[2].contains("\t-\t"));
        assert!(rows[3].contains("\tctg_unplaced\t"));
        assert!(rows[3].ends_with("\tunplaced"));

        let pseudo = std::fs::read_to_string(output.pseudochromosome_fasta)?;
        assert!(pseudo.starts_with(">pseudochromosome_ordered_from_query_contigs\n"));
        let pseudo_seq = pseudo.lines().skip(1).collect::<String>();
        assert!(pseudo_seq.starts_with(left));
        Ok(())
    }

    fn deterministic_dna(len: usize, mut state: u64) -> String {
        let mut seq = String::with_capacity(len);
        for _ in 0..len {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            seq.push(match (state >> 32) & 3 {
                0 => 'A',
                1 => 'C',
                2 => 'G',
                _ => 'T',
            });
        }
        seq
    }

    fn wrap_string(seq: &str) -> String {
        seq.as_bytes()
            .chunks(80)
            .map(|chunk| std::str::from_utf8(chunk).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
    }
}
