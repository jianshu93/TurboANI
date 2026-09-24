//! End-to-end checks that a persisted sketch matches a freshly built index.

use std::path::PathBuf;

use anyhow::Result;
use tempfile::tempdir;
use turboani::{AniConfig, compare_paths, compare_paths_with_sketch, write_reference_sketch};

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data").join(name)
}

fn genomes() -> (Vec<PathBuf>, Vec<PathBuf>) {
    (
        vec![data("Faecalibacterium_prausnitzii_LS2233.fna.gz")],
        vec![data("GCF_019967995.1.fna.gz")],
    )
}

#[test]
fn sketch_reproduces_in_memory_ani_exactly() -> Result<()> {
    let (queries, references) = genomes();
    let config = AniConfig::default();

    let direct = compare_paths(&queries, &references, &config)?;

    let dir = tempdir()?;
    let sketch = dir.path().join("ref.tsk");
    write_reference_sketch(&references, &config, &sketch, true)?;
    let from_sketch = compare_paths_with_sketch(&queries, &sketch, &config)?.results;

    assert_eq!(direct.len(), from_sketch.len(), "result count differs");
    for (a, b) in direct.iter().zip(from_sketch.iter()) {
        assert_eq!(a.query, b.query);
        assert_eq!(a.reference, b.reference);
        assert_eq!(
            a.ani.to_bits(),
            b.ani.to_bits(),
            "ANI differs: {} vs {}",
            a.ani,
            b.ani
        );
        assert_eq!(a.mapped_fragments, b.mapped_fragments);
        assert_eq!(a.total_query_fragments, b.total_query_fragments);
    }
    Ok(())
}

#[test]
fn sketch_rejects_mismatched_parameters() -> Result<()> {
    let (queries, references) = genomes();
    let dir = tempdir()?;
    let sketch = dir.path().join("ref.tsk");

    let built_with = AniConfig::default();
    write_reference_sketch(&references, &built_with, &sketch, true)?;

    let queried_with = AniConfig {
        kmer_size: 15,
        ..AniConfig::default()
    };
    let err = compare_paths_with_sketch(&queries, &sketch, &queried_with)
        .expect_err("mismatched k must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("--kmer"),
        "error should name the offending parameter, got: {msg}"
    );
    Ok(())
}

#[test]
fn sketch_rejects_foreign_files() -> Result<()> {
    let dir = tempdir()?;
    let bogus = dir.path().join("not-a-sketch.tsk");
    std::fs::write(&bogus, b"this is not a turboani sketch file at all")?;

    let (queries, _) = genomes();
    let err = compare_paths_with_sketch(&queries, &bogus, &AniConfig::default())
        .expect_err("non-sketch file must be rejected");
    assert!(format!("{err:#}").contains("not a TurboANI sketch"));
    Ok(())
}

#[test]
fn uncompressed_sketch_matches_compressed() -> Result<()> {
    let (queries, references) = genomes();
    let config = AniConfig::default();
    let dir = tempdir()?;

    let packed = dir.path().join("packed.tsk");
    let plain = dir.path().join("plain.tsk");
    let a = write_reference_sketch(&references, &config, &packed, true)?.0;
    let b = write_reference_sketch(&references, &config, &plain, false)?.0;
    assert!(
        a.bytes < b.bytes,
        "compressed sketch should be smaller: {} vs {}",
        a.bytes,
        b.bytes
    );

    let from_packed = compare_paths_with_sketch(&queries, &packed, &config)?.results;
    let from_plain = compare_paths_with_sketch(&queries, &plain, &config)?.results;
    assert_eq!(from_packed.len(), from_plain.len());
    for (x, y) in from_packed.iter().zip(from_plain.iter()) {
        assert_eq!(x.ani.to_bits(), y.ani.to_bits());
    }
    Ok(())
}
