//! On-disk reference sketches: 2-bit k-mers plus positions, hash-sorted.
//! Hashes and the lookup structures are rebuilt on load.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use rayon::slice::ParallelSliceMut;

use crate::simd_minimizer::{
    Minimizer, MinimizerMode, TabulationMode, deterministic_tabulation_hasher, minimizer_token,
};
use crate::utils::{AniConfig, ContigInfo, GenomeInfo, ReferenceIndex, ReferenceTiming};
use rayon::prelude::*;

const MAGIC: &[u8; 8] = b"TANISKT1";
const FORMAT_VERSION: u32 = 5;

const COMPRESSION_NONE: u8 = 0;
const COMPRESSION_ZSTD: u8 = 1;

const ZSTD_LEVEL: i32 = 3;

#[derive(Debug, Clone, Copy)]
pub struct SketchStats {
    pub genomes: usize,
    pub contigs: usize,
    pub minimizers: usize,
    pub bytes: u64,
    pub uncompressed_bytes: u64,
}

/// Parameters a sketch must match to be reusable. `ignore_top_percent` is
/// excluded on purpose so it stays a query-time knob.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Fingerprint {
    kmer_size: u32,
    window_size: u32,
    tab_hash_seed: u64,
    tabulation_mode: u8,
    minimizer_mode: u8,
    distance_model: u8,
    fragment_len: u32,
    min_identity: f64,
    p_value: f64,
    reference_size: u64,
}

impl Fingerprint {
    fn from_config(config: &AniConfig, window_size: usize) -> Result<Self> {
        Ok(Self {
            kmer_size: u32::try_from(config.kmer_size).context("kmer size too large")?,
            window_size: u32::try_from(window_size).context("window size too large")?,
            tab_hash_seed: config.tab_hash_seed,
            tabulation_mode: match config.tabulation_mode {
                TabulationMode::Twisted => 0,
                TabulationMode::Simple => 1,
            },
            minimizer_mode: match config.minimizer_mode {
                MinimizerMode::Simd => 0,
                MinimizerMode::Scalar => 1,
                MinimizerMode::ScalarMinmer => 2,
            },
            distance_model: config.distance_model.code(),
            fragment_len: u32::try_from(config.fragment_len)
                .context("fragment length too large")?,
            min_identity: config.min_identity,
            p_value: config.p_value,
            reference_size: config.reference_size,
        })
    }

    fn explain_mismatch(&self, other: &Self) -> Option<String> {
        macro_rules! check {
            ($field:ident, $label:literal) => {
                if self.$field != other.$field {
                    return Some(format!(
                        "{} differs: sketch was built with {}, current run uses {}",
                        $label, self.$field, other.$field
                    ));
                }
            };
        }
        check!(kmer_size, "--kmer");
        check!(window_size, "minimizer window size");
        check!(tab_hash_seed, "--tabSeed");
        check!(tabulation_mode, "tabulation mode (--simpleTabulation)");
        check!(minimizer_mode, "minimizer mode");
        check!(distance_model, "--model");
        check!(fragment_len, "--fragLen");
        check!(min_identity, "--minIdentity");
        check!(p_value, "--pValue");
        check!(reference_size, "--referenceSize");
        None
    }
}

struct Writer<W: Write> {
    inner: W,
    written: u64,
}

impl<W: Write> Writer<W> {
    fn new(inner: W) -> Self {
        Self { inner, written: 0 }
    }

    fn bytes(&mut self, buf: &[u8]) -> Result<()> {
        self.inner.write_all(buf)?;
        self.written += buf.len() as u64;
        Ok(())
    }

    fn u8(&mut self, v: u8) -> Result<()> {
        self.bytes(&[v])
    }

    fn u32(&mut self, v: u32) -> Result<()> {
        self.bytes(&v.to_le_bytes())
    }

    fn u64(&mut self, v: u64) -> Result<()> {
        self.bytes(&v.to_le_bytes())
    }

    fn f64(&mut self, v: f64) -> Result<()> {
        self.bytes(&v.to_bits().to_le_bytes())
    }

    fn usize(&mut self, v: usize, label: &str) -> Result<()> {
        self.u64(u64::try_from(v).with_context(|| format!("{label} too large"))?)
    }

    fn string(&mut self, s: &str) -> Result<()> {
        self.usize(s.len(), "string length")?;
        self.bytes(s.as_bytes())
    }
}

enum BodyWriter<W: Write> {
    Plain(W),
    Zstd(Box<zstd::Encoder<'static, W>>),
}

impl BodyWriter<BufWriter<File>> {
    fn finish(self) -> Result<()> {
        let sink = match self {
            BodyWriter::Plain(w) => w,
            BodyWriter::Zstd(e) => e.finish().context("finish zstd stream")?,
        };
        let file = sink.into_inner().context("flush sketch file")?;
        file.sync_all().context("sync sketch file")?;
        Ok(())
    }
}

impl<W: Write> Write for BodyWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            BodyWriter::Plain(w) => w.write(buf),
            BodyWriter::Zstd(w) => w.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            BodyWriter::Plain(w) => w.flush(),
            BodyWriter::Zstd(w) => w.flush(),
        }
    }
}

enum Body<R: Read> {
    Plain(R),
    Zstd(Box<zstd::Decoder<'static, R>>),
}

impl<R: BufRead> Read for Body<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Body::Plain(r) => r.read(buf),
            Body::Zstd(r) => r.read(buf),
        }
    }
}

struct Reader<R: Read> {
    inner: R,
}

impl<R: Read> Reader<R> {
    fn new(inner: R) -> Self {
        Self { inner }
    }

    fn exact(&mut self, buf: &mut [u8]) -> Result<()> {
        self.inner
            .read_exact(buf)
            .context("sketch file ended early (truncated or corrupt)")
    }

    fn u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.exact(&mut b)?;
        Ok(b[0])
    }

    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn usize(&mut self, label: &str) -> Result<usize> {
        usize::try_from(self.u64()?).with_context(|| format!("{label} too large for this platform"))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.usize("string length")?;
        let mut buf = vec![0u8; len];
        self.exact(&mut buf)?;
        String::from_utf8(buf).context("sketch contains invalid UTF-8 name")
    }
}

fn write_fingerprint<W: Write>(w: &mut Writer<W>, fp: &Fingerprint) -> Result<()> {
    w.u32(fp.kmer_size)?;
    w.u32(fp.window_size)?;
    w.u64(fp.tab_hash_seed)?;
    w.u8(fp.tabulation_mode)?;
    w.u8(fp.minimizer_mode)?;
    w.u8(fp.distance_model)?;
    w.u8(0)?; // padding, keeps the following u32 on a 4-byte boundary
    w.u32(fp.fragment_len)?;
    w.f64(fp.min_identity)?;
    w.f64(fp.p_value)?;
    w.u64(fp.reference_size)
}

fn read_fingerprint<R: Read>(r: &mut Reader<R>) -> Result<Fingerprint> {
    let kmer_size = r.u32()?;
    let window_size = r.u32()?;
    let tab_hash_seed = r.u64()?;
    let tabulation_mode = r.u8()?;
    let minimizer_mode = r.u8()?;
    let distance_model = r.u8()?;
    let _pad = r.u8()?;
    let fragment_len = r.u32()?;
    let min_identity = r.f64()?;
    let p_value = r.f64()?;
    let reference_size = r.u64()?;
    Ok(Fingerprint {
        kmer_size,
        window_size,
        tab_hash_seed,
        tabulation_mode,
        minimizer_mode,
        distance_model,
        fragment_len,
        min_identity,
        p_value,
        reference_size,
    })
}

fn read_u64_column<R: Read>(r: &mut Reader<R>, out: &mut [u64]) -> Result<()> {
    const CHUNK: usize = 1 << 16;
    let mut buf = vec![0u8; CHUNK * 8];
    for block in out.chunks_mut(CHUNK) {
        let bytes = &mut buf[..block.len() * 8];
        r.exact(bytes)?;
        for (slot, raw) in block.iter_mut().zip(bytes.chunks_exact(8)) {
            *slot = u64::from_le_bytes(raw.try_into().unwrap());
        }
    }
    Ok(())
}

fn read_packed_column<R: Read>(r: &mut Reader<R>, out: &mut [u64], width: usize) -> Result<()> {
    const CHUNK: usize = 1 << 16;
    let mut buf = vec![0u8; CHUNK * width];
    for block in out.chunks_mut(CHUNK) {
        let bytes = &mut buf[..block.len() * width];
        r.exact(bytes)?;
        for (slot, raw) in block.iter_mut().zip(bytes.chunks_exact(width)) {
            let mut v = [0u8; 8];
            v[..width].copy_from_slice(raw);
            *slot = u64::from_le_bytes(v);
        }
    }
    Ok(())
}

fn read_u32_column<R: Read>(r: &mut Reader<R>, out: &mut [u32]) -> Result<()> {
    const CHUNK: usize = 1 << 16;
    let mut buf = vec![0u8; CHUNK * 4];
    for block in out.chunks_mut(CHUNK) {
        let bytes = &mut buf[..block.len() * 4];
        r.exact(bytes)?;
        for (slot, raw) in block.iter_mut().zip(bytes.chunks_exact(4)) {
            *slot = u32::from_le_bytes(raw.try_into().unwrap());
        }
    }
    Ok(())
}

fn read_u24_column<R: Read>(r: &mut Reader<R>, out: &mut [u32]) -> Result<()> {
    const CHUNK: usize = 1 << 16;
    let mut buf = vec![0u8; CHUNK * 3];
    for block in out.chunks_mut(CHUNK) {
        let bytes = &mut buf[..block.len() * 3];
        r.exact(bytes)?;
        for (slot, raw) in block.iter_mut().zip(bytes.chunks_exact(3)) {
            *slot = u32::from_le_bytes([raw[0], raw[1], raw[2], 0]);
        }
    }
    Ok(())
}

/// Serialize a built reference index to `path`.
pub(crate) fn write_sketch(
    path: &Path,
    index: &ReferenceIndex,
    config: &AniConfig,
    window_size: usize,
    compress: bool,
) -> Result<SketchStats> {
    ensure!(
        config.minimizer_mode == MinimizerMode::Simd,
        "sketching requires the SIMD minimizer mode"
    );

    // Build beside the target and rename on success, so a failure part-way
    // through cannot destroy an existing sketch.
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    match write_sketch_inner(&tmp, index, config, window_size, compress) {
        Ok(stats) => {
            std::fs::rename(&tmp, path)
                .with_context(|| format!("rename {} to {}", tmp.display(), path.display()))?;
            Ok(stats)
        }
        Err(err) => {
            let _ = std::fs::remove_file(&tmp);
            Err(err)
        }
    }
}

fn write_sketch_inner(
    path: &Path,
    index: &ReferenceIndex,
    config: &AniConfig,
    window_size: usize,
    compress: bool,
) -> Result<SketchStats> {
    let file =
        File::create(path).with_context(|| format!("create sketch file {}", path.display()))?;

    // Plaintext so a mismatch is rejected without decompressing the body.
    let mut header = Writer::new(BufWriter::new(file));
    header.bytes(MAGIC)?;
    header.u32(FORMAT_VERSION)?;
    header.string(env!("CARGO_PKG_VERSION"))?;
    write_fingerprint(&mut header, &Fingerprint::from_config(config, window_size)?)?;
    header.usize(index.genomes.len(), "genome count")?;
    header.usize(index.contigs.len(), "contig count")?;
    header.usize(index.minimizers.len(), "minimizer count")?;
    header.u8(if compress {
        COMPRESSION_ZSTD
    } else {
        COMPRESSION_NONE
    })?;
    let sink = header.inner;

    let mut w = Writer::new(if compress {
        let mut encoder = zstd::Encoder::new(sink, ZSTD_LEVEL).context("init zstd encoder")?;
        encoder
            .multithread(num_cpus::get().min(8) as u32)
            .context("configure zstd threads")?;
        BodyWriter::Zstd(Box::new(encoder))
    } else {
        BodyWriter::Plain(sink)
    });

    for genome in &index.genomes {
        w.usize(genome.length, "genome length")?;
        w.string(&genome.path.to_string_lossy())?;
    }

    for contig in &index.contigs {
        w.usize(contig.len, "contig length")?;
        w.usize(contig.genome_id, "genome id")?;
        w.string(&contig.name)?;
    }

    // Hash order is load-bearing: it is what makes the columns compress.
    let mut sorted = index.minimizers.clone();
    sorted.par_sort_unstable_by_key(|m| m.hash);

    let kmer_bytes = (2 * config.kmer_size).div_ceil(8);
    for minimizer in &sorted {
        ensure!(
            kmer_bytes == 8 || minimizer.kmer < (1u64 << (8 * kmer_bytes)),
            "k-mer value {} does not fit in {kmer_bytes} bytes; expected 2-bit packing",
            minimizer.kmer
        );
        w.bytes(&minimizer.kmer.to_le_bytes()[..kmer_bytes])?;
    }
    for minimizer in &sorted {
        w.u32(u32::try_from(minimizer.seq_id).context("contig id too large")?)?;
    }
    // Contig-local, so 24 bits is enough for any microbial contig.
    for minimizer in &sorted {
        let wpos = u32::try_from(minimizer.wpos).context("minimizer position too large")?;
        ensure!(
            wpos < (1 << 24),
            "contig position {wpos} exceeds the 24-bit sketch position limit"
        );
        w.bytes(&wpos.to_le_bytes()[..3])?;
    }

    let body_bytes = w.written;
    w.inner.finish()?;

    let on_disk = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    Ok(SketchStats {
        genomes: index.genomes.len(),
        contigs: index.contigs.len(),
        minimizers: index.minimizers.len(),
        bytes: on_disk,
        uncompressed_bytes: body_bytes,
    })
}

/// Load a sketch, rejecting one built with incompatible parameters.
pub(crate) fn read_sketch(
    path: &Path,
    config: &AniConfig,
    window_size: usize,
) -> Result<(ReferenceIndex, SketchStats, ReferenceTiming)> {
    let load_start = std::time::Instant::now();
    let file = File::open(path).with_context(|| format!("open sketch file {}", path.display()))?;
    let bytes = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut r = Reader::new(BufReader::new(file));

    let mut magic = [0u8; 8];
    r.exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("{} is not a TurboANI sketch file", path.display());
    }

    let version = r.u32()?;
    ensure!(
        version == FORMAT_VERSION,
        "sketch format version {version} is not supported (this build reads version {FORMAT_VERSION}); rebuild the sketch"
    );
    let built_by = r.string()?;

    let stored = read_fingerprint(&mut r)?;
    let current = Fingerprint::from_config(config, window_size)?;
    if let Some(reason) = stored.explain_mismatch(&current) {
        bail!(
            "sketch {} was built with incompatible parameters: {reason}. \
             Rebuild it with --sketch-out using the current settings.",
            path.display()
        );
    }
    log::debug!(
        "loaded sketch {} written by turboani {built_by}",
        path.display()
    );

    let genome_count = r.usize("genome count")?;
    let contig_count = r.usize("contig count")?;
    let minimizer_count = r.usize("minimizer count")?;
    let compression = r.u8()?;

    let body = match compression {
        COMPRESSION_NONE => Body::Plain(r.inner),
        COMPRESSION_ZSTD => Body::Zstd(Box::new(
            zstd::Decoder::with_buffer(r.inner).context("init zstd decoder")?,
        )),
        other => bail!(
            "sketch {} uses unknown compression code {other}",
            path.display()
        ),
    };
    let mut r = Reader::new(BufReader::with_capacity(1 << 20, body));

    let mut genomes = Vec::with_capacity(genome_count);
    for _ in 0..genome_count {
        let length = r.usize("genome length")?;
        let path = PathBuf::from(r.string()?);
        genomes.push(GenomeInfo { path, length });
    }

    let mut contigs = Vec::with_capacity(contig_count);
    for _ in 0..contig_count {
        let len = r.usize("contig length")?;
        let genome_id = r.usize("genome id")?;
        ensure!(
            genome_id < genome_count,
            "sketch references genome id {genome_id} but only {genome_count} genomes are present"
        );
        let name = r.string()?;
        contigs.push(ContigInfo {
            name,
            len,
            genome_id,
        });
    }

    let kmer_bytes = (2 * config.kmer_size).div_ceil(8);
    let mut kmers = vec![0u64; minimizer_count];
    read_packed_column(&mut r, &mut kmers, kmer_bytes)?;
    let mut seq_ids = vec![0u32; minimizer_count];
    read_u32_column(&mut r, &mut seq_ids)?;
    let mut positions = vec![0u32; minimizer_count];
    read_u24_column(&mut r, &mut positions)?;

    let tab = deterministic_tabulation_hasher(config.tab_hash_seed, config.tabulation_mode);
    let hashes: Vec<u64> = kmers
        .par_iter()
        .map(|&kmer| minimizer_token(kmer, config.kmer_size, &tab))
        .collect();

    let mut minimizers = Vec::with_capacity(minimizer_count);
    for i in 0..minimizer_count {
        let seq_id = seq_ids[i] as usize;
        ensure!(
            seq_id < contig_count,
            "sketch references contig id {seq_id} but only {contig_count} contigs are present"
        );
        minimizers.push(Minimizer {
            hash: hashes[i],
            kmer: kmers[i],
            seq_id,
            wpos: positions[i] as usize,
        });
    }
    drop(hashes);
    drop(kmers);
    drop(seq_ids);
    drop(positions);

    let read_wall_ns = load_start.elapsed().as_nanos();
    let (index, sort_wall_ns, lookup_wall_ns) =
        ReferenceIndex::finalize(genomes, contigs, minimizers, Vec::new(), config)?;

    let timing = ReferenceTiming {
        total_wall_ns: load_start.elapsed().as_nanos(),
        read_wall_ns,
        assemble_wall_ns: 0,
        sort_wall_ns,
        lookup_wall_ns,
        genomes: genome_count,
        contigs: contig_count,
        minimizers: minimizer_count,
        lookup_keys: index.lookup_len(),
        freq_threshold: index.freq_threshold,
    };

    Ok((
        index,
        SketchStats {
            genomes: genome_count,
            contigs: contig_count,
            minimizers: minimizer_count,
            bytes,
            uncompressed_bytes: 0,
        },
        timing,
    ))
}
