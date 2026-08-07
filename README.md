
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/turboani/README.html)
![](https://anaconda.org/bioconda/turboani/badges/license.svg)
![](https://anaconda.org/bioconda/turboani/badges/version.svg)
![](https://anaconda.org/bioconda/turboani/badges/latest_release_relative_date.svg)
![](https://anaconda.org/bioconda/turboani/badges/platforms.svg)
[![install with conda](https://anaconda.org/bioconda/turboani/badges/downloads.svg)](https://anaconda.org/bioconda/turboani)

[![Crates.io](https://img.shields.io/crates/v/turboani.svg)](https://crates.io/crates/turboani)

<div align="center">
  <img width="100%" src ="TurboANI_logo.svg">
</div>

# TurboANI
TurboANI is a super fast ANI estimation algorithm implemented in pure Rust:

The main algorithm flow:

1. Build a reference minimizer lookup. Intial reference window identification with inverted index.
2. L1: Cluster query-fragment minimizer seed hits into candidate reference intervals (via diagnoal clustering), followed by minimap2 chaining or the much slower optimal ChainX (co-linear chaining).
3. L2: slide query-length windows over each L1 window interval and score them via a sliding bottom-s MinHash sketch algorithm.
4. Keep best-hit and reference-bin reciprocal filters before averaging ANI.



Key ideas:
- `simd-minimizers` for canonical minimizer positions and super-k-mer window coordinates. Note that the super-k-mer windows and corresponding minimizer positions were retained. 
- `tab-hash::Tab64Twisted` for deterministic 64-bit tabulation-hashed minimizer. This step is a rehash to obtain true pseudo-randomness via twisted/simple tabulation hashing. 
- Compact inverted minimizer index. It avoids HashMap's pointer-chasing and per-key vector overhead by flattening everything into two arrays with open-addressed slot lookup.
- A MUMmer-style diagonal clustering followed by minimap2-style fast chaining for L1 candidate window screening.
- A cache-friendly exact L2 bottom-s MinHash sketch slide mapper with local coordinate indices and a two-level bitset pivot.
- Reuses precomputed distance tables so that the post-L2 distance calculation becomes a direct table lookup instead of repeated confidence-bound and Mash-distance calculations.
- Efficient Rayon parallelism: work-stealing parallel iterators at multiple levels, including reference-genome indexing, query-genome mapping, and per-fragment L1/L2 mapping. The shared reference index is immutable, while fragment-level minimizers, candidate windows, L2 sliding state, mappings, and counters are worker-local, avoiding lock contention in the dominant mapping stages. This design gives fine-grained load balancing for highly uneven genome-pair workloads.



L2 scoring uses a new mashmap-like incremental bottom-sketch slide mapper: a query fragment's unique minimizer seed the bottom-k union, and each candidate reference super-window is updated as the window slides. The active implementation uses local coordinate compression plus a summary bitset to move the bottom-k pivot with word-level operations instead of a tree lookup on every insert/delete.

The final Mash distance and confidence-bound calculation is cached exactly by `(sketch_size, best_shared)`, so each L2 candidate performs a table lookup rather than recomputing the same binomial-bound math.

`simd-minimizers` uses ntHash internally with SIMD to choose minimizer positions. The turboani binary takes the returned canonical minimizer k-mer value and applies `Tab64Twisted` tabulation hashing once. That minimizer is reused for L1 lookup and exact L2 bottom-sketch comparison. The tabulation table is deterministic by default and controlled by `--tabSeed`.

The split mode allows query mapping to a small subset of references to reduce RAM. 

Alternative evolutionary models can be used, e.g., binomial model.

## Quick install and usage 
On Linux or MacOS (CPU) via bioconda
```bash
conda install -c bioconda -c conda-forge turboani
```
This is how you can run all-versua-all ANI for a list of genomes (gz supprted, one per line):
```bash
### obtain some testing genomes first
wget https://github.com/jianshu93/TurboANI/releases/download/v0.1.6/strep_30_sampled_genomes.tar.gz
tar -xzvf strep_30_sampled_genomes.tar.gz
cd strep_30_sampled_genomes
find . -name "*.fna.gz" > queries.txt
cp queries.txt references.txt
### run all-versus-all in list mode
turboani --ql queries.txt --rl references.txt -o turboani.tsv
```

## Pre-built binaries
```bash
## Linux (no visualization feature, see build section if you want it)
wget https://github.com/jianshu93/TurboANI/releases/download/v0.1.6/turboani_linux_x86-64_v0.1.6.gz
gunzip turboani_linux_x86-64_v0.1.6.gz
mv turboani_linux_x86-64_v0.1.6 turboani
chmod a+x ./turboani
./turboani -h

## MacOS (visualization feature)
wget https://github.com/jianshu93/TurboANI/releases/download/v0.1.6/turboani_darwin_aarch64_v0.1.6.tar.gz
tar -xzvf turboani_darwin_aarch64_v0.1.6.tar.gz
chmod a+x ./turboani
./turboani -h

## Homebrew install for MacOS
## install homebrew first: https://brew.sh
brew update
brew tap jianshu93/TurboANI
brew trust jianshu93/TurboANI
brew install TurboANI
turboani -h

```
## Install from crates.io via cargo
### Install cargo first
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

```bash
## On Linux
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo install turboani

## MacOS
RUSTFLAGS="-C target-cpu=native" cargo install turboani

```

## Build from source (Linux)
```bash
### Install rustup here: here: https://rustup.rs
### After intalling rustup
rustup install nightly
rustup default nightly
git clone https://github.com/jianshu93/TurboANI
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
./target/release/turboani -h
```

## Build from source (MacOS)
```bash
### Install rustup here: here: https://rustup.rs
### After intalling rustup
rustup install nightly
rustup default nightly
git clone https://github.com/jianshu93/TurboANI
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/turboani -h
```


## Detailed usage
### single pair comparsion
Single-pair visualization writes a PDF with a query/reference map and fragment identity panel:

```bash
./target/release/turboani \
  -q query.fa \
  -r reference.fa \
  -o pair.tsv \
  --visualize pair.pdf
```
`--visualize` intentionally supports only one `-q` and one `-r`; it rejects list mode. It is only available via the "--features visual" when compiling (font configuration libraries are requried):

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features visual

```

This it the plot.

<div align="center">
  <img width="100%" src ="Figures.jpg">
</div>


### Many comparsions
List mode is also supported (one file per line, .gz supported):

```bash
./target/release/turboani \
  --ql queries.txt \
  --rl references.txt \
  -o turboani.tsv \
```

Output columns match FastANI:

```text
query_path  reference_path  ani  mapped_fragments  total_query_fragments
```

## Separate fastANI binary
FastANI compatibility mode is a separate binary that implements the original FastANI algorithm:

```bash
./target/release/fastani -- \
  --ql queries.txt \
  --rl references.txt \
  -o fastani-style.tsv \
```


## Useful knobs:

- `--fragLen 3000`
- `-k 16`
- `--minIdentity 80`
- `--minFraction 0.2`
- `--tabSeed 42`
- `--windowSize N` to override p-value-derived minimizer window.
- `--ignoreTopPercent P` to ignore the most frequent minimizers during L1 lookup.
- `--visualize pair.pdf` to write a single-pair PDF map/identity plot.

Timing is collected internally and emitted only in debug mode. For a concise timing/counter summary, run with `RUST_LOG=debug`.

Rayon is used at three levels where the data are independent: reference genome indexing across reference files, query-file comparisons, and fragment mapping inside each query file.

## Notes

The implementation skips minimizer windows spanning ambiguous bases before passing sequence runs to `simd-minimizers`. This is usually preferable for Rust SIMD packing and keeps `N`-rich phage assemblies from creating artificial seeds.

## References

Paper to come
