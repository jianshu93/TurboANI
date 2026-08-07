use anyhow::Result;
use simd_minimizers::packed_seq::{PackedSeqVec, SeqVec};
use tab_hash::{Tab64Simple, Tab64Twisted};

use crate::{AniConfig, HashValue, Offset, SeqId};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabulationMode {
    Twisted,
    Simple,
}

impl TabulationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Twisted => "twisted",
            Self::Simple => "simple",
        }
    }
}

pub(crate) enum TabulationHasher {
    Twisted(Tab64Twisted),
    Simple(Tab64Simple),
}

impl TabulationHasher {
    fn hash(&self, key: u64) -> u64 {
        match self {
            Self::Twisted(hasher) => hasher.hash(key),
            Self::Simple(hasher) => hasher.hash(key),
        }
    }
}

pub(crate) fn simd_compatible_window_size(k: usize, w: usize) -> usize {
    if (k + w - 1) % 2 == 1 {
        w
    } else if w > 1 {
        w - 1
    } else {
        w + 1
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Minimizer {
    pub(crate) hash: HashValue,
    pub(crate) seq_id: SeqId,
    pub(crate) wpos: Offset,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QuerySeed {
    pub(crate) hash: HashValue,
    pub(crate) qpos: Offset,
}

pub(crate) fn sequence_minimizers(
    seq: &[u8],
    config: &AniConfig,
    w: usize,
    seq_id: SeqId,
    tab_hasher: &TabulationHasher,
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
    tab_hasher: &TabulationHasher,
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

fn minimizer_token(canonical_kmer_value: u64, k: usize, tab_hasher: &TabulationHasher) -> u64 {
    let key = canonical_kmer_value ^ ((k as u64) << 56) ^ 0xD1B5_4A32_D192_ED03;
    tab_hasher.hash(key)
}

pub(crate) fn deterministic_tabulation_hasher(seed: u64, mode: TabulationMode) -> TabulationHasher {
    match mode {
        TabulationMode::Twisted => TabulationHasher::Twisted(deterministic_tab64_twisted(seed)),
        TabulationMode::Simple => TabulationHasher::Simple(deterministic_tab64_simple(seed)),
    }
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

fn deterministic_tab64_simple(seed: u64) -> Tab64Simple {
    let mut state = seed ^ 0xA076_1D64_78BD_642F;
    let mut table = [[0u64; 256]; 8];
    for row in &mut table {
        for value in row {
            *value = splitmix64_next(&mut state);
        }
    }
    Tab64Simple::with_table(table)
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    splitmix64_permute(*state)
}

#[cfg(test)]
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    splitmix64_permute(x)
}

pub(crate) fn splitmix64_permute(x: u64) -> u64 {
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
