//! Reading COCO JSON into [`Dataset`] and [`Annotation`] records.
//!
//! One rule shapes this module: **peak memory is the output, not the input
//! or a parse tree.** A tape-building parser turns every number and bracket
//! into a node before serde sees any of it, and on COCO files — which are
//! almost entirely numbers — that tape dwarfs the records it produces.
//! serde_json streams straight into the structs. And the file is read a
//! block at a time, so a results file that outweighs its own records never
//! sits in memory beside them.
//!
//! The annotations array is the bulk of any COCO file, so it is parsed in
//! parallel, in place: a dataset object is walked by hand until an array of
//! records, and each block of it is cut into runs at guessed record
//! boundaries (a `{` that follows `},`), parsed with rayon while the next
//! block is read, and moved into the output as the block finishes; the record
//! a block ends inside waits for the next. A guess can land inside a record —
//! a caption containing `},{`, or a nested list of objects — so every run
//! must end exactly where the next begins, and the run that reaches the
//! array's `]` says where the walk resumes. A block whose guesses fail is
//! parsed serially instead, record by record up to the one it cuts off, and
//! the guessing stops for the rest of that array. When the walk finds a
//! shape it does not expect, or bytes it cannot parse, one serial serde pass
//! over the whole file takes over — a second read — and is also what reports
//! the error for a malformed file.
//!
//! The output vector is reserved from the first block's record density and
//! grown only if that falls short, then shrunk to what it holds; each block's
//! runs are moved into it, so a block's records exist twice only for that
//! moment.
//!
//! Floats are read with serde_json's `float_roundtrip` feature. Its default
//! parser is best-effort and can land one ULP off, and an `area` one ULP from
//! 32² changes an annotation's size bucket.

use std::borrow::Cow;
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;

use rayon::prelude::*;
use serde::de::{DeserializeOwned, IgnoredAny};

use crate::types::{Annotation, Dataset};

/// Read and parse a dataset file. The second value is the number of
/// non-finite float tokens rewritten to `null` on the way in.
pub(crate) fn read_dataset(path: &Path) -> crate::error::Result<(Dataset, usize)> {
    read(path, stream_dataset, dataset_whole)
}

/// Read and parse a results file: a bare array of annotations, or a dataset
/// object whose `annotations` are the results.
pub(crate) fn read_results(path: &Path) -> crate::error::Result<(Vec<Annotation>, usize)> {
    read(path, stream_results, results_whole)
}

/// Stream the file in blocks, and only when that declines read it whole: a
/// clean file of the expected shape, which is nearly every file, is never in
/// memory at once, and a dirty one is sanitized only after a parse fails.
fn read<T>(
    path: &Path,
    stream: fn(&mut Source) -> io::Result<Option<T>>,
    whole: fn(&[u8]) -> serde_json::Result<T>,
) -> crate::error::Result<(T, usize)> {
    if let Some(value) = stream(&mut Source::open(path, BLOCK_BYTES)?)? {
        return Ok((value, 0));
    }
    // A shape the walk does not expect, a non-finite token, or a malformed
    // file: the whole file decides which, and the serial pass reports the
    // error. A sanitized file streams again from memory, so its peak is its
    // bytes plus its records.
    let raw = std::fs::read(path)?;
    let (fixed, n_fixed) = sanitize_non_finite(&raw);
    if n_fixed == 0 {
        return Ok((whole(&raw)?, 0));
    }
    let fixed = fixed.into_owned();
    drop(raw);
    match stream(&mut Source::new(
        Cursor::new(&fixed[..]),
        fixed.len() as u64,
        BLOCK_BYTES,
    ))? {
        Some(value) => Ok((value, n_fixed)),
        None => Ok((whole(&fixed)?, n_fixed)),
    }
}

/// How much of the input is read at a time. Two blocks are in memory once
/// an array of records is reached — the one being parsed and the one being
/// read — and before that, the buffer holds whatever the walk is on, so a
/// file whose `images` are larger than a block holds them whole for the
/// rest of the read. Larger blocks mean fewer parallel parses to
/// synchronize per file.
const BLOCK_BYTES: usize = 4 << 20;

/// The input's bytes, a window at a time.
///
/// `buf` holds the bytes from the first not yet parsed to the last read.
/// While an array's block is parsed, the next block is read into `spare`
/// behind the record the block ends inside, and the two are swapped; the
/// walk between arrays appends to `buf` as it needs more.
struct Source<'a> {
    reader: Box<dyn Read + Send + 'a>,
    /// Every byte has been read.
    done: bool,
    buf: Vec<u8>,
    spare: Vec<u8>,
    /// The input offset of `buf[0]`, and the input's length as opened: the
    /// size estimate for a record vector projects over what lies beyond the
    /// parse. A pipe has no length, and an input that grows or shrinks
    /// underneath the read only makes the estimate worse.
    base: u64,
    len: u64,
    block: usize,
}

impl<'a> Source<'a> {
    fn open(path: &Path, block: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Source::new(file, len, block))
    }

    fn new(reader: impl Read + Send + 'a, len: u64, block: usize) -> Self {
        Source {
            reader: Box::new(reader),
            done: false,
            buf: Vec::new(),
            spare: Vec::new(),
            base: 0,
            len,
            block,
        }
    }

    /// Append up to `n` more bytes of the input to `buf`; fewer only at its end.
    fn fill(&mut self, n: usize) -> io::Result<()> {
        read_block(&mut self.reader, &mut self.done, &mut self.buf, n)
    }

    /// Bytes of the input beyond `buf[..upto]`.
    fn beyond(&self, upto: usize) -> u64 {
        self.len.saturating_sub(self.base + upto as u64)
    }

    /// Drop `buf[..upto]`; what follows moves to the front.
    fn discard(&mut self, upto: usize) {
        self.buf.drain(..upto);
        self.base += upto as u64;
    }

    /// Run `parse` over `buf` while the next block is read into `spare`
    /// behind a copy of `buf[stop..]`, the record the block ends inside.
    fn parse_while_reading<U: Send>(
        &mut self,
        stop: usize,
        parse: impl FnOnce(&[u8]) -> U + Send,
    ) -> (U, io::Result<()>) {
        let Source {
            reader,
            done,
            buf,
            spare,
            block,
            ..
        } = self;
        let buf: &Vec<u8> = buf;
        rayon::join(
            || parse(buf),
            || {
                spare.clear();
                spare.extend_from_slice(&buf[stop..]);
                read_block(reader, done, spare, *block)
            },
        )
    }

    /// After [`parse_while_reading`](Self::parse_while_reading): make the
    /// block read into `spare` the buffer, `buf[stop..]` at its front.
    fn swap(&mut self, stop: usize) {
        std::mem::swap(&mut self.buf, &mut self.spare);
        self.base += stop as u64;
    }

    /// After [`parse_while_reading`](Self::parse_while_reading): keep `buf`
    /// and append the block read into `spare`, dropping the copy of
    /// `buf[stop..]` at its front.
    fn stitch(&mut self, stop: usize) {
        self.buf.truncate(stop);
        self.buf.reserve_exact(self.spare.len());
        self.buf.append(&mut self.spare);
    }

    /// The byte at `pos`, reading on when `buf` ends before it.
    fn byte(&mut self, pos: usize) -> io::Result<Option<u8>> {
        while pos >= self.buf.len() && !self.done {
            self.fill(self.block)?;
        }
        Ok(self.buf.get(pos).copied())
    }

    fn skip_ws(&mut self, mut pos: usize) -> io::Result<usize> {
        loop {
            pos = skip_ws(&self.buf, pos);
            if pos < self.buf.len() || self.done {
                return Ok(pos);
            }
            self.fill(self.block)?;
        }
    }

    /// One JSON value starting at `pos`, and the offset just past it; `None`
    /// when the bytes there are not one. A value cut off by the end of `buf`
    /// is retried with more of the input, doubling each time so a large one
    /// is parsed a bounded number of times over. (A bare number cut off is
    /// not seen as cut off — serde accepts the prefix — and the walk then
    /// declines at the next byte; the whole-file pass reads it right.)
    fn value<T: DeserializeOwned>(&mut self, pos: usize) -> io::Result<Option<(T, usize)>> {
        let mut more = self.block;
        loop {
            match value_at(&self.buf, pos) {
                Some(Ok(parsed)) => return Ok(Some(parsed)),
                Some(Err(err)) if !err.is_eof() => return Ok(None),
                _ if self.done => return Ok(None),
                _ => {
                    self.fill(more)?;
                    more *= 2;
                }
            }
        }
    }
}

/// Append up to `n` bytes of `reader` to `into`; fewer only at its end,
/// which sets `done`.
fn read_block(
    reader: &mut (impl Read + ?Sized),
    done: &mut bool,
    into: &mut Vec<u8>,
    n: usize,
) -> io::Result<()> {
    if *done {
        return Ok(());
    }
    // Exactly, not amortized: a buffer that doubled would hold two blocks
    // for the rest of the read.
    into.reserve_exact(n);
    let got = reader.take(n as u64).read_to_end(into)?;
    if got < n {
        *done = true;
    }
    Ok(())
}

/// The annotations of a results file, whichever of its two shapes it takes.
fn stream_results(src: &mut Source) -> io::Result<Option<Vec<Annotation>>> {
    let pos = src.skip_ws(0)?;
    if src.byte(pos)? != Some(b'[') {
        return Ok(stream_dataset(src)?.map(|dataset| dataset.annotations));
    }
    let Some((anns, end)) = stream_array(src, pos, true)? else {
        return Ok(None);
    };
    let end = src.skip_ws(end)?;
    Ok((end == src.buf.len()).then_some(anns))
}

/// The hand walk over a dataset object, or `None` for a shape it does not
/// expect, which the serde derive then judges. Keys outside the schema are
/// ignored, a missing `annotations` reads as none, and a repeated key keeps
/// its last value, as Python's `json` does. The struct literal at
/// the end is deliberate: a field added to [`Dataset`] fails to compile here
/// instead of being silently dropped.
fn stream_dataset(src: &mut Source) -> io::Result<Option<Dataset>> {
    let (mut info, mut licenses) = Default::default();
    let (mut images, mut annotations, mut categories) = (Vec::new(), Vec::new(), Vec::new());
    let mut pos = src.skip_ws(0)?;
    if src.byte(pos)? != Some(b'{') {
        return Ok(None);
    }
    pos = src.skip_ws(pos + 1)?;
    if src.byte(pos)? != Some(b'}') {
        loop {
            let Some((key, next)) = src.value::<String>(pos)? else {
                return Ok(None);
            };
            pos = src.skip_ws(next)?;
            if src.byte(pos)? != Some(b':') {
                return Ok(None);
            }
            pos = src.skip_ws(pos + 1)?;
            let next = match key.as_str() {
                "info" => take(src, pos, &mut info)?,
                "licenses" => take(src, pos, &mut licenses)?,
                "images" => take_array(src, pos, &mut images, false)?,
                "categories" => take_array(src, pos, &mut categories, false)?,
                "annotations" => take_array(src, pos, &mut annotations, true)?,
                _ => src.value::<IgnoredAny>(pos)?.map(|(_, next)| next),
            };
            let Some(next) = next else {
                return Ok(None);
            };
            pos = src.skip_ws(next)?;
            match src.byte(pos)? {
                Some(b',') => pos = src.skip_ws(pos + 1)?,
                Some(b'}') => break,
                _ => return Ok(None),
            }
        }
    }
    let end = src.skip_ws(pos + 1)?;
    Ok((end == src.buf.len()).then_some(Dataset {
        info,
        images,
        annotations,
        categories,
        licenses,
    }))
}

/// Parse the value at `pos` into `slot`; the offset just past it.
fn take<T: DeserializeOwned>(
    src: &mut Source,
    pos: usize,
    slot: &mut T,
) -> io::Result<Option<usize>> {
    Ok(src.value(pos)?.map(|(value, next)| {
        *slot = value;
        next
    }))
}

/// [`take`] for an array of records: streamed when it is one, serial
/// otherwise. `bulk` says the array is the rest of the file, near enough:
/// its record density projects over what remains, and it is worth parsing
/// in parallel runs.
fn take_array<T: DeserializeOwned + Send>(
    src: &mut Source,
    pos: usize,
    slot: &mut Vec<T>,
    bulk: bool,
) -> io::Result<Option<usize>> {
    if src.byte(pos)? != Some(b'[') {
        return take(src, pos, slot);
    }
    Ok(stream_array(src, pos, bulk)?.map(|(records, next)| {
        *slot = records;
        next
    }))
}

/// The array of records whose `[` is at `open`, a block at a time, and the
/// offset just past its `]` in the buffer as it then stands. `None` when the
/// array is not one of objects or the bytes are malformed: the source may
/// then be part way through the input, and the caller starts over.
fn stream_array<T: DeserializeOwned + Send>(
    src: &mut Source,
    open: usize,
    bulk: bool,
) -> io::Result<Option<(Vec<T>, usize)>> {
    let mut pos = src.skip_ws(open + 1)?;
    match src.byte(pos)? {
        Some(b']') => return Ok(Some((Vec::new(), pos + 1))),
        Some(b'{') => {}
        _ => return Ok(None),
    }
    let mut out: Vec<T> = Vec::new();
    // Guessing pays only for the bulk array: the cut is the last record
    // start in the buffer, which for a smaller array lies in whatever
    // follows it, and the runs past its end are parsed for nothing. Off, too,
    // once a block's guesses fail: the array's records carry `},{` inside
    // them, and every block would fail the same way.
    let mut guess = bulk;
    loop {
        // Cut at the last record start in the buffer, parse up to it in
        // parallel runs while the next block reads, and if every run hands
        // over exactly, carry on from that block.
        let cut = if guess && !src.done {
            last_record_start(&src.buf, pos + 1)
        } else {
            None
        };
        if let Some(stop) = cut {
            let (spanned, read) =
                src.parse_while_reading(stop, move |buf| parse_span::<T>(buf, pos, Some(stop)));
            read?;
            match spanned {
                Some((runs, RunEnd::Continues(_))) => {
                    let projected = if bulk { src.beyond(stop) } else { 0 };
                    append(&mut out, runs, projected, stop - pos);
                    src.swap(stop);
                    pos = 0;
                    continue;
                }
                Some((runs, RunEnd::Ends(after))) => {
                    append(&mut out, runs, 0, 0);
                    src.stitch(stop);
                    out.shrink_to_fit();
                    return Ok(Some((out, after)));
                }
                None => {
                    src.stitch(stop);
                    guess = false;
                }
            }
        }
        // Immune to guesses: at the end of the input in parallel runs to the
        // `]` if guessing is still on, else record by record up to the one
        // cut off.
        let spanned = (guess && src.done)
            .then(|| parse_span(&src.buf, pos, None))
            .flatten()
            .or_else(|| {
                parse_prefix(&src.buf, pos, src.done).map(|(records, end)| (vec![records], end))
            });
        let Some((runs, end)) = spanned else {
            return Ok(None);
        };
        match end {
            RunEnd::Ends(after) => {
                append(&mut out, runs, 0, 0);
                out.shrink_to_fit();
                return Ok(Some((out, after)));
            }
            RunEnd::Continues(next) if next == pos => src.fill(src.block)?,
            RunEnd::Continues(next) => {
                let projected = if bulk { src.beyond(next) } else { 0 };
                append(&mut out, runs, projected, next - pos);
                src.discard(next);
                pos = 0;
                src.fill(src.block)?;
            }
        }
    }
}

/// Move `runs` into `out`. On the first, `out` is reserved for these records
/// and the number their density projects over `beyond` more bytes, with a
/// little slack; later, only if that fell short.
fn append<T>(out: &mut Vec<T>, runs: Vec<Vec<T>>, beyond: u64, consumed: usize) {
    let parsed: usize = runs.iter().map(Vec::len).sum();
    if out.capacity() - out.len() < parsed {
        if out.is_empty() && consumed > 0 {
            let projected = (parsed as f64 * beyond as f64 / consumed as f64) as usize;
            out.reserve_exact(parsed + projected + projected / 16);
        } else {
            out.reserve(parsed);
        }
    }
    for run in runs {
        out.extend(run);
    }
}

/// The serial parse of a whole dataset file.
fn dataset_whole(bytes: &[u8]) -> serde_json::Result<Dataset> {
    let mut dataset: Dataset = serde_json::from_slice(bytes)?;
    dataset.annotations.shrink_to_fit();
    Ok(dataset)
}

/// The serial parse of a whole results file, whichever of its two shapes it takes.
fn results_whole(bytes: &[u8]) -> serde_json::Result<Vec<Annotation>> {
    if bytes.get(skip_ws(bytes, 0)) == Some(&b'[') {
        let mut anns: Vec<Annotation> = serde_json::from_slice(bytes)?;
        anns.shrink_to_fit();
        Ok(anns)
    } else {
        dataset_whole(bytes).map(|dataset| dataset.annotations)
    }
}

/// How many runs to cut `len` bytes of array into: none below one chunk's
/// worth, and no more than a few per thread.
fn chunk_count(len: usize) -> usize {
    (len / MIN_CHUNK_BYTES).min(crate::RUNS_PER_THREAD * rayon::current_num_threads())
}

/// Below this many bytes per chunk, cutting the array up costs more than the
/// parallel parse saves.
const MIN_CHUNK_BYTES: usize = 64 * 1024;

/// The records from `start`, the first of a run, in parallel runs cut at
/// guessed boundaries, up to `stop` — a record start the runs must hand over
/// at exactly — or, with none, the array's `]`; the runs, and which of the
/// two ended them. `None` when a run fails: a boundary guess inside a
/// record, or bytes that are not records.
///
/// The array may end before `stop` does (a dataset's `categories` usually
/// follow it), so the first run to reach a `]` ends the span, every run
/// before it must hand over exactly at the next run's start, and the runs
/// that start past the array's end are discarded.
fn parse_span<T: DeserializeOwned + Send>(
    bytes: &[u8],
    start: usize,
    stop: Option<usize>,
) -> Option<(Vec<Vec<T>>, RunEnd)> {
    let limit = stop.unwrap_or(bytes.len());
    let chunks = chunk_count(limit - start);
    let mut starts: Vec<usize> = std::iter::once(start)
        .chain(
            (1..chunks)
                .filter_map(|i| record_start(bytes, start + (limit - start) / chunks * i))
                .filter(|&guess| guess < limit),
        )
        .collect();
    starts.dedup();
    let runs: Vec<_> = (0..starts.len())
        .into_par_iter()
        .map(|i| parse_run(bytes, starts[i], starts.get(i + 1).copied().or(stop)))
        .collect();
    let mut kept = Vec::with_capacity(runs.len());
    let mut last = RunEnd::Continues(start);
    for run in runs {
        let (records, end) = run?;
        kept.push(records);
        last = end;
        if let RunEnd::Ends(_) = end {
            break;
        }
    }
    Some((kept, last))
}

/// Where a run of records stopped.
#[derive(Clone, Copy)]
enum RunEnd {
    /// At a record start: the next run's, or one the buffer cuts off.
    Continues(usize),
    /// At the array's `]`; the offset just past it.
    Ends(usize),
}

/// The records from `start` to exactly `end` (the next run's start) or to
/// the array's `]`, whichever comes first. `None` for anything else: not
/// records, or records that overran the next run's start — a boundary guess
/// inside a record, a run that began past the array, or a malformed file.
fn parse_run<T: DeserializeOwned>(
    bytes: &[u8],
    start: usize,
    end: Option<usize>,
) -> Option<(Vec<T>, RunEnd)> {
    let mut out = Vec::new();
    let mut pos = start;
    let stop = loop {
        let (record, next) = value_at::<T>(bytes, pos)?.ok()?;
        out.push(record);
        pos = skip_ws(bytes, next);
        match bytes.get(pos) {
            Some(b',') => {
                pos = skip_ws(bytes, pos + 1);
                if end.is_some_and(|end| pos >= end) {
                    if Some(pos) != end {
                        return None;
                    }
                    break RunEnd::Continues(pos);
                }
            }
            Some(b']') => break RunEnd::Ends(pos + 1),
            _ => return None,
        }
    };
    Some((out, stop))
}

/// The records from `start`, one after another, to the array's `]` or,
/// unless `done` says the bytes are all there is, to the record the bytes
/// cut off, which is where the next block's parse starts. `None` for bytes
/// that are not records.
fn parse_prefix<T: DeserializeOwned>(
    bytes: &[u8],
    start: usize,
    done: bool,
) -> Option<(Vec<T>, RunEnd)> {
    let mut out = Vec::new();
    let mut pos = start;
    loop {
        let (record, next) = match value_at::<T>(bytes, pos) {
            Some(Ok(parsed)) => parsed,
            Some(Err(err)) if !err.is_eof() => return None,
            _ if done => return None,
            _ => return Some((out, RunEnd::Continues(pos))),
        };
        let after = skip_ws(bytes, next);
        match bytes.get(after) {
            Some(b',') => {
                out.push(record);
                pos = skip_ws(bytes, after + 1);
            }
            Some(b']') => {
                out.push(record);
                return Some((out, RunEnd::Ends(after + 1)));
            }
            None if !done => return Some((out, RunEnd::Continues(pos))),
            _ => return None,
        }
    }
}

/// Whether the `{` at `brace` follows a `}` and a `,`, with JSON whitespace
/// allowed around the comma: where a record starts, unless the bytes are
/// inside a record.
fn is_record_start(bytes: &[u8], brace: usize) -> bool {
    last_non_ws(bytes, brace).is_some_and(|comma| {
        bytes[comma] == b',' && last_non_ws(bytes, comma).is_some_and(|close| bytes[close] == b'}')
    })
}

/// The first record start at or after `from`.
fn record_start(bytes: &[u8], from: usize) -> Option<usize> {
    let mut pos = from;
    loop {
        pos += memchr::memchr(b'{', bytes.get(pos..)?)?;
        if is_record_start(bytes, pos) {
            return Some(pos);
        }
        pos += 1;
    }
}

/// The last record start at or after `from`.
fn last_record_start(bytes: &[u8], from: usize) -> Option<usize> {
    let mut end = bytes.len();
    loop {
        let brace = from + memchr::memrchr(b'{', bytes.get(from..end)?)?;
        if is_record_start(bytes, brace) {
            return Some(brace);
        }
        end = brace;
    }
}

/// The last byte before `pos` that is not JSON whitespace.
fn last_non_ws(bytes: &[u8], pos: usize) -> Option<usize> {
    bytes[..pos].iter().rposition(|&b| !is_ws(b))
}

/// One JSON value starting at `pos`, and the offset just past it; `None`
/// past the end of `bytes`, or when only whitespace follows.
fn value_at<'de, T: serde::Deserialize<'de>>(
    bytes: &'de [u8],
    pos: usize,
) -> Option<serde_json::Result<(T, usize)>> {
    let mut stream = serde_json::Deserializer::from_slice(bytes.get(pos..)?).into_iter::<T>();
    let next = stream.next()?;
    Some(next.map(|value| (value, pos + stream.byte_offset())))
}

fn skip_ws(bytes: &[u8], mut pos: usize) -> usize {
    while bytes.get(pos).is_some_and(|&b| is_ws(b)) {
        pos += 1;
    }
    pos
}

/// JSON's whitespace, which is narrower than ASCII's.
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// Normalize non-finite JSON float tokens (`NaN`, `Infinity`, `-Infinity`) to
/// `null`, matching the leniency of Python's `json` module.
///
/// Python emits these bare tokens by default and reads them back, so files
/// produced by pycocotools / numpy pipelines frequently contain them, even
/// though they are not valid JSON. serde_json (correctly) rejects them. To load
/// such files, each non-finite token is rewritten to `null` — which serde also
/// uses when *serializing* a non-finite `f64` — but only when the token appears
/// outside a JSON string, so string values that merely contain the substring
/// `"NaN"`/`"Infinity"` — a file name, say — are left untouched. On `Option<f64>`
/// fields (`area`, `score`) the `null` deserializes to `None`.
///
/// Returns the input unchanged and borrowed (no allocation) when it contains no
/// such tokens, so the common case pays only a single linear scan. The second
/// element is the number of tokens rewritten.
fn sanitize_non_finite(input: &[u8]) -> (Cow<'_, [u8]>, usize) {
    // Prefilter: if the tokens never occur as substrings *anywhere* — even
    // inside strings, where they would not count — the scan below cannot
    // rewrite anything. Two SIMD substring searches cost ~1ms on a 19 MB
    // file; the byte-at-a-time state machine they skip cost ~24ms, paid on
    // every load of a clean file, which is nearly every load. ("-Infinity"
    // contains "Infinity", so two needles cover all three tokens.)
    if memchr::memmem::find(input, b"NaN").is_none()
        && memchr::memmem::find(input, b"Infinity").is_none()
    {
        return (Cow::Borrowed(input), 0);
    }

    let n = input.len();
    let mut out: Option<Vec<u8>> = None;
    let mut count = 0usize;
    let mut in_string = false;
    let mut i = 0;

    while i < n {
        let b = input[i];

        if in_string {
            if b == b'\\' {
                // Copy the backslash and the escaped byte verbatim so an
                // escaped quote (`\"`) does not toggle the string state.
                if let Some(o) = out.as_mut() {
                    o.push(b);
                    if i + 1 < n {
                        o.push(input[i + 1]);
                    }
                }
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
            if let Some(o) = out.as_mut() {
                o.push(b);
            }
            i += 1;
            continue;
        }

        if b == b'"' {
            in_string = true;
            if let Some(o) = out.as_mut() {
                o.push(b);
            }
            i += 1;
            continue;
        }

        // Outside a string, the only bare identifier-like tokens are
        // true/false/null and the non-finite floats we rewrite here. Gate the
        // substring comparisons on the first byte so the common case (digits,
        // punctuation, whitespace) skips them entirely.
        let token_len = match b {
            b'N' if input[i..].starts_with(b"NaN") => Some(3),
            b'I' if input[i..].starts_with(b"Infinity") => Some(8),
            b'-' if input[i..].starts_with(b"-Infinity") => Some(9),
            _ => None,
        };

        if let Some(len) = token_len {
            let o = out.get_or_insert_with(|| {
                let mut v = Vec::with_capacity(n);
                v.extend_from_slice(&input[..i]);
                v
            });
            o.extend_from_slice(b"null");
            count += 1;
            i += len;
            continue;
        }

        if let Some(o) = out.as_mut() {
            o.push(b);
        }
        i += 1;
    }

    match out {
        Some(v) => (Cow::Owned(v), count),
        None => (Cow::Borrowed(input), count),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod sanitize_tests {
    use super::sanitize_non_finite;

    fn run(s: &str) -> (String, usize) {
        let (bytes, n) = sanitize_non_finite(s.as_bytes());
        (String::from_utf8(bytes.into_owned()).unwrap(), n)
    }

    #[test]
    fn clean_input_is_borrowed_unchanged() {
        let input = br#"{"a": [1.0, -2.5], "b": null}"#;
        let (bytes, n) = sanitize_non_finite(input);
        assert_eq!(n, 0);
        assert!(matches!(bytes, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn rewrites_the_non_finite_family() {
        let (out, n) = run(r#"{"a": NaN, "b": Infinity, "c": -Infinity, "d": -3.5}"#);
        assert_eq!(n, 3);
        // -3.5 (a real negative number) must be preserved, not mangled.
        assert_eq!(out, r#"{"a": null, "b": null, "c": null, "d": -3.5}"#);
    }

    #[test]
    fn leaves_non_finite_substrings_inside_strings_alone() {
        // Strings containing the tokens — including an escaped quote — untouched.
        let (out, n) = run(r#"{"name": "NaN and \"Infinity\"", "v": NaN}"#);
        assert_eq!(n, 1);
        assert_eq!(out, r#"{"name": "NaN and \"Infinity\"", "v": null}"#);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(i: u64, tail: &str) -> String {
        format!(
            r#"{{"image_id": {i}, "category_id": {}, "bbox": [1.5, 2.25, 3.125, 4.0625], "score": 0.{i}{tail}}}"#,
            i % 7 + 1
        )
    }

    /// A record with a nested object, so a `{` is not always a record start.
    fn nested(i: u64) -> String {
        record(
            i,
            r#", "attributes": {"occluded": false, "tags": [{"k": 1}]}"#,
        )
    }

    /// A record with a nested list of objects: a `},{` that is not a record
    /// boundary, the shape panoptic `segments_info` takes.
    fn listed(i: u64) -> String {
        record(
            i,
            r#", "segments_info": [{"id": 1, "area": 2}, {"id": 3, "area": 4}, {"id": 5}]"#,
        )
    }

    fn array(records: impl IntoIterator<Item = String>, sep: &str) -> Vec<u8> {
        let records: Vec<String> = records.into_iter().collect();
        format!("[\n{}\n]\n", records.join(sep)).into_bytes()
    }

    fn serial(bytes: &[u8]) -> Vec<Annotation> {
        serde_json::from_slice(bytes).expect("fixture parses serially")
    }

    /// Serialized form, so a comparison covers every field bit for bit (ryu
    /// output round-trips) and a failure prints the differing record.
    fn json(anns: &[Annotation]) -> String {
        serde_json::to_string(anns).expect("annotations serialize")
    }

    fn source(bytes: &[u8], block: usize) -> Source<'_> {
        Source::new(Cursor::new(bytes), bytes.len() as u64, block)
    }

    /// Block sizes from a few bytes (every record spans blocks) to more than
    /// any fixture (one block, the whole input).
    const BLOCKS: [usize; 7] = [7, 16, 64, 256, 1000, 4096, 1 << 20];

    /// Stream `bytes` as a results file at every block size: the records
    /// must match the serial parse, hold no slack, and the buffers must stay
    /// a few blocks, never growing to the file.
    fn streams_like_serial(bytes: &[u8]) {
        let expected = serial(bytes);
        for block in BLOCKS {
            let mut src = source(bytes, block);
            let got = stream_results(&mut src)
                .expect("read")
                .unwrap_or_else(|| panic!("a clean array streams at block={block}"));
            assert_eq!(json(&got), json(&expected), "block={block}");
            assert_eq!(got.capacity(), got.len(), "block={block}");
            let held = src.buf.capacity() + src.spare.capacity();
            assert!(
                held < 8 * block || held * 2 < bytes.len(),
                "block={block}: the buffers held {held} of {} bytes",
                bytes.len()
            );
        }
    }

    #[test]
    fn streamed_parse_matches_serial_at_every_block_size() {
        // Long records first, so the density of the first block projects far
        // fewer records than the short ones that follow amount to.
        let long = format!(r#", "caption": "{}""#, "x".repeat(400));
        streams_like_serial(&array(
            (0..300).map(|i| match i {
                0..20 => record(i, &long),
                i if i % 3 == 0 => nested(i),
                i => record(i, ""),
            }),
            ",\n  ",
        ));
    }

    #[test]
    fn records_with_nested_object_lists_stream_serially() {
        streams_like_serial(&array((0..300).map(listed), ", "));
        // Mixed in, they only take the blocks that hold them off the
        // parallel path.
        streams_like_serial(&array(
            (0..300).map(|i| {
                if i % 50 == 25 {
                    listed(i)
                } else {
                    record(i, "")
                }
            }),
            ",",
        ));
    }

    #[test]
    fn boundary_guess_inside_a_string_is_rejected_not_misparsed() {
        // Every third record carries a string that looks like a record boundary.
        let bytes = array(
            (0..90).map(|i| {
                if i % 3 == 0 {
                    record(
                        i,
                        r#", "caption": "a}, {\"image_id\": 1, \"category_id\": 1}, {""#,
                    )
                } else {
                    record(i, "")
                }
            }),
            ", ",
        );
        assert_eq!(serial(&bytes).len(), 90);
        streams_like_serial(&bytes);
        for stop in [None, Some(bytes.len() - 3)] {
            if let Some((runs, _)) = parse_span::<Annotation>(&bytes, skip_ws(&bytes, 1), stop) {
                let got: Vec<Annotation> = runs.into_iter().flatten().collect();
                assert!(json(&serial(&bytes)).starts_with(json(&got).trim_end_matches(']')));
            }
        }
    }

    #[test]
    fn a_run_that_overruns_its_end_is_rejected() {
        let bytes = array((0..3).map(|i| record(i, "")), ",");
        let second = record_start(&bytes, 2).expect("three records have a second start");
        // Cutting the first run one byte before the real boundary must fail,
        // not hand back a record that belongs to the next run.
        assert!(parse_run::<Annotation>(&bytes, skip_ws(&bytes, 1), Some(second - 1)).is_none());
        let Some((run, RunEnd::Continues(at))) =
            parse_run::<Annotation>(&bytes, skip_ws(&bytes, 1), Some(second))
        else {
            panic!("exact boundary parses");
        };
        assert_eq!((run.len(), at), (1, second));
        // The last run reports where the array ends, whatever follows it.
        let Some((rest, RunEnd::Ends(end))) = parse_run::<Annotation>(&bytes, second, None) else {
            panic!("last run reaches the closing bracket");
        };
        assert_eq!((rest.len(), &bytes[end - 1..end]), (2, &b"]"[..]));
    }

    #[test]
    fn a_serial_prefix_stops_at_the_record_the_bytes_cut_off() {
        let bytes = array((0..3).map(|i| record(i, "")), ",");
        let third = last_record_start(&bytes, 1).expect("three records have a last start");
        let Some((records, RunEnd::Continues(at))) =
            parse_prefix::<Annotation>(&bytes[..third + 20], skip_ws(&bytes, 1), false)
        else {
            panic!("two whole records and a cut-off third");
        };
        assert_eq!((records.len(), at), (2, third));
        // Cut right after a record, before its comma, the record is redone.
        let close = third
            - bytes[..third]
                .iter()
                .rev()
                .position(|&b| b == b'}')
                .unwrap_or(0);
        let Some((records, RunEnd::Continues(at))) =
            parse_prefix::<Annotation>(&bytes[..close], skip_ws(&bytes, 1), false)
        else {
            panic!("one whole record and one without its comma");
        };
        assert!(records.len() == 1 && at < close && at > skip_ws(&bytes, 1));
        assert!(
            parse_prefix::<Annotation>(&bytes[..third + 20], skip_ws(&bytes, 1), true).is_none()
        );
        let Some((records, RunEnd::Ends(end))) =
            parse_prefix::<Annotation>(&bytes, skip_ws(&bytes, 1), true)
        else {
            panic!("the whole array");
        };
        assert_eq!((records.len(), &bytes[end - 1..end]), (3, &b"]"[..]));
    }

    #[test]
    fn record_starts_skip_nested_objects_and_whitespace() {
        let bytes = b"[{\"a\": 1}, {\"b\": {\"c\": 2}} ,\n {\"d\": {\"e\": {}}}]";
        let second = bytes.iter().position(|&b| b == b'b').map_or(0, |p| p - 2);
        let last = last_record_start(bytes, 1).expect("three records");
        assert_eq!(&bytes[last..last + 5], b"{\"d\":");
        assert_eq!(record_start(bytes, 0), Some(second));
        assert_eq!(record_start(bytes, second + 1), Some(last));
        assert_eq!(last_record_start(bytes, last + 1), None);
        assert_eq!(last_record_start(b"[{\"a\": {}}]", 1), None);
    }

    #[test]
    fn results_shape_is_decided_by_the_first_byte() {
        let array = array((0..2).map(|i| record(i, "")), ",");
        let object = format!(
            r#" {{"images": [], "annotations": {}}}"#,
            String::from_utf8_lossy(&array)
        );
        let from_array = stream_results(&mut source(&array, 32)).expect("read");
        let from_object = stream_results(&mut source(object.as_bytes(), 32)).expect("read");
        assert_eq!(
            from_array.as_deref().map(json).expect("array form"),
            from_object.as_deref().map(json).expect("object form")
        );
        assert_eq!(
            json(&results_whole(object.as_bytes()).expect("serial object form")),
            json(&serial(&array))
        );
    }

    #[test]
    fn a_dirty_file_is_sanitized_only_after_a_clean_parse_fails() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("res.json");
        std::fs::write(
            &path,
            br#"[{"image_id": 1, "category_id": 1, "score": NaN}]"#,
        )
        .expect("write");
        let (anns, n_fixed) = read_results(&path).expect("NaN reads as null");
        assert_eq!((anns.len(), n_fixed, anns[0].score), (1, 1, None));
        std::fs::write(&path, br#"[{"image_id": 1,"#).expect("write");
        let err = read_results(&path).expect_err("truncated file fails");
        assert!(matches!(err, crate::error::Error::Json(_)), "{err}");
    }

    #[test]
    fn non_object_arrays_and_malformed_input_go_serial() {
        let stream = |bytes: &[u8]| stream_results(&mut source(bytes, 16)).expect("read");
        assert_eq!(stream(b"[]").map(|anns| anns.len()), Some(0));
        assert_eq!(stream(b" [ ] ").map(|anns| anns.len()), Some(0));
        assert!(stream(b"[1, 2]").is_none());
        assert!(stream(b"[{\"image_id\": 1}] trailing").is_none());
        assert_eq!(
            stream(b"{\"annotations\": []}").map(|anns| anns.len()),
            Some(0)
        );
        let truncated = &array((0..5).map(|i| record(i, "")), ",")[..80];
        assert!(parse_span::<Annotation>(truncated, 2, None).is_none());
        let err = results_whole(truncated).expect_err("truncated input fails");
        assert!(err.to_string().contains("line"), "{err}");
        for block in BLOCKS {
            assert!(
                stream_results(&mut source(truncated, block))
                    .expect("read")
                    .is_none(),
                "block={block}"
            );
        }
    }

    #[test]
    fn dataset_object_ignores_unknown_keys_and_defaults_missing_ones() {
        let dataset = |bytes: &[u8]| stream_dataset(&mut source(bytes, 16)).expect("read");
        let json = br#"{"custom": {"nested": [1, 2]}, "images": [{"id": 3, "width": 4, "height": 5}],
                       "annotations": [{"id": 9, "image_id": 3, "category_id": 1, "bbox": [0, 0, 1, 1]}],
                       "categories": [{"id": 1, "name": "x"}]}"#;
        let ds = dataset(json).expect("dataset parses");
        assert_eq!(ds.images.len(), 1);
        assert_eq!(ds.annotations[0].id, 9);
        assert_eq!(ds.categories[0].name, "x");
        assert!(ds.info.is_none());
        let ds = dataset(br#"{"images": []}"#).expect("dataset parses");
        assert!(ds.annotations.is_empty());
        assert!(
            dataset(br"{}")
                .expect("empty object")
                .annotations
                .is_empty()
        );
        // Repeated keys keep the last value, as Python's `json` does.
        let ds = dataset(br#"{"images": [{"id": 1}], "annotations": [], "images": []}"#)
            .expect("repeated key");
        assert!(ds.images.is_empty() && ds.annotations.is_empty());
        // Shapes the walk declines, and the derive's verdict on each.
        for bytes in [
            &br#"{"images": [] trailing"#[..],
            br#"["not", "an", "object"]"#,
            br#"{"images": [}"#,
            br#"{"images": null}"#,
            br#"{"annotations": null}"#,
            br#"{"annotations": nul}"#,
            br"[1]",
        ] {
            assert!(
                dataset(bytes).is_none(),
                "{}",
                String::from_utf8_lossy(bytes)
            );
            assert!(
                dataset_whole(bytes).is_err(),
                "{}",
                String::from_utf8_lossy(bytes)
            );
        }
    }

    #[test]
    fn dataset_walk_streams_every_record_array_whatever_follows_it() {
        // Categories after the annotations, unknown keys before and after,
        // images that span blocks, and enough bytes that chunk targets land
        // inside the categories.
        let imgs: Vec<String> = (0..500)
            .map(|i| {
                format!(
                    r#"{{"id": {i}, "width": {}, "height": 3, "file_name": "{i}.jpg"}}"#,
                    i % 9
                )
            })
            .collect();
        let anns: Vec<String> = (0..1000)
            .map(|i| if i % 5 == 0 { nested(i) } else { record(i, "") })
            .collect();
        let cats: Vec<String> = (0..2000)
            .map(|i| format!(r#"{{"id": {i}, "name": "cat{i}", "supercategory": "s"}}"#))
            .collect();
        let doc = format!(
            "{{\n \"info\": {{\"year\": 2026}}, \"extra\": [{{\"a\": 1}}, {{\"b\": 2}}],\n \"images\": [{}],\n \"annotations\": [\n{}\n],\n \"categories\": [{}],\n \"licenses\": [], \"more\": {{\"x\": [1, 2]}}\n}}\n",
            imgs.join(", "),
            anns.join(",\n"),
            cats.join(", ")
        );
        let slow = dataset_whole(doc.as_bytes()).expect("the derive parses it too");
        for block in BLOCKS {
            let mut src = source(doc.as_bytes(), block);
            let fast = stream_dataset(&mut src)
                .expect("read")
                .unwrap_or_else(|| panic!("the walk handles this shape at block={block}"));
            assert_eq!(
                json(&fast.annotations),
                json(&slow.annotations),
                "block={block}"
            );
            assert_eq!(fast.annotations.capacity(), 1000, "block={block}");
            assert_eq!(
                (fast.images.len(), fast.images.capacity()),
                (500, 500),
                "block={block}"
            );
            assert_eq!(fast.images[499].file_name, slow.images[499].file_name);
            assert_eq!(
                (fast.categories.len(), fast.categories.capacity()),
                (2000, 2000)
            );
            assert_eq!(fast.categories[1999].name, "cat1999");
            assert_eq!(fast.info.map(|i| i.year), Some(Some(2026)));
        }
        // In one span, the guessed starts past the array's end are discarded.
        let open = doc.find("\"annotations\": [").unwrap_or(0) + "\"annotations\": ".len();
        let first = skip_ws(doc.as_bytes(), open + 1);
        assert!(
            chunk_count(doc.len() - first) >= 2,
            "the fixture must be worth cutting up"
        );
        let (runs, end) =
            parse_span::<Annotation>(doc.as_bytes(), first, None).expect("chunked in place");
        assert_eq!(runs.iter().map(Vec::len).sum::<usize>(), 1000);
        let RunEnd::Ends(end) = end else {
            panic!("the span reaches the array's end");
        };
        assert_eq!(&doc.as_bytes()[end - 1..end], b"]");
    }
}
