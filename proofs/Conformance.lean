/-
Conformance: the Lean model against the Rust tables.

Reads `fixtures/*.json` (emitted by `scripts/lean_fixtures.rs`) and checks
each table against the executable model, so that "model = code" is
tested where "model = spec" is proved. Run with `lake exe conformance`
(or `just lean-conformance`). Exit code 1 on any mismatch.
-/
import Nice
import Lean.Data.Json

open Lean

namespace Conformance

structure Report where
  checks : Nat := 0
  failures : List String := []

def Report.check (r : Report) (name : String) (ok : Bool) : Report :=
  { checks := r.checks + 1, failures := if ok then r.failures else name :: r.failures }

def getNat (j : Json) (k : String) : Except String Nat := do
  let v ← j.getObjVal? k
  match v with
  | .num n => if n.exponent == 0 && n.mantissa ≥ 0 then pure n.mantissa.toNat else throw s!"{k}: not a nat"
  | .str s => match s.toNat? with | some n => pure n | none => throw s!"{k}: not a nat string"
  | _ => throw s!"{k}: not a number"

def getNatOpt (j : Json) (k : String) : Except String (Option Nat) := do
  match j.getObjVal? k with
  | .ok .null => pure none
  | .ok _ => pure (some (← getNat j k))
  | .error e => throw e

def getNatList (j : Json) (k : String) : Except String (List Nat) := do
  let arr ← (← j.getObjVal? k).getArr?
  arr.toList.mapM fun v => match v with
    | .num n => pure n.mantissa.toNat
    | .str s => match s.toNat? with | some n => pure n | none => throw s!"{k}: not a nat string"
    | _ => throw s!"{k}: array element is not a number"

def getArr (j : Json) (k : String) : Except String (List Json) := do
  pure (← (← j.getObjVal? k).getArr?).toList

def load (name : String) : IO (List Json) := do
  let text ← IO.FS.readFile ("fixtures/" ++ name)
  match Json.parse text with
  | .ok j => match j.getArr? with
    | .ok a => pure a.toList
    | .error e => throw (IO.userError s!"{name}: {e}")
  | .error e => throw (IO.userError s!"{name}: {e}")

def sortedList (s : Finset ℕ) : List ℕ := s.sort (· ≤ ·)

def orFail {α} (e : Except String α) : IO α :=
  match e with
  | .ok a => pure a
  | .error s => throw (IO.userError s)

def checkResidues (r : Report) : IO Report := do
  let mut r := r
  for j in ← load "residue.json" do
    let b ← orFail (getNat j "base")
    let want ← orFail (getNatList j "residues")
    r := r.check s!"residueFilter {b}" (sortedList (NiceSearch.residueFilter b) == want)
  pure r

def checkLsd (r : Report) : IO Report := do
  let mut r := r
  for j in ← load "lsd.json" do
    let b ← orFail (getNat j "base")
    let k ← orFail (getNat j "k")
    let want ← orFail (getNatList j "valid")
    r := r.check s!"lsdBitmap {b} {k}" (sortedList (NiceSearch.lsdBitmap b k) == want)
  pure r

/-- Is `n` the least valid number at or after `start`? Checked by scanning. -/
def leastValidFrom (b k start n : ℕ) : Bool :=
  start ≤ n && decide (NiceSearch.IsValid b k n) &&
    (List.range (n - start)).all fun d => !decide (NiceSearch.IsValid b k (start + d))

def checkStride (r : Report) : IO Report := do
  let mut r := r
  for j in ← load "stride.json" do
    let b ← orFail (getNat j "base")
    let k ← orFail (getNat j "k")
    let modulus ← orFail (getNat j "modulus")
    let residues ← orFail (getNatList j "valid_residues")
    let gaps ← orFail (getNatList j "gap_table")
    r := r.check s!"strideModulus {b} {k}" (NiceSearch.strideModulus b k == modulus)
    let model := NiceSearch.residueList b k
    r := r.check s!"validResidues {b} {k}" (model == residues)
    -- gaps: each entry is the distance to the next residue, wrapping at the modulus
    let n := residues.length
    let wantGaps := (List.range n).map fun i =>
      if i + 1 < n then residues.getD (i + 1) 0 - residues.getD i 0
      else modulus - residues.getD i 0 + residues.getD 0 0
    r := r.check s!"gap_table {b} {k}" (gaps == wantGaps)
    -- low digit sets per residue (STR-4)
    let lows ← orFail (getArr j "low_digits")
    let lowSets ← orFail (lows.mapM fun v => do
      let a ← v.getArr?
      a.toList.mapM fun x => match x with
        | .num m => pure m.mantissa.toNat
        | _ => throw "low_digits element")
    let modelLows := residues.map fun res => sortedList (NiceSearch.lowMask b k res)
    r := r.check s!"low_digit_masks {b} {k}" (modelLows == lowSets)
    -- first_valid_at_or_after against the spec
    for fv in ← orFail (getArr j "first_valid") do
      let start ← orFail (getNat fv "start")
      let nv ← orFail (getNat fv "n")
      let idx ← orFail (getNat fv "idx")
      r := r.check s!"first_valid_at_or_after {b} {k} {start}"
        (leastValidFrom b k start nv && residues.getD idx (modulus + 1) == nv % modulus)
  pure r

/-- The closed-form range is exactly the digit-count set: checked by scanning
past the end (small bases only). -/
def checkRanges (r : Report) : IO Report := do
  let mut r := r
  for j in ← load "range.json" do
    let b ← orFail (getNat j "base")
    let start ← orFail (getNatOpt j "start")
    let stop ← orFail (getNatOpt j "end")
    let ok := match start, stop with
      | some s, some e =>
        (List.range (e + e / 4 + 8)).all fun n =>
          decide (NiceSearch.InBaseRange b n) == (s ≤ n && n < e)
      | none, none => (List.range (b ^ 3 + 8)).all fun n => !decide (NiceSearch.InBaseRange b n)
      | _, _ => false
    r := r.check s!"baseRange {b}" ok
  pure r

def checkSeeded (r : Report) : IO Report := do
  let mut r := r
  for j in ← load "seeded.json" do
    let b ← orFail (getNat j "base")
    let k ← orFail (getNat j "k")
    let samples ← orFail (getArr j "samples")
    for s in samples do
      let arr ← orFail s.getArr?
      match arr.toList with
      | [.str nStr, .bool seeded, .bool plain] =>
        let some n := nStr.toNat? | throw (IO.userError "seeded sample n")
        r := r.check s!"seeded {b} {k} {n}" (decide (NiceSearch.seededDigits b k n).Nodup == seeded)
        r := r.check s!"isNice {b} {n}" (decide (NiceSearch.IsNice b n) == plain)
      | _ => throw (IO.userError "seeded sample shape")
  pure r

def checkMsd (r : Report) : IO Report := do
  let mut r := r
  for j in ← load "msd.json" do
    let b ← orFail (getNat j "base")
    for v in ← orFail (getArr j "verdicts") do
      let arr ← orFail v.getArr?
      match arr.toList with
      | [.str s, .str e, .bool rejected] =>
        let some start := s.toNat? | throw (IO.userError "verdict start")
        let some stop := e.toNat? | throw (IO.userError "verdict end")
        -- Rust `has_duplicate_msd_prefix` is true exactly when the model's
        -- SDR search fails (size-1 ranges are always Live in Rust).
        let model := if stop - start ≤ 1 then false else !NiceSearch.analyzeRange b start (stop - 1)
        r := r.check s!"analyze_range {b} [{start}, {stop})" (model == rejected)
      | _ => throw (IO.userError "verdict shape")
    for l in ← orFail (getArr j "leaves") do
      let arr ← orFail l.getArr?
      match arr.toList with
      | [.str s, .str e, .num d, .str m, .arr leaves] =>
        let some start := s.toNat? | throw (IO.userError "leaves start")
        let some stop := e.toNat? | throw (IO.userError "leaves end")
        let some minSize := m.toNat? | throw (IO.userError "leaves min")
        let depth := d.mantissa.toNat
        let want ← orFail (leaves.toList.mapM fun x => do
          let a ← x.getArr?
          match a.toList with
          | [.str ls, .str le] => match ls.toNat?, le.toNat? with
            | some a, some b => pure (a, b)
            | _, _ => throw "leaf nat"
          | _ => throw "leaf shape")
        r := r.check s!"get_valid_ranges_recursive {b} depth {depth} min {minSize}"
          (NiceSearch.validRanges b minSize depth start stop == want)
      | _ => throw (IO.userError "leaves shape")
  pure r

def checkPipeline (r : Report) : IO Report := do
  let mut r := r
  for j in ← load "pipeline.json" do
    let b ← orFail (getNat j "base")
    let k ← orFail (getNat j "k")
    let start ← orFail (getNat j "start")
    let stop ← orFail (getNat j "end")
    for l in ← orFail (getArr j "masked_leaves") do
      let arr ← orFail l.getArr?
      match arr.toList with
      | [.num d, .str m, .arr leaves] =>
        let depth := d.mantissa.toNat
        let some minSize := m.toNat? | throw (IO.userError "masked min")
        let want ← orFail (leaves.toList.mapM fun x => do
          let a ← x.getArr?
          match a.toList with
          | [.str ls, .str le, .arr ds] =>
            let digits ← ds.toList.mapM fun v => match v with
              | .num n => pure n.mantissa.toNat
              | _ => throw "mask digit"
            match ls.toNat?, le.toNat? with
            | some a, some b => pure (a, b, digits)
            | _, _ => throw "leaf nat"
          | _ => throw "masked leaf shape")
        let model := (NiceSearch.validRangesMasked b k minSize depth start stop ∅).map fun r =>
          (r.1, r.2.1, sortedList r.2.2)
        r := r.check s!"get_valid_ranges_recursive_masked {b} depth {depth} min {minSize}"
          (model == want)
      | _ => throw (IO.userError "masked leaves shape")
    let nice ← orFail (getNatList j "nice")
    -- production constants: MSD_RECURSIVE_MIN_RANGE_SIZE = 4000, MAX_DEPTH = 22
    let model := NiceSearch.processRangeNiceonly b k 4000 22 start stop
    r := r.check s!"process_range_niceonly {b}" (model == nice)
  pure r

def getPair (j : Json) (k : String) : Except String (Nat × Nat) := do
  let arr ← (← j.getObjVal? k).getArr?
  match arr.toList with
  | [.num a, .num b] => pure (a.mantissa.toNat, b.mantissa.toNat)
  | _ => throw s!"{k}: not a pair"

def checkGpuConfig (r : Report) : IO Report := do
  let mut r := r
  for j in ← load "gpu_config.json" do
    let b ← orFail (getNat j "base")
    let (e, d) ← orFail (getPair j "chunk")
    let (e16, d16) ← orFail (getPair j "chunk_u16")
    r := r.check s!"chunk_constants {b}" (NiceSearch.chunkExp b (2 ^ 31) == e && b ^ e == d)
    r := r.check s!"chunk_constants_u16 {b}" (NiceSearch.chunkExp b (2 ^ 16) == e16 && b ^ e16 == d16)
    let pf ← orFail (getNatOpt j "prefilter_digits")
    let start ← orFail (getNatOpt j "range_start")
    match pf, start with
    | some p, some s =>
      -- the Rust depth (float log, minus one for safety) never exceeds the exact depth
      r := r.check s!"prefilter_digits {b}" (p ≤ NiceSearch.prefilterDepth b s)
    | some _, none => r := r.check s!"prefilter_digits {b} without a range" false
    | none, _ => pure ()
  pure r

def checkAffine (r : Report) : IO Report := do
  let mut r := r
  for j in ← load "affine.json" do
    let b ← orFail (getNat j "base")
    for s in ← orFail (getArr j "samples") do
      let arr ← orFail s.getArr?
      match arr.toList with
      | [.str nStr, .arr known, .bool survives] =>
        let some nmod := nStr.toNat? | throw (IO.userError "affine nmod")
        let ks ← orFail (known.toList.mapM fun v => match v with
          | .num m => pure m.mantissa.toNat
          | _ => throw "known digit")
        r := r.check s!"affine_filter::survives {b} {nmod}"
          (NiceSearch.affineSurvives b 3 nmod ks.toFinset == survives)
      | _ => throw (IO.userError "affine sample shape")
  pure r

def digitList (v : Json) : Except String (List Nat) := do
  let a ← v.getArr?
  a.toList.mapM fun x => match x with
    | .num m => pure m.mantissa.toNat
    | _ => throw "digit"

def checkJoin (r : Report) : IO Report := do
  let mut r := r
  for j in ← load "join.json" do
    let b ← orFail (getNat j "base")
    let s ← orFail (getNat j "start")
    let e ← orFail (getNat j "end")
    let t ← orFail (getNat j "t")
    let k ← orFail (getNat j "k")
    let p ← orFail (getNat j "p")
    let L ← orFail (getNat j "l")
    let k2 ← orFail (getNat j "k2")
    let tag := s!"{b} [{s}, {e}) t={t} k={k} p={p}"
    -- certificates and their floors
    for c in ← orFail (getArr j "certs") do
      let arr ← orFail c.getArr?
      match arr.toList with
      | [.str aS, .str eS, .num cap, digits, .num floor] =>
        let some a := aS.toNat? | throw (IO.userError "cert a")
        let some ee := eS.toNat? | throw (IO.userError "cert e")
        let want ← match digits with
          | .null => pure none
          | v => do pure (some (← orFail (digitList v)))
        let cap := cap.mantissa.toNat
        r := r.check s!"cert {tag} [{a}, {ee}] cap {cap}"
          ((NiceSearch.cert b a ee cap).map sortedList == want)
        r := r.check s!"cert_floor {tag} [{a}, {ee}] cap {cap}"
          (NiceSearch.certFloor b a ee cap == floor.mantissa.toNat)
      | _ => throw (IO.userError "cert shape")
    -- the bottom list's first level
    let want ← orFail ((← orFail (getArr j "bpre")).mapM fun q => do
      let a ← q.getArr?
      match a.toList with
      | [.str rS, ds] => match rS.toNat? with
        | some rr => pure (rr, ← digitList ds)
        | none => throw "bpre residue"
      | _ => throw "bpre shape")
    r := r.check s!"bot_dfs {tag}"
      ((NiceSearch.botDfs b (L - t) 0 0 (L - t) 0 0 ∅).map (fun q => (q.1, sortedList q.2)) == want)
    -- every partition: survivors in order, the prefilter's count, the hits
    for part in ← orFail (getArr j "parts") do
      let v ← orFail (getNat part "v")
      let surv ← orFail (getNatList part "survivors")
      let checked ← orFail (getNat part "checked")
      let hits ← orFail (getNatList part "hits")
      let model := NiceSearch.joinPartition b L t k p s e v
      r := r.check s!"join_range {tag} partition {v}" (model == surv)
      r := r.check s!"prefilter {tag} partition {v}"
        ((model.filter fun n => NiceSearch.prefilterAt b L t k k2 s e n).length == checked)
      r := r.check s!"hits {tag} partition {v}"
        (model.filter (fun n => decide (NiceSearch.IsNice b n)) == hits)
    -- the slicing
    for sl in ← orFail (getArr j "slices") do
      let arr ← orFail sl.getArr?
      match arr.toList with
      | [.str mS, .arr ss] =>
        let some m := mS.toNat? | throw (IO.userError "slices max")
        let want ← orFail (ss.toList.mapM fun x => do
          let a ← x.getArr?
          match a.toList with
          | [.str aS, .str bS] => match aS.toNat?, bS.toNat? with
            | some a, some b => pure (a, b)
            | _, _ => throw "slice bound"
          | _ => throw "slice shape")
        let block := b ^ (L - t + p)
        r := r.check s!"join_slices {tag} max {m}"
          (NiceSearch.joinSlices block (block * max m 1) (e - s) s e == want)
      | _ => throw (IO.userError "slices shape")
  pure r

end Conformance

open Conformance in
def main : IO UInt32 := do
  let mut r : Report := {}
  r ← checkResidues r
  r ← checkLsd r
  r ← checkStride r
  r ← checkRanges r
  r ← checkMsd r
  r ← checkPipeline r
  r ← checkGpuConfig r
  r ← checkSeeded r
  r ← checkAffine r
  r ← checkJoin r
  for f in r.failures.reverse do
    IO.println s!"FAIL: {f}"
  IO.println s!"{r.checks} checks, {r.failures.length} failures"
  pure (if r.failures.isEmpty then 0 else 1)
