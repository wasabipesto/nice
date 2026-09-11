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
    r := r.check s!"residueFilter {b}" (sortedList (Nice.residueFilter b) == want)
  pure r

def checkLsd (r : Report) : IO Report := do
  let mut r := r
  for j in ← load "lsd.json" do
    let b ← orFail (getNat j "base")
    let k ← orFail (getNat j "k")
    let want ← orFail (getNatList j "valid")
    r := r.check s!"lsdBitmap {b} {k}" (sortedList (Nice.lsdBitmap b k) == want)
  pure r

/-- Is `n` the least valid number at or after `start`? Checked by scanning. -/
def leastValidFrom (b k start n : ℕ) : Bool :=
  start ≤ n && decide (Nice.IsValid b k n) &&
    (List.range (n - start)).all fun d => !decide (Nice.IsValid b k (start + d))

def checkStride (r : Report) : IO Report := do
  let mut r := r
  for j in ← load "stride.json" do
    let b ← orFail (getNat j "base")
    let k ← orFail (getNat j "k")
    let modulus ← orFail (getNat j "modulus")
    let residues ← orFail (getNatList j "valid_residues")
    let gaps ← orFail (getNatList j "gap_table")
    r := r.check s!"strideModulus {b} {k}" (Nice.strideModulus b k == modulus)
    let model := Nice.residueList b k
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
    let modelLows := residues.map fun res => sortedList (Nice.lowMask b k res)
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
          decide (Nice.InBaseRange b n) == (s ≤ n && n < e)
      | none, none => (List.range (b ^ 3 + 8)).all fun n => !decide (Nice.InBaseRange b n)
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
        r := r.check s!"seeded {b} {k} {n}" (decide (Nice.seededDigits b k n).Nodup == seeded)
        r := r.check s!"isNice {b} {n}" (decide (Nice.IsNice b n) == plain)
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
        let model := if stop - start ≤ 1 then false else !Nice.analyzeRange b start (stop - 1)
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
          (Nice.validRanges b minSize depth start stop == want)
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
        let model := (Nice.validRangesMasked b k minSize depth start stop ∅).map fun r =>
          (r.1, r.2.1, sortedList r.2.2)
        r := r.check s!"get_valid_ranges_recursive_masked {b} depth {depth} min {minSize}"
          (model == want)
      | _ => throw (IO.userError "masked leaves shape")
    let nice ← orFail (getNatList j "nice")
    -- production constants: MSD_RECURSIVE_MIN_RANGE_SIZE = 8000, MAX_DEPTH = 22
    let model := Nice.processRangeNiceonly b k 8000 22 start stop
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
    r := r.check s!"chunk_constants {b}" (Nice.chunkExp b (2 ^ 31) == e && b ^ e == d)
    r := r.check s!"chunk_constants_u16 {b}" (Nice.chunkExp b (2 ^ 16) == e16 && b ^ e16 == d16)
    let pf ← orFail (getNatOpt j "prefilter_digits")
    let start ← orFail (getNatOpt j "range_start")
    match pf, start with
    | some p, some s =>
      -- the Rust depth (float log, minus one for safety) never exceeds the exact depth
      r := r.check s!"prefilter_digits {b}" (p ≤ Nice.prefilterDepth b s)
    | some _, none => r := r.check s!"prefilter_digits {b} without a range" false
    | none, _ => pure ()
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
  for f in r.failures.reverse do
    IO.println s!"FAIL: {f}"
  IO.println s!"{r.checks} checks, {r.failures.length} failures"
  pure (if r.failures.isEmpty then 0 else 1)
