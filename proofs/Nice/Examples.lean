/-
Worked examples, all by computation: sanity checks that the definitions say
what we think they say.
-/
import Nice.Spec.Nice
import Nice.Model.Residue

namespace Nice

/-- 69 is nice in base 10: 69² = 4761, 69³ = 328509. -/
theorem nice_69 : IsNice 10 69 := by decide

/-- 68 is not. -/
example : ¬ IsNice 10 68 := by decide

/-- Base 10's only nice number below 100 is 69. -/
example : (List.range 100).filter (fun n => decide (IsNice 10 n)) = [69] := by decide

/-- 69's residue passes the base-10 filter, as it must. -/
example : 69 % 9 ∈ residueFilter 10 := mem_residueFilter_of_isNice (by norm_num) nice_69

end Nice
