# Report: Choose-at-Most-One with a Fixed Activation Cost

## 1. Real-world background

The model describes a stylized **single-choice investment / project-selection problem with a shared setup cost**. An organization has a set of nine candidate options (indexed `A` through `I`), each of which would yield a known nonnegative benefit if adopted. Adopting any option requires incurring a one-time activation (fixed) cost that is paid once, regardless of which option is selected — for example, renting a facility, hiring a contractor, or launching a program that is a prerequisite for whichever alternative is ultimately pursued. Because the options are mutually exclusive (the organization can commit to at most one), the decision reduces to: *is any option worth pursuing, and if so, which one maximizes benefit minus the fixed cost?*

Because the benefit data are `A 9, B 12, C 11, D 5, E 8, F 4, G 10, H 7, I 6` and the activation cost is `6`, option `B` is the best single choice, yielding net benefit 12 − 6 = 6.

## 2. Original model structure

**Decision variables (10 independent variables).**
- `x {OPTIONS}` — binary indicators, one per option (`|OPTIONS| = 9`), equal to 1 if the option is chosen.
- `y` — a single continuous variable, unrestricted in sign, representing the level of activation actually incurred.

**Constraints (10 constraints).**
- `choose_at_most_one`: `sum {i in OPTIONS} x[i] <= 1`. This is the single coupling constraint across the nine binaries; together with `x binary` it enforces that at most one option is selected.
- `activate_if_chosen {i in OPTIONS}`: `y >= x[i]`, for each of the nine options. These are the fixed-charge linking constraints: if any option is chosen, `y` must be at least 1, i.e., the activation cost must be paid for.

**Objective (maximization).**
`maximize net_benefit: sum {i in OPTIONS} benefit[i] * x[i] - activation_cost * y`.

The objective is the total benefit of the chosen option minus the activation cost times the activation level. Since `y` appears with the strictly negative coefficient `-activation_cost` (`activation_cost = 6 > 0`) and has no upper bound, the optimizer always drives `y` to its smallest feasible value. Consequently, in the original formulation `y` is pinned to `max_i x[i]` at any optimal solution, and the objective is exactly "benefit of the single chosen option, minus the fixed cost if and only if something is chosen." No constraint forces `y` upward when nothing is chosen, which correctly avoids charging the fixed cost for an empty selection.

## 3. Reformulation and why it is valid

**Technique used: constraint aggregation (a problem-specific reduction exploiting integrality and a cardinality bound).**

The nine linking constraints `y >= x[i]` were replaced by the **single** aggregated constraint

```
subject to activation_link:
    y >= sum {i in OPTIONS} x[i];
```

This reduces the constraint count from 10 to 2. No variables, data, or objective terms were changed, and the variable set is identical to the original. The replacement is an *exact* equivalence of the feasible regions, not a relaxation:

- **(Original ⇒ aggregated).** Suppose `y >= x[i]` holds for every `i`. All `x[i]` are binary and `choose_at_most_one` states `sum_i x[i] <= 1`, so at most one of them equals 1. Hence `sum_i x[i] = max_i x[i]`, and `y >= max_i x[i] = sum_i x[i]`. The aggregated constraint is satisfied.
- **(Aggregated ⇒ original).** Suppose `y >= sum_i x[i]`. Since every `x[i]` is binary and therefore nonnegative, `x[i] <= sum_j x[j]` for each `i`. Chaining the inequalities gives `y >= sum_j x[j] >= x[i]` for every `i`, so each of the nine original linking constraints is satisfied.

Because the two feasible sets coincide for every value of the binaries (including the all-zero selection, where the aggregate reduces to `y >= 0` exactly as the nine originals do), the optimum value and the objective direction are unchanged, and every optimal solution of one model has a counterpart in the other with the same `x` and the same optimal `y`. The argument relies only on two structural facts visible in the model source: `x` is declared `binary` and `choose_at_most_one` bounds `sum {i in OPTIONS} x[i]` by 1. If either the binomial domain or the at-most-one row were removed, the aggregation would no longer be valid — this is precisely why the cardinality constraint must be retained, and it has been.

**Secondary effect and why it helps.** Replacing nine near-parallel rows by one row removes substantial redundancy from the LP relaxation: the nine rows were heavily degenerate with respect to one another (all bounded the same single continuous variable `y` from below by individual binaries). The aggregated row is a single, nonredundant cut separating "any selection" from "no selection" and inherits the cardinality structure directly, which typically yields a tighter, better-conditioned relaxation and requires less branching on `y`.

**What was deliberately not changed.** The variable set is untouched, so `x[...]` and `y` remain the independent decision variables of the reported solution, and the solution-mapping script is a pass-through that copies exactly the original `x[...]` and `y` entries and rejects non-finite or non-numeric values. Embedding the data section verbatim keeps the model self-contained and the instance data unchanged. One could go further — for example, substituting `y = sum {i in OPTIONS} x[i]` (or fixing `y` to the maximum selected option) as a preprocessing step, since optimality forces `y` to that value; this would eliminate `y` entirely and collapse the problem to picking the single best benefit. That step is *not* applied here, because it changes the reported variable set and would require a reconstruction rule for `y`; keeping `y` explicit preserves exact correspondence with the original model's interface at no cost in solution difficulty.
