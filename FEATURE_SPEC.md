# Feature specification v1

All inputs precede candidate Repair. No feature calls any solver or Repair. History features: **none**, uniformly across groups. X is the original 12-feature implementation in vendor/point_select/features.py. Route IDs are never inputs. Symmetric edges are unordered **multisets**: depot multiplicity is preserved, including singleton routes. Complexity below is per state/candidate (n=100, k=20); caching is optional, included in measured runtime.

G_diff (6; O(n²), available after reference generation):
1. Released current non-reference edge multiplicity / current edge multiplicity.
2. Length of those edges / current cost.
3. Released shared edge multiplicity / current edge multiplicity.
4. Length of those shared edges / current cost.
5. Reference edges missing in current solution incident to deletion / reference edge multiplicity.
6. XOR of current/reference customer co-route relation for unordered pairs incident to deletion / number of such pairs.

G_ready (6; O(n), actual original Destroy serialization): reference missing edges whose endpoints are both visible open fragment endpoints (depot always visible), different fragments and combined demand <=50; this is a **necessary-condition proxy**, not guaranteed reachability. Closing one open depot-anchored fragment to depot is permitted; joining the two ends of the same depot-free fragment is not. Complete retained routes and internal nodes are locked.
1. Endpoint-visible missing edge multiplicity / missing multiplicity.
2. Different-fragment (or depot closure) eligible multiplicity / missing multiplicity.
3. Capacity-eligible multiplicity / missing multiplicity.
4. Capacity-eligible missing edge length / reference cost.
5. Incomplete fragments incident to a capacity-eligible edge / incomplete fragments.
6. Missing edge multiplicity with a locked endpoint / missing multiplicity.
Zero denominator yields zero. Every occurrence of a depot edge counts. Fragment orientation has no semantic effect. New edge aggregates canonicalize route order/orientation, and use stable floating summation for the cost denominator. The original X implementation remains byte-identical; its permutation equivalence is checked to floating tolerance. These conditions do not model sequential future connections or policy probability.

G_current (12; O(n), no reference access): contact length/current cost, contact edge fraction, kept internal edge length/current cost, incomplete fragment count/101, complete retained route count/101, released singleton count/101, open customer endpoint count/101, mean/max/std incomplete fragment customer length/100, mean incomplete demand/capacity, fraction of unordered distinct incomplete fragment pairs with combined demand<=capacity. Depot-only tour excluded from fragment statistics. X already contains some related information; overlap is intentional.

B0=X (12), Bcap=X+G_current (24), Tdiff=X+G_diff (18), Tready=X+G_diff+G_ready (24), Tshuffle=X+permuted(G_diff+G_ready) (24). Tshuffle permutes the 16 native **slots within each state**, via independent fixed stream. It is a negative control, not a causal test. Duplicate sets retain slots; G vectors agree before shuffling, X may differ with centers.

Reference fallback: all reference groups use the frozen strongest legal no-reference B* scores for states with no feasible reference or reference cost >= current cost (strict comparison). Development selection is evaluated using the same state fallback. Teacher predictions remain stored separately. Tests never choose fallback using labels. All three reference outcomes are retained; main reference is the minimum recomputed feasible cost, tie by seed order. Alternate references are prespecified descriptive sensitivity only.

Preprocessing: fold-only imputation/weighted standardization. New Ridge uses raw mean z target, weights summing to 1, objective sum(w*(y-b-Xβ)²)+alpha*||β||², alpha=.01,.1,1; intercept unpenalized. This deliberately differs from historical sum-to-row-count regularization; alpha is not claimed equivalent. Same grid across dimensions, effective capacity still differs. GBDT uses original deterministic weighted CART, lr=.05, min_leaf=64, (depth,trees)=(1,32),(2,32),(2,64), <=4 leaves/tree. Six configurations/group, 120 OOF fits +5 final fits, no coverage fits or additional baseline fits.
