# Labelling guide: Real-Record Panel, hand-labelled stratum (G3.4)

You label 48 cells in `labelling_sheet.csv`. Each row is one installed component and one real CVE record that the deterministic matcher could not decide. Label each row before any model answer on this panel is seen, and do not open model outputs, run logs or result files while labelling.

## The question

**Does this CVE record say that the installed component, at exactly the installed version, is affected?**

Use only what is in the row: the installed component (vendor, product, version), the CNA description and the `affected` entries. Do not use outside knowledge of the product, other databases (NVD, vendor pages), or the other candidates of the sub-profile.

## Answers

- **yes**: the record covers this product, and the installed version falls inside an affected range or matches an affected version.
- **no**: the record does not cover this product, or the installed version falls outside every affected range, or the record marks it unaffected.
- **unsure**: the record text does not settle it. Write in `note` what is missing (for example "range has no lower bound and the version scheme differs").

Prefer yes or no when a careful reader would agree. Use unsure when two careful readers could reasonably disagree; do not use it to avoid a hard call.

## How to read the entries

- Each line of `affected` is one product entry: `vendor / product: default status; status: version [up to, not including X | up to and including X] (version type)`. Lines starting `(ADP enrichment)` were added by a third party such as CISA, not by the CNA; read them, but where they contradict the CNA, the CNA wins.
- **Product match:** decide first whether the entry is about the installed product. Spelling, case, separators and vendor prefixes may differ (for example "Eclipse Jetty" and "jetty"). A different product of the same vendor, or a different edition or module, is **no** unless the record says that edition or module is included.
- **Ranges:** "affected: A up to, not including, B" covers A ≤ v < B; "up to and including B" covers A ≤ v ≤ B. A start of "0" or "*" means no lower bound, and an end of "*" means no upper bound. **"unspecified", "n/a" or an empty value is not a bound but a missing one:** the entry does not say where the range starts or ends, so take that bound from the description if it gives one; if nothing in the record gives it, the row is **unsure**. A missing bound matters only when the answer depends on it: a version above the known end (or below the known start) is outside that range whatever the missing bound is.
- **Other fields** of an entry (platforms, modules, package name, CPE, program files) narrow what the entry covers. A platform or module the inventory does not mention does not make the answer no by itself; label by product and version, and note the restriction.
- **When entries and description disagree:** the more specific statement wins (an exact version or a bounded range over an open or missing bound, the CNA over ADP). If two equally specific statements contradict each other, the row is **unsure**; say which statements conflict in `note`.
- **Non-numeric versions** (letters, build numbers, "R30", "v2.001", dates, pre-releases such as "3.14.0a1"): compare them the way the vendor's scheme obviously reads, and say so in `note` when you had to interpret.
- **Default status:** "default status: affected" means versions not listed are affected; "unaffected" means they are not; "unknown" or no default leaves unlisted versions undecided by the entries themselves; then use the description.
- **Prose conditions** in the description ("before 4.2", "all versions prior to the March 2024 update", "only when module X is enabled"): apply the version condition. Configuration conditions you cannot check from the inventory ("only when HTTP/3 is enabled") do **not** make it no: label by version and product, and write the condition in `note`.

## Mechanics

- Fill `label` with exactly `yes`, `no` or `unsure`; `note` is free text and optional for yes or no.
- Do not change any other column or the row order.
- Label in one or two sittings; if you change a label on a second look, keep the final answer only.
