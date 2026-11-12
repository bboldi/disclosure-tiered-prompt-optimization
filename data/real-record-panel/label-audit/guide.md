# Labelling guide: blind label audit

You label 120 rows in `audit_sheet.csv`. Each row is one installed inventory and one real CVE record. The rows come from two sources and are mixed in random order. The sheet does not show any row's reference label; which source a row comes from may be recognizable from how the inventory is written, and that is fine. Do not try to infer the reference label from it, and do not open the key (`~/phd_runs/rev1/label-audit-key/`), the benchmark or panel files, model outputs or result files while labelling.

## The question

**Does this CVE record say that any installed component, at its installed version, is affected?**

Use only what is in the row: the installed components (vendor, product, version, separated by `;`), the CNA description and the `affected` entries. Do not use outside knowledge of the product, other databases (NVD, vendor pages), or other rows.

## Answers

First decide each installed component the record covers: its status at the installed version is **affected**, **unaffected** or **undecided**, from the entries, status changes, default status and description (rules below). Then combine:

- **yes**: at least one covered component is affected (its installed version falls inside an affected range, matches an affected version, or is unlisted under a default status of affected).
- **no**: the record covers none of the installed products, or every covered component is unaffected: an explicit unaffected status, a default status of unaffected, or a description that bounds the affected versions. A version outside every listed range is **not** unaffected when the default status is unknown or missing and the description does not settle it; that component is undecided.
- **unsure**: no component is affected and at least one is undecided. Write in `note` what is missing (for example "range has no lower bound and the version scheme differs").

Decide a component when a careful reader would agree. Leave it undecided only when two careful readers could reasonably disagree; do not use it to avoid a hard call.

## How to read the entries

- Each line of `affected` is one product entry: `vendor / product: default status; status: version [up to, not including X | up to and including X] (version type)`. Lines starting `(ADP enrichment)` were added by a third party such as CISA, not by the CNA; read them, but where they contradict the CNA, the CNA wins.
- **Product match:** first decide which installed component, if any, the record is about. Inventory names may be written as identifiers (`apache_http_server`, lower case, underscores) or as display names; spelling, case, separators and vendor prefixes may differ (for example "Eclipse Jetty" and "jetty"). A different product of the same vendor, or a different edition or module, is **no** unless the record says that edition or module is included.
- **Ranges:** "affected: A up to, not including, B" covers A ≤ v < B; "up to and including B" covers A ≤ v ≤ B. A start of "0" means no lower bound, and an end of "*" means no upper bound. A start of "*" is not defined by the CVE format: treat it like a missing lower bound (below). **"unspecified", "n/a" or an empty value is not a bound but a missing one:** the entry does not say where the range starts or ends, so take that bound from the description if it gives one; if nothing in the record gives it, that component is **undecided**. A missing bound matters only when the answer depends on it: a version above the known end (or below the known start) is outside that range whatever the missing bound is.
- **Other fields** of an entry (platforms, modules, package name, CPE, program files) narrow what the entry covers. A platform or module the inventory does not mention does not make the answer no by itself; label by product and version, and note the restriction.
- **When entries and description disagree:** the more specific statement wins (an exact version or a bounded range over an open or missing bound, the CNA over ADP). If two equally specific statements contradict each other, that component is **undecided**; say which statements conflict in `note`.
- **Non-numeric versions** (letters, build numbers, "R30", "v2.001", dates, pre-releases such as "3.14.0a1"): compare them the way the vendor's scheme obviously reads, and say so in `note` when you had to interpret.
- **Status changes inside a range:** an entry such as "affected: 2.0 up to, not including, 3.0; changes: unaffected at 2.4.1" starts with its first status and switches at each listed version. A version in the range takes the status of the last change at or below it: in this example 2.3 is affected and 2.5 is unaffected.
- **Default status:** "default status: affected" means versions not listed are affected; "unaffected" means they are not; "unknown" or no default leaves unlisted versions undecided by the entries themselves; then use the description.
- **Prose conditions** in the description ("before 4.2", "all versions prior to the March 2024 update", "only when module X is enabled"): apply the version condition. Configuration conditions you cannot check from the inventory ("only when HTTP/3 is enabled") do **not** make it no: label by version and product, and write the condition in `note`.

## Mechanics

- Fill `label` with exactly `yes`, `no` or `unsure`; `note` is free text and optional for yes or no.
- Do not change any other column or the row order.
- Edit the file in a text editor, or import every column as text in a spreadsheet, so that no value is reformatted; the analysis rejects a sheet whose rows or record text changed.
- Label in one or two sittings; if you change a label on a second look, keep the final answer only.
- A second annotator works on an untouched copy of the sheet, without seeing the first annotator's answers.
