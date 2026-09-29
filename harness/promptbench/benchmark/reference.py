"""Separate raw-CPE label checker; intentionally imports no production oracle helpers.

Agreement checks implementation consistency, not the correctness of NVD annotations.
"""

from __future__ import annotations

import itertools
import re
from typing import Any


def compare(left: str, right: str) -> int:
    values = []
    for value in (left, right):
        if not isinstance(value, str) or len(value) > 100:
            raise ValueError("unsupported reference version")
        pieces = value.split(".")
        if any(
            not p.isascii() or not p.isdecimal() or (len(p) > 1 and p[0] == "0") for p in pieces
        ):
            raise ValueError("unsupported reference version")
        values.append([int(p) for p in pieces])
    for a, b in itertools.zip_longest(*values, fillvalue=0):
        if a != b:
            return 1 if a > b else -1
    return 0


def names(criteria: str) -> list[str]:
    token = r"(?:\\.|[^:\\])+"
    if re.fullmatch((token + ":") * 12 + token, criteria) is None:
        raise ValueError("invalid reference CPE")
    fields = [re.sub(r"\\(.)", r"\1", item) for item in re.findall(token, criteria)]
    if fields[:3] != ["cpe", "2.3", "a"] or fields[6:] != ["*"] * 7:
        raise ValueError("unsupported reference CPE")
    if re.search(r"\\[?*]", criteria) or any(
        not item or item == "-" or "*" in item or "?" in item for item in fields[3:5]
    ):
        raise ValueError("unsupported reference product")
    return fields


def check(raw: dict[str, Any], components: list[dict[str, str]]) -> bool:
    decisions = []
    configurations = raw.get("configurations")
    if not configurations:
        raise ValueError("missing reference configurations")
    for config in configurations:
        if config.get("operator", "OR") != "OR" or config.get("negate", False) is not False:
            raise ValueError("unsupported reference configuration")
        nodes = config["nodes"]
        if not nodes or ("operator" not in config and len(nodes) != 1):
            raise ValueError("ambiguous reference nodes")
        for node in nodes:
            if node.get("operator") != "OR" or node.get("negate", False) is not False:
                raise ValueError("unsupported reference node")
            if "children" in node or "nodes" in node or not node.get("cpeMatch"):
                raise ValueError("unsupported reference nesting")
            for match in node["cpeMatch"]:
                if match.get("vulnerable") is not True:
                    raise ValueError("unsupported reference prerequisite")
                fields = names(match["criteria"])
                exact = fields[5]
                bounds = {
                    k: match[k]
                    for k in (
                        "versionStartIncluding",
                        "versionStartExcluding",
                        "versionEndIncluding",
                        "versionEndExcluding",
                    )
                    if k in match
                }
                if exact != "*" and bounds:
                    raise ValueError("reference exact/range conflict")
                for component in components:
                    product_match = component.get("part", "a") == "a" and (
                        component["vendor"].casefold(),
                        component["product"].casefold(),
                    ) == (fields[3].casefold(), fields[4].casefold())
                    version = component["version"]
                    predicates = []
                    if exact != "*":
                        predicates.append(compare(version, exact) == 0)
                    for name, bound in bounds.items():
                        comparison = compare(version, bound)
                        predicates.append(
                            {
                                "versionStartIncluding": comparison >= 0,
                                "versionStartExcluding": comparison > 0,
                                "versionEndIncluding": comparison <= 0,
                                "versionEndExcluding": comparison < 0,
                            }[name]
                        )
                    decisions.append(product_match and all(predicates))
    return any(decisions)
