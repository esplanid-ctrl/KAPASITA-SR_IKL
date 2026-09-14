"""
Minimal, dependency-free SQL INSERT-VALUES parser.

Used ONLY for offline preprocessing of the cahyadsn/wilayah MySQL dump
files (db/wilayah.sql, db/wilayah_level_1_2.sql) into the canonical region
master artifact (see app.data.region_master.build_canonical_region_master).
This is NOT a general SQL engine -- it understands exactly the one shape
this dump uses: "INSERT INTO table(cols) VALUES (v1,v2,...),(...),...;"
(single or multi-row VALUES lists), with single-quoted strings and doubled
`''` as the only escape convention actually observed in this dump
(verified against the provided ZIP; no other escaping style was found).

Why not use a real SQL engine: no network access is available in this
environment to install sqlparse/mysqlclient, and the dump is MyISAM-
flavoured MySQL syntax that SQLite cannot load directly. This hand-rolled
tokenizer was verified to correctly parse all 552 rows of
wilayah_level_1_2.sql and can parse wilayah.sql's simpler 2-column rows
the same way.
"""
from __future__ import annotations

import re
from typing import List

_VALUES_BLOCK_RE = re.compile(r"VALUES\s*(\(.*?\));", re.DOTALL)


def _split_top_level_tuples(values_blob: str) -> List[str]:
    """Splits '(a,b),(c,d)' into ['(a,b)', '(c,d)'], respecting nested
    parens/brackets and single-quoted strings. Separator commas/whitespace
    BETWEEN tuples (at depth 0) are discarded, not accumulated."""
    tuples = []
    depth = 0
    in_string = False
    buf: List[str] = []
    i, n = 0, len(values_blob)
    while i < n:
        c = values_blob[i]
        if in_string:
            if c == "'" and i + 1 < n and values_blob[i + 1] == "'":
                buf.append("''")
                i += 2
                continue
            if c == "'":
                in_string = False
            buf.append(c)
            i += 1
            continue
        if c == "'":
            in_string = True
            buf.append(c)
            i += 1
            continue
        if c in "([":
            depth += 1
            buf.append(c)
            i += 1
            continue
        if c in ")]":
            depth -= 1
            buf.append(c)
            i += 1
            if depth == 0:
                tuples.append("".join(buf))
                buf = []
            continue
        if depth == 0:
            # between tuples (comma, whitespace, newline) -- discard
            i += 1
            continue
        buf.append(c)
        i += 1
    return tuples


def _split_fields(tuple_str: str) -> List[str]:
    """tuple_str includes the surrounding parens, e.g. "(a,'b,c',[1,2])"."""
    if not (tuple_str.startswith("(") and tuple_str.endswith(")")):
        raise ValueError(f"Not a well-formed tuple: {tuple_str[:30]!r}...")
    inner = tuple_str[1:-1]
    fields: List[str] = []
    depth = 0
    in_string = False
    buf: List[str] = []
    i, n = 0, len(inner)
    while i < n:
        c = inner[i]
        if in_string:
            if c == "'" and i + 1 < n and inner[i + 1] == "'":
                buf.append("'")
                i += 2
                continue
            if c == "'":
                in_string = False
                i += 1
                continue
            buf.append(c)
            i += 1
            continue
        if c == "'":
            in_string = True
            i += 1
            continue
        if c in "([":
            depth += 1
            buf.append(c)
            i += 1
            continue
        if c in ")]":
            depth -= 1
            buf.append(c)
            i += 1
            continue
        if c == "," and depth == 0:
            fields.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    fields.append("".join(buf))
    return fields


def parse_insert_rows(sql_text: str, expected_field_count: int) -> List[List[str]]:
    """Parses every `INSERT ... VALUES (...);` statement in `sql_text` into
    a list of field-value lists (all values as raw strings; numeric/typed
    conversion is the caller's job). Raises if a row doesn't have exactly
    `expected_field_count` fields -- silent truncation would corrupt the
    region master, which this module refuses to do.
    """
    rows = []
    for value_block in _VALUES_BLOCK_RE.findall(sql_text):
        for t in _split_top_level_tuples(value_block):
            fields = _split_fields(t)
            if len(fields) != expected_field_count:
                raise ValueError(
                    f"Expected {expected_field_count} fields, got {len(fields)} "
                    f"in tuple starting {t[:60]!r}"
                )
            rows.append(fields)
    return rows
