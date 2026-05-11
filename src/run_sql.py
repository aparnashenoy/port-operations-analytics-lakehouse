"""
DuckDB SQL showcase runner.

Executes any of the five SQL showcase files against the live Parquet files
and prints formatted results for each SELECT statement found in the file.
COPY … TO statements are skipped so no new Parquet files are written.

Usage:
    python src/run_sql.py sql/05_business_analysis_queries.sql
    python src/run_sql.py sql/03_data_quality_checks.sql --limit 10

Arguments:
    sql_file   Path to a .sql file in the sql/ directory.
    --limit    Maximum rows to display per result set (default: 20).
    --all      Run all five showcase files in order.
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SQL_FILES = [
    "sql/01_bronze_to_silver.sql",
    "sql/02_gold_terminal_kpis.sql",
    "sql/03_data_quality_checks.sql",
    "sql/04_ml_features.sql",
    "sql/05_business_analysis_queries.sql",
]

# Statements matching these patterns produce no rows and should be skipped.
_SKIP_RE = re.compile(
    r"^\s*(COPY\s|CREATE\s|INSERT\s|DROP\s|ALTER\s|EXPLAIN\s)",
    re.IGNORECASE,
)


def _split_statements(sql: str) -> list[str]:
    """
    Split a SQL string on semicolons, discarding blank and comment-only chunks.
    Does not attempt to parse quoted semicolons — these files do not contain them.
    """
    raw = sql.split(";")
    stmts = []
    for chunk in raw:
        # Strip leading/trailing whitespace and block comments
        stripped = chunk.strip()
        if not stripped:
            continue
        # Keep only non-trivial statements (ignore comment-only blocks)
        code_lines = [
            ln for ln in stripped.splitlines()
            if ln.strip() and not ln.strip().startswith("--")
        ]
        if code_lines:
            stmts.append(stripped)
    return stmts


def _is_query(stmt: str) -> bool:
    """Return True if the statement is expected to produce a result set."""
    # Find the first non-comment, non-blank line
    for line in stmt.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            return not _SKIP_RE.match(stripped)
    return False


def _extract_header_comment(stmt: str) -> str:
    """
    Extract a leading block of -- comment lines from a statement as a header.
    Returns the first non-empty comment block, or an empty string.
    """
    lines = []
    for line in stmt.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            lines.append(stripped.lstrip("- ").strip())
        elif stripped:
            break
    return " ".join(lines).strip()


def _format_table(rows: list[tuple], columns: list[str], limit: int) -> str:
    """Render a result set as a plain-text table with column headers."""
    if not rows:
        return "  (no rows)\n"

    display_rows = rows[:limit]
    col_widths = [len(c) for c in columns]
    for row in display_rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val) if val is not None else "NULL"))

    header = "  " + "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(columns))
    sep    = "  " + "  ".join("-" * w for w in col_widths)
    body   = "\n".join(
        "  " + "  ".join(
            (str(v) if v is not None else "NULL").ljust(col_widths[i])
            for i, v in enumerate(row)
        )
        for row in display_rows
    )
    footer = f"\n  ({len(rows)} rows{', showing first ' + str(limit) if len(rows) > limit else ''})\n"
    return f"{header}\n{sep}\n{body}{footer}"


def run_file(sql_path: Path, con: duckdb.DuckDBPyConnection, limit: int) -> None:
    """Execute one SQL file and print results for each SELECT statement."""
    print(f"\n{'=' * 72}")
    print(f"  {sql_path.name}")
    print(f"{'=' * 72}")

    sql = sql_path.read_text()
    statements = _split_statements(sql)

    query_count = 0
    for stmt in statements:
        if not _is_query(stmt):
            # Still execute non-SELECT statements (CREATE VIEW etc.) silently
            try:
                con.execute(stmt)
            except Exception as exc:  # noqa: BLE001
                print(f"\n  [skipped — {exc}]")
            continue

        query_count += 1
        header = _extract_header_comment(stmt)
        label  = textwrap.shorten(header, width=65, placeholder="…") if header else f"Query {query_count}"

        print(f"\n── {label}")
        try:
            result = con.execute(stmt)
            rows   = result.fetchall()
            cols   = [d[0] for d in result.description]
            print(_format_table(rows, cols, limit))
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}\n")

    if query_count == 0:
        print("\n  (no SELECT statements found)\n")


def build_connection() -> duckdb.DuckDBPyConnection:
    """Return an in-process DuckDB connection with the project root as CWD."""
    con = duckdb.connect(database=":memory:")
    # Execute from project root so relative Parquet paths in SQL files resolve
    con.execute(f"SET FILE_SEARCH_PATH = '{PROJECT_ROOT}'")
    return con


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DuckDB SQL showcase files against live Parquet data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sql_file",
        nargs="?",
        help="Path to a .sql file (relative or absolute). Omit when using --all.",
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Maximum rows to display per result set (default: 20).",
    )
    parser.add_argument(
        "--all", dest="run_all", action="store_true",
        help="Run all five showcase files in sequence.",
    )
    args = parser.parse_args()

    if not args.sql_file and not args.run_all:
        parser.print_help()
        print("\nAvailable showcase files:")
        for f in SQL_FILES:
            print(f"  {f}")
        sys.exit(0)

    files_to_run: list[Path] = []
    if args.run_all:
        files_to_run = [PROJECT_ROOT / f for f in SQL_FILES]
    else:
        p = Path(args.sql_file)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)
        files_to_run = [p]

    con = build_connection()

    for sql_path in files_to_run:
        run_file(sql_path, con, args.limit)

    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    main()
