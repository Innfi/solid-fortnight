#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pymupdf4llm


def convert_pdf_to_markdown(input_path: str, output_path: str | None = None) -> None:
    pdf_path = Path(input_path)

    if not pdf_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        print(f"Error: not a PDF file: {input_path}", file=sys.stderr)
        sys.exit(1)

    md_text = pymupdf4llm.to_markdown(str(pdf_path))

    if output_path:
        out = Path(output_path)
    else:
        out = pdf_path.with_suffix(".md")

    out.write_text(md_text, encoding="utf-8")
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF document to Markdown using pymupdf4llm"
    )
    parser.add_argument("input", help="Path to the input PDF file")
    parser.add_argument(
        "-o", "--output", help="Path to the output Markdown file (default: <input>.md)"
    )
    args = parser.parse_args()

    convert_pdf_to_markdown(args.input, args.output)


if __name__ == "__main__":
    main()
