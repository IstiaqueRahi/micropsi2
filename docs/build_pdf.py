#!/usr/bin/env python3
"""Render docs/agent-decision-making.md to docs/agent-decision-making.pdf.

Standalone helper, not part of the simulation. Dependencies are intentionally
kept out of requirements-modern.txt:

    pip install fpdf2 markdown
    python docs/build_pdf.py
"""
import os
import re
import markdown
from fpdf import FPDF
from fpdf.fonts import TextStyle

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "agent-decision-making.md")
OUT = os.path.join(HERE, "agent-decision-making.pdf")


def main():
    with open(SRC, encoding="utf-8") as f:
        md_text = f.read()

    html = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )

    # fpdf2's write_html cannot render inline tags inside table cells; flatten
    # <code>/<strong>/<em> to plain text within <td>/<th> only.
    def _flatten_cell(m):
        inner = re.sub(r"</?(?:code|strong|em|b|i)>", "", m.group(2))
        return f"<{m.group(1)}>{inner}</{m.group(1)}>"

    html = re.sub(r"<(td|th)>(.*?)</\1>", _flatten_cell, html, flags=re.S)

    pdf = FPDF(format="A4")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_font("helvetica", size=10.5)

    # keep code blocks and tables from running off the page
    pdf.write_html(
        html,
        table_line_separators=True,
        tag_styles={
            "h1": TextStyle("helvetica", "B", font_size_pt=19, t_margin=2, b_margin=4),
            "h2": TextStyle("helvetica", "B", font_size_pt=14, t_margin=6, b_margin=3),
            "h3": TextStyle("helvetica", "B", font_size_pt=11.5, t_margin=4, b_margin=2),
            "pre": TextStyle("courier", font_size_pt=8),
            "code": TextStyle("courier", font_size_pt=8),
        },
    )

    pdf.output(OUT)
    print("wrote", OUT, "(%d bytes)" % os.path.getsize(OUT))


if __name__ == "__main__":
    main()
