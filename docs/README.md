# docs

Explanatory notes for the simulation drivers in the repo root. Not required to
run anything.

| File | What it is |
| --- | --- |
| `agent-decision-making.md` | Source: how the Survivor agent selects a motive, locks a target, and steers toward it, with a worked example and the failure modes. |
| `agent-decision-making.pdf` | Rendered from the `.md`. |
| `build_pdf.py` | Regenerates the PDF from the Markdown. |

## Regenerating the PDF

The renderer deliberately keeps its dependencies out of `requirements-modern.txt`:

```bash
pip install fpdf2 markdown
python docs/build_pdf.py
```
