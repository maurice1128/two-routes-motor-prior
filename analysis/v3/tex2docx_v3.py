# -*- coding: utf-8 -*-
"""Convert wm_prior/tcds_merged/paper.tex (IEEEtran) to a single-column Word
file with python-docx. Numbers, tables and references are taken from the same
.tex/.aux/.bbl that made the PDF, so the two cannot drift."""
import re, io, os, sys
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = r"C:\Users\maurice\Desktop\robotic_research\wm_prior\tcds_v3"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SRC, "paper.docx")
tex = io.open(os.path.join(SRC, "paper.tex"), encoding="utf-8").read()
# the v3 tables II, IV and V are generated files pulled in with \input
tex = re.sub(r"\\input\{(tables/[a-z]+)\}",
             lambda m: io.open(os.path.join(SRC, m.group(1) + ".tex"), encoding="utf-8").read(), tex)
assert "\\input{" not in tex
# v3 cleanups the v2 converter did not need
tex = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}",
             "\n\nL_π = q·L_SAC + c·‖tanh μ_student(o) − tanh μ_teacher(o)‖²,   (1)\n\n", tex, flags=re.S)
tex = tex.replace("\\IEEEtriggeratref{18}", "")
tex = tex.replace("\\quad ", "   ").replace("^\\ddagger", "‡").replace("\\ddagger", "‡").replace("^{*\\ddagger}", "*‡")
aux = io.open(os.path.join(SRC, "paper.aux"), encoding="utf-8").read()
bbl = io.open(os.path.join(SRC, "paper.bbl"), encoding="utf-8").read()

# ---------- label and citation numbers from the .aux ----------
LAB = {}
for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{(.*?)\}\{\d+\}", aux):
    v = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", m.group(2)).strip()
    LAB[m.group(1)] = v
CITE = {m.group(1): m.group(2) for m in re.finditer(r"\\bibcite\{([^}]+)\}\{(\d+)\}", aux)}

# ---------- inline LaTeX -> runs ----------
GREEK = {"pi": "π", "mu": "μ", "theta": "θ", "alpha": "α", "tau": "τ", "gamma": "γ", "sigma": "σ", "tanh": "tanh"}


def math_to_text(s):
    s = s.replace("\\,", " ").replace("\\;", " ").replace("\\ ", " ").replace("~", " ")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathcal\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\texttt\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\big\\\|", "‖", s).replace("\\|", "‖")
    s = s.replace("^\\dagger", "†").replace("^*", "*")
    s = s.replace("\\{", "\x01").replace("\\}", "\x02")
    for k, v in GREEK.items():
        s = re.sub(r"\\" + k + r"(?![a-zA-Z])", v, s)
    reps = {"\\tanh": "tanh", "\\times": "×", "\\le": "≤", "\\ge": "≥", "\\to": "→", "\\pm": "±", "\\mapsto": "↦",
            "\\in": "∈", "\\dagger": "†", "\\emph": "", "\\big": "", "\\,": " ", "\\%": "%"}
    for k, v in reps.items():
        s = s.replace(k, v)
    s = re.sub(r"\^\{([^}]*)\}", r"^\1", s)
    s = re.sub(r"_\{([^}]*)\}", r"_\1", s)
    s = s.replace("{", "").replace("}", "").replace("\x01", "{").replace("\x02", "}")
    s = s.replace("--", "–")
    return s


def inline_runs(s):
    """Return list of (text, bold, italic, mono) from a LaTeX inline string."""
    s = s.replace("\n", " ")
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\s+", " ", s)
    # citations and refs
    s = re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + ", ".join(CITE.get(k.strip(), "?") for k in m.group(1).split(",")) + "]", s)
    s = re.sub(r"Sec\.~\\ref\{([^}]+)\}", lambda m: "Sec. " + LAB.get(m.group(1), "?"), s)
    s = re.sub(r"Table~\\ref\{([^}]+)\}", lambda m: "Table " + LAB.get(m.group(1), "?"), s)
    s = re.sub(r"Fig\.~\\ref\{([^}]+)\}", lambda m: "Fig. " + LAB.get(m.group(1), "?"), s)
    s = re.sub(r"\\ref\{([^}]+)\}", lambda m: LAB.get(m.group(1), "?"), s)
    s = re.sub(r"\\label\{[^}]+\}", "", s).strip()
    s = re.sub(r"\\IEEEPARstart\{(.)\}\{([^}]*)\}", r"\1\2", s)
    s = re.sub(r"\\ci\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}", r"\1 [\2, \3]", s)
    s = s.replace("``", "“").replace("''", "”").replace("---", "—").replace("--", "–")
    s = s.replace("{,}", ",").replace("\\%", "%").replace("\\&", "&").replace("\\_", "_").replace("~", "\u00a0")
    # tokenize styles: \textbf{}, \emph{}, \texttt{}, $...$
    out = []
    pat = re.compile(r"\\(textbf|emph|texttt)\{((?:[^{}]|\{[^{}]*\})*)\}|\$([^$]*)\$")
    pos = 0
    for m in pat.finditer(s):
        if m.start() > pos:
            out.append((clean(s[pos:m.start()]), False, False, False))
        if m.group(1):
            inner = m.group(2)
            inner = re.sub(r"\$([^$]*)\$", lambda k: math_to_text(k.group(1)), inner)
            style = m.group(1)
            out.append((clean(inner), style == "textbf", style == "emph", style == "texttt"))
        else:
            out.append((math_to_text(m.group(3)), False, False, False))
        pos = m.end()
    if pos < len(s):
        out.append((clean(s[pos:]), False, False, False))
    return [r for r in out if r[0]]


def clean(s):
    s = re.sub(r"\\texttt\{([^}]*)\}", r"\1", s)
    s = s.replace("\\dagger", "†").replace("\\,", " ")
    return s.replace("{", "").replace("}", "")


def add_runs(par, runs, size=None):
    for text, b, i, mono in runs:
        r = par.add_run(text)
        r.bold = b or None; r.italic = i or None
        if mono:
            r.font.name = "Consolas"; r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        if size:
            r.font.size = Pt(size)


def para(doc, latex, style=None, size=None, align=None, italic=False, space_after=6):
    p = doc.add_paragraph(style=style)
    runs = inline_runs(latex)
    if italic:
        runs = [(t, b, True, m) for t, b, i, m in runs]
    add_runs(p, runs, size)
    p.paragraph_format.space_after = Pt(space_after)
    if align: p.alignment = align
    return p


# ---------- document ----------
doc = docx.Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
for s in doc.sections:
    s.left_margin = s.right_margin = Inches(1); s.top_margin = s.bottom_margin = Inches(1)

title = re.search(r"\\title\{(.*?)\}\n\n", tex, re.S).group(1).replace("\\\\", " ")
title = re.sub(r"\s+", " ", title).strip()
p = doc.add_paragraph(); r = p.add_run(title); r.bold = True; r.font.size = Pt(16); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
author = re.search(r"\\author\{([^}]*)\}", tex).group(1).replace("~", " ").strip()
p = doc.add_paragraph(); r = p.add_run(author); r.font.size = Pt(12); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph(); r = p.add_run("Reading copy of the anonymized TCDS submission (the IEEEtran PDF is the submission file)"); r.italic = True; r.font.size = Pt(10); p.alignment = WD_ALIGN_PARAGRAPH.CENTER

abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
p = doc.add_paragraph(); r = p.add_run("Abstract—"); r.bold = True
add_runs(p, inline_runs(abstract)); p.paragraph_format.space_after = Pt(6)
kw = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", tex, re.S).group(1)
p = doc.add_paragraph(); r = p.add_run("Index Terms—"); r.bold = True
add_runs(p, [(re.sub(r"\s+", " ", kw.replace("--", "–")).strip(), False, True, False)])

body = tex[tex.index("\\section{Introduction}"):tex.index("\\bibliographystyle")]

# split floats out first
FLOAT = re.compile(r"\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}\n?", re.S)
floats = []
def stash(m):
    floats.append(m.group(0)); return "\n\n@@FLOAT%d@@\n\n" % (len(floats) - 1)
body = FLOAT.sub(stash, body)
body = re.sub(r"\\\[(.*?)\\\]", lambda m: "\n\n@@MATH@@" + m.group(1).replace("\n", " ") + "\n\n", body, flags=re.S)

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
sec_i = 0; sub_i = 0


def set_cell(cell, latex, bold_all=False, size=9, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    bold = "\\mathbf" in latex or "\\textbf" in latex or bold_all
    runs = inline_runs(latex)
    if bold: runs = [(t, True, i, m) for t, b, i, m in runs]
    add_runs(p, runs, size)
    if align: p.alignment = align
    p.paragraph_format.space_after = Pt(0)


def cell_border_bottom(cell):
    tcPr = cell._element.get_or_add_tcPr(); b = OxmlElement("w:tcBorders")
    e = OxmlElement("w:bottom"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000")
    b.append(e); tcPr.append(b)


def emit_table(doc, latex):
    cap = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", latex, re.S).group(1)
    lab = re.search(r"\\label\{([^}]+)\}", latex).group(1)
    num = LAB.get(lab, "?")
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("TABLE %s" % num); r.bold = True; r.font.size = Pt(9)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, inline_runs(cap), 9); p.paragraph_format.space_after = Pt(3)
    tab = re.search(r"\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabular\}", latex, re.S).group(1)
    tab = re.sub(r"\\(toprule|bottomrule|cmidrule\(lr\)\{[^}]*\})", "", tab)
    rows = []
    for raw in tab.split("\\\\"):
        line = raw.strip()
        if "\\midrule" in line:
            if rows: rows[-1]["rule"] = True
        line = line.replace("\\midrule", "").strip()
        rule = False
        if not line:
            continue
        cells = []
        for c in line.split("&"):
            c = c.strip()
            m = re.match(r"\\multicolumn\{(\d+)\}\{(?:[^{}]|\{[^{}]*\})*\}\{(.*)\}$", c, re.S)
            if m:
                cells.append((re.sub(r"\\(footnotesize|small|scriptsize)\s*", "", m.group(2)), int(m.group(1))))
            else:
                cells.append((c, 1))
        rows.append({"cells": cells, "rule": rule})
    ncol = max(sum(sp for _, sp in r["cells"]) for r in rows)
    fs = 8 if ncol >= 7 else 9
    t = doc.add_table(rows=len(rows), cols=ncol); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    if ncol >= 10:
        widths = [0.75, 0.8] + [0.5] * (ncol - 3) + [1.7]
    elif ncol == 7:
        widths = [1.3] + [0.55] * 4 + [1.5, 1.5]
    else:
        widths = None
    if widths:
        t.autofit = False
        for row_ in t.rows:
            for k, c in enumerate(row_.cells):
                if k < len(widths): c.width = Inches(widths[k])
    t.style = "Table Grid"
    # remove all borders, then add rules
    tblPr = t._element.tblPr; borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement("w:" + edge); e.set(qn("w:val"), "nil"); borders.append(e)
    tblPr.append(borders)
    for ri, row in enumerate(rows):
        ci = 0
        for text, span in row["cells"]:
            cell = t.cell(ri, ci)
            if span > 1:
                cell = cell.merge(t.cell(ri, ci + span - 1))
            first_col = (ci == 0)
            set_cell(cell, text, size=fs, align=(WD_ALIGN_PARAGRAPH.LEFT if first_col else WD_ALIGN_PARAGRAPH.CENTER))
            ci += span
        if ri == 0 or row["rule"]:
            for c in t.rows[ri].cells: cell_border_bottom(c)
    # top rule
    for c in t.rows[0].cells:
        tcPr = c._element.get_or_add_tcPr(); b = OxmlElement("w:tcBorders")
        e = OxmlElement("w:top"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "8"); e.set(qn("w:color"), "000000")
        b.append(e); tcPr.append(b)
    for c in t.rows[-1].cells: cell_border_bottom(c)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def emit_figure(doc, latex):
    for m in re.finditer(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", latex):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(os.path.join(SRC, m.group(1)), width=Inches(6.3))
        p.paragraph_format.space_after = Pt(2)
    cap = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", latex, re.S).group(1)
    lab = re.search(r"\\label\{([^}]+)\}", latex).group(1)
    p = doc.add_paragraph(); r = p.add_run("Fig. %s. " % LAB.get(lab, "?")); r.font.size = Pt(9); r.bold = True
    add_runs(p, inline_runs(cap), 9); p.paragraph_format.space_after = Pt(8)


pending_floats = []
for chunk in re.split(r"\n\s*\n", body):
    c = chunk.strip()
    if not c: continue
    m = re.match(r"\\section\{([^}]*)\}", c)
    if m:
        # flush floats before a new section
        for f in pending_floats:
            emit_table(doc, f) if "tabular" in f else emit_figure(doc, f)
        pending_floats = []
        sec_i += 1; sub_i = 0
        h = doc.add_heading("%s. %s" % (ROMAN[sec_i - 1], m.group(1).upper()), level=1)
        for r in h.runs: r.font.name = "Times New Roman"; r.font.color.rgb = RGBColor(0, 0, 0); r.font.size = Pt(12)
        rest = c[m.end():].strip()
        if rest:
            para(doc, rest)
        continue
    m = re.match(r"\\subsection\{([^}]*)\}", c)
    if m:
        sub_i += 1
        h = doc.add_heading("%s. %s" % (chr(64 + sub_i), m.group(1)), level=2)
        for r in h.runs: r.font.name = "Times New Roman"; r.font.color.rgb = RGBColor(0, 0, 0); r.font.size = Pt(11); r.italic = True
        rest = c[m.end():].strip()
        rest = re.sub(r"^\\label\{[^}]+\}\s*", "", rest)
        if rest:
            para(doc, rest)
        continue
    m = re.match(r"@@FLOAT(\d+)@@", c)
    if m:
        pending_floats.append(floats[int(m.group(1))]); continue
    if c.startswith("@@MATH@@"):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(math_to_text(c[8:]).strip().rstrip(",")); r.italic = True
        continue
    para(doc, c)
    # floats collected inside this subsection are emitted right after the first paragraph that follows them
    if pending_floats:
        for f in pending_floats:
            emit_table(doc, f) if "tabular" in f else emit_figure(doc, f)
        pending_floats = []
for f in pending_floats:
    emit_table(doc, f) if "tabular" in f else emit_figure(doc, f)

# ---------- references ----------
h = doc.add_heading("REFERENCES", level=1)
for r in h.runs: r.font.name = "Times New Roman"; r.font.color.rgb = RGBColor(0, 0, 0); r.font.size = Pt(12)
items = re.findall(r"\\bibitem\{([^}]+)\}\n(.*?)(?=\n\\bibitem|\n\\end\{thebibliography\})", bbl, re.S)
for key, txt in items:
    txt = re.sub(r"\\newblock", " ", txt); txt = re.sub(r"\\BIBentry\w+", "", txt)
    txt = re.sub(r"\\url\{([^}]*)\}", r"\1", txt); txt = re.sub(r"\\emph\{([^}]*)\}", r"\1", txt)
    txt = txt.replace("``", "“").replace("''", "”").replace("--", "–").replace("\\&", "&").replace("~", " ")
    txt = re.sub(r"\\hskip[^\\]*\\relax", " ", txt)
    txt = re.sub(r"\{\\'([a-zA-Z])\}", lambda m: {"s": "ś", "e": "é", "a": "á", "o": "ó"}.get(m.group(1), m.group(1)), txt)
    txt = re.sub(r"\{\\`([a-zA-Z])\}", lambda m: {"o": "ò", "e": "è", "a": "à"}.get(m.group(1), m.group(1)), txt)
    txt = re.sub(r'\{?\\"\{?([a-zA-Z])\}?\}?', lambda m: {"u": "ü", "o": "ö", "a": "ä", "O": "Ö"}.get(m.group(1), m.group(1)), txt)
    txt = txt.replace("{\\i}", "ı").replace("\\i ", "ı").replace("\\i", "ı").replace("{\\l}", "ł").replace("\\l", "ł")
    txt = txt.replace("{", "").replace("}", "")
    txt = re.sub(r"\s+", " ", txt).strip()
    p = doc.add_paragraph(); r = p.add_run("[%s] " % CITE.get(key, "?")); r.font.size = Pt(10)
    r = p.add_run(txt); r.font.size = Pt(10)
    p.paragraph_format.space_after = Pt(2); p.paragraph_format.left_indent = Inches(0.35); p.paragraph_format.first_line_indent = Inches(-0.35)

doc.save(OUT)
print("saved", OUT, "paragraphs", len(doc.paragraphs), "tables", len(doc.tables))
