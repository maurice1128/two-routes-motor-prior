# -*- coding: utf-8 -*-
"""Words in the abstract (TCDS: 150-250) and per section of paper.tex, floats excluded."""
import re, io, os
s = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper.tex"), encoding="utf-8").read()


def words(t):
    t = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{(table|figure)\*?\}", " ", t, flags=re.S)
    t = re.sub(r"(?<!\\)%.*", "", t)
    t = re.sub(r"\\cite\{[^}]*\}|\\ref\{[^}]*\}|\\label\{[^}]*\}", "", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}$~\\^_]", " ", t)
    return len(t.split())


ab = s[s.index("\\begin{abstract}"):s.index("\\end{abstract}")]
print("abstract words:", words(ab))
body = s[s.index("\\section{Introduction}"):s.index("\\bibliographystyle")]
parts = re.split(r"(\\section\{[^}]*\}|\\subsection\{[^}]*\})", body)
tot = 0
for i in range(1, len(parts), 2):
    w = words(parts[i + 1]); tot += w
    print("%5d  %s" % (w, parts[i]))
print("%5d  total body" % tot)
