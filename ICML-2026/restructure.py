#!/usr/bin/env python3
"""
One-shot restructuring of main.tex:
- Cut subsections 4.5 to 4.14 from main body
- Add compensating paragraphs to Discussion (sec 5)
- Move Limitations / Reproducibility / Ethics to \section*{} after Conclusion
- Add appendix scaffolding with full-width + manual TOC
- Re-insert moved subsections as appendix sections B-K
"""
import re
from pathlib import Path

P = Path(__file__).parent / "main.tex"
text = P.read_text()

# --- Anchor markers for splitting ---
# We split on exact section/subsection lines we know exist.
ANCHOR_DISJOINT = "\\subsection{Disjoint Prompt Validation}"
ANCHOR_DISCUSSION = "\\section{Discussion}"
ANCHOR_LIMITATIONS_PARA = "\\paragraph{Limitations.}"
ANCHOR_CONCLUSION = "\\section{Conclusion}"
ANCHOR_BIB = "\\bibliography{references}"
ANCHOR_APPENDIX_LLM = "\\section{Statement on LLM Usage}"
ANCHOR_END = "\\end{document}"

assert text.count(ANCHOR_DISJOINT) == 1
assert text.count(ANCHOR_DISCUSSION) == 1
assert text.count(ANCHOR_CONCLUSION) == 1

# Locate split positions
i_disjoint = text.index(ANCHOR_DISJOINT)
i_discussion = text.index(ANCHOR_DISCUSSION)
i_limitations = text.index(ANCHOR_LIMITATIONS_PARA)
i_conclusion = text.index(ANCHOR_CONCLUSION)
i_bib = text.index(ANCHOR_BIB)
i_appendix_llm = text.index(ANCHOR_APPENDIX_LLM)

# Slice the document
preamble_and_body_to_4_5 = text[:i_disjoint]            # everything before sec 4.5
moved_subsections_block = text[i_disjoint:i_discussion]  # sec 4.5 ... 4.14 (will be moved)
discussion_block_full   = text[i_discussion:i_conclusion]  # \section{Discussion} ... before \section{Conclusion}
conclusion_block        = text[i_conclusion:i_bib]      # \section{Conclusion} ... before bibliography
bib_block               = text[i_bib:i_appendix_llm]    # \bibliography{...} ... before \section{Statement...}
existing_appendix_llm   = text[i_appendix_llm:]         # \section{Statement on LLM Usage} ... \end{document}

# --- Split discussion block: pre-Limitations and the Limitations paragraph ---
discussion_pre_lim_idx = discussion_block_full.index(ANCHOR_LIMITATIONS_PARA)
discussion_pre_limitations = discussion_block_full[:discussion_pre_lim_idx]
limitations_paragraph = discussion_block_full[discussion_pre_lim_idx:]

# --- Pull out the inline Reproducibility/Ethics from Conclusion ---
# These are after the conclusion main paragraph; we'll detach and rewrap as \section*.
# Find their starts:
repro_marker = "\\noindent\\textbf{Reproducibility.}"
ethics_marker = "\\noindent\\textbf{Ethics.}"
i_repro = conclusion_block.index(repro_marker)
i_ethics = conclusion_block.index(ethics_marker)

conclusion_main = conclusion_block[:i_repro].rstrip() + "\n\n"
repro_paragraph = conclusion_block[i_repro:i_ethics].strip()
ethics_paragraph = conclusion_block[i_ethics:].strip()

# --- Rebuild conclusion: drop the inline Reproducibility/Ethics; they become \section*{} after Conclusion ---
new_conclusion = conclusion_main

# --- Assemble \section*{Limitations}, \section*{Reproducibility}, \section*{Ethical Considerations} ---
# Promote \paragraph{Limitations.} to a section* with its body intact
lim_body = limitations_paragraph.replace("\\paragraph{Limitations.}", "", 1).strip()
limitations_section = f"\\section*{{Limitations}}\n\n{lim_body}\n\n"

repro_body = repro_paragraph.replace(repro_marker, "", 1).strip()
repro_section = f"\\section*{{Reproducibility}}\n\n{repro_body}\n\n"

ethics_body = ethics_paragraph.replace(ethics_marker, "", 1).strip()
ethics_section = f"\\section*{{Ethical Considerations}}\n\n{ethics_body}\n\n"

# --- Compensating paragraphs to slot into Discussion (after Threat Model, before Design Principles box) ---
# Find the spot: insert just before \paragraph{Design Principles ...}
design_principles_anchor = "\\paragraph{Design Principles (revised in light of the mechanistic data).}"
assert design_principles_anchor in discussion_pre_limitations

compensating_block = r"""\paragraph{Generalization beyond the training prompts.}
The dual attack reaches 54--88\% ASR on 50 held-out prompts on 10 of 13 models with clean baselines (Appendix~\ref{app:disjoint}); ablation specificity is confirmed by 15 random unit-vector controls per model, with the refusal direction yielding 3.9--32.7$\times$ higher ASR than random directions on 10 of 13 models (Appendix~\ref{app:randdir}). Ablation does not corrupt benign generation (0/100 incoherent outputs across all evaluated scales; Appendix~\ref{app:utility}).

\paragraph{Two probes for two functions; transfer to GCG.}
A linear compliance-prediction probe trained at every layer achieves best-AUC at the separability peak on 8/9 tested models, with ctrl-peak AUC 1--27 pts lower (Appendix~\ref{app:defense}). The same sep-peak probe transfers to GCG~\citep{zou2023universal}: on three models, GCG-successful prompts have refusal projections 39--104 units below GCG-failed prompts, far above within-class variance (Appendix~\ref{app:gcg}). Compared to Llama Guard 3 (input classifier, bypassed by assistant prefill) and Circuit Breakers \citep{zou2024circuit} (output corruption with $\sim$83--90\% incoherent responses), the sep-peak probe is the cheapest cross-attack detector (Appendix~\ref{app:defbase}).

\paragraph{Distributed rebuild, not a single relay.}
Residual-stream activation patching across three model families (Llama-3.1-8B, Qwen2.5-7B, Mistral-Small-24B) shows recovery climbs smoothly from ctrl peak to sep peak with no localized relay: every late layer contributes incrementally and no single mid-layer patch exceeds 65\% recovery on Llama-3.1-8B (Appendix~\ref{app:pathpatch}). The refusal signal at the final layer is reconstituted from many orthogonal sources rather than relayed by one, consistent with the output-readout interpretation.

\paragraph{The attack surface is trainable away.}
LoRA fine-tuning (200 steps, $r$=16) with refusal-direction-ablation counterexamples drives ablation-ASR to 0\% across five tested models on held-out prompts; full dual-judge re-evaluation (2000 responses) confirms 0\% COMPLIANCE on every condition (Appendix~\ref{app:advtrain}). However, principled XSTest evaluation \citep{rottger2024xstest} shows the same adapters over-refuse safe prompts at 85--99\%, an 80--98 pp exaggerated-safety cost. The ctrl-peak attack surface is therefore a property of the alignment recipe, not the architecture, but the recipe needs a benign-preservation component before deployment.

\paragraph{New failure modes in reasoning models.}
Reasoning-trained models (QwQ-32B, DS-R1-Distill-Llama-70B) leak harmful content inside their \texttt{<think>} block before emitting a final refusal---\textbf{chain-of-thought leakage}---a failure mode orthogonal to the gap (Appendix~\ref{app:cotleak}). On QwQ-32B, sockpuppeting \emph{increases} the final-layer refusal-token preference (the model ``recognises'' the attack at output) yet produces detailed harmful reasoning earlier in the trace. Output-level safety is therefore an incomplete objective for reasoning models. Judge sensitivity (comply-then-refuse pattern, weak baselines, CoT leakage) is documented in Appendix~\ref{app:judges}.

"""

discussion_pre_limitations_with_comp = discussion_pre_limitations.replace(
    design_principles_anchor,
    compensating_block + design_principles_anchor,
    1,
)

# Strip the now-stale "\paragraph{Limitations.}..." paragraph -- it's already excluded from the pre-Limitations slice.
new_discussion = discussion_pre_limitations_with_comp

# --- Build appendix block ---
# We renumber moved subsections as \section{...} inside \appendix.
# Extract each subsection block from moved_subsections_block by splitting on \subsection{...}
# preserving the markers themselves.
subsections_pattern = re.compile(r"(?=\\subsection\{)")
parts = subsections_pattern.split(moved_subsections_block)
parts = [p for p in parts if p.strip()]
assert all(p.startswith("\\subsection{") for p in parts), "Unexpected leading content"

# Map old subsection title to (new section title, label key)
title_to_label = [
    ("Disjoint Prompt Validation", "app:disjoint"),
    ("Random Direction Control", "app:randdir"),
    ("Utility Evaluation", "app:utility"),
    ("Defense Probes: Intervene at Ctrl, Detect at Sep", "app:defense"),
    ("Defense-Baseline Comparison", "app:defbase"),
    ("GCG Generalization: Sep-Peak Detection Survives Learned Suffixes", "app:gcg"),
    ("Path Patching: The Relay is Distributed, Not Localized", "app:pathpatch"),
    ("Adversarial Training Closes the Gap (Positive Contribution)", "app:advtrain"),
    ("Chain-of-Thought Leakage in Reasoning Models", "app:cotleak"),
    ("Judge Sensitivity and Borderline Responses", "app:judges"),
]
assert len(parts) == len(title_to_label), f"got {len(parts)} parts, expected {len(title_to_label)}"

# Convert each \subsection{X} block into \section{X}\label{newlabel}
# Also keep the existing \label{sec:...} so cross-refs in the body still work.
def to_appendix_section(block, new_label):
    # Replace \subsection{...} with \section{...}, preserving original \label if any.
    block = re.sub(r"\\subsection\{", r"\\section{", block, count=1)
    # Insert new label right after \section line if not already labeled.
    # Find first newline after the opening \section{...}
    first_nl = block.index("\n")
    sec_line = block[: first_nl]
    rest = block[first_nl + 1:]
    # The original \label{sec:...} (if present) is already on the next line; keep it.
    # Add new app label on the line after the section.
    if "\\label{" in rest.split("\n", 2)[0]:
        # Original label exists; just insert app label too
        first_label_line, _, after = rest.partition("\n")
        new_block = f"{sec_line}\n{first_label_line}\n\\label{{{new_label}}}\n{after}"
    else:
        new_block = f"{sec_line}\n\\label{{{new_label}}}\n{rest}"
    return new_block

new_appendix_sections = []
for block, (_, lbl) in zip(parts, title_to_label):
    new_appendix_sections.append(to_appendix_section(block, lbl))

appendix_sections_str = "".join(new_appendix_sections)

# --- Add Appendix Table of Contents (manual, with \pageref) ---
toc_rows = [("A", "Statement on LLM Usage", "app:llm")]
for (title, lbl) in title_to_label:
    # Use a slightly compressed title for the TOC where original is very long
    compact = {
        "Defense Probes: Intervene at Ctrl, Detect at Sep": "Defense Probes",
        "GCG Generalization: Sep-Peak Detection Survives Learned Suffixes": "GCG Generalization",
        "Path Patching: The Relay is Distributed, Not Localized": "Path Patching",
        "Adversarial Training Closes the Gap (Positive Contribution)": "Adversarial Training",
        "Chain-of-Thought Leakage in Reasoning Models": "Chain-of-Thought Leakage in Reasoning Models",
        "Judge Sensitivity and Borderline Responses": "Judge Sensitivity",
    }
    toc_rows.append((None, compact.get(title, title), lbl))

# Letter sequence
letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O"]
toc_lines = []
for idx, (letter_override, title, lbl) in enumerate(toc_rows):
    letter = letter_override if letter_override else letters[idx]
    toc_lines.append(f"\\textbf{{{letter}}} & {title} \\dotfill & \\pageref{{{lbl}}} \\\\")
toc_table = "\n".join(toc_lines)

appendix_header = f"""\\onecolumn
\\setlength{{\\parskip}}{{0pt}}

{{\\Large\\bfseries Appendix Table of Contents}}\\par
\\vspace{{0.5em}}
\\noindent
\\renewcommand{{\\arraystretch}}{{1.45}}
\\begin{{tabular*}}{{\\textwidth}}{{@{{}}p{{0.05\\textwidth}}@{{\\hspace{{0.5em}}}}p{{0.80\\textwidth}}@{{\\extracolsep{{\\fill}}}}r@{{}}}}
{toc_table}
\\end{{tabular*}}
\\vspace{{1.5em}}

"""

# Add label to LLM Usage
existing_appendix_llm_labeled = existing_appendix_llm.replace(
    "\\section{Statement on LLM Usage}\n\\label{sec:llm_usage}",
    "\\section{Statement on LLM Usage}\n\\label{app:llm}\n\\label{sec:llm_usage}",
    1,
)
# In case the label form is slightly different, fall back to inserting after \section line
if existing_appendix_llm_labeled == existing_appendix_llm:
    existing_appendix_llm_labeled = existing_appendix_llm.replace(
        "\\section{Statement on LLM Usage}",
        "\\section{Statement on LLM Usage}\n\\label{app:llm}",
        1,
    )

# --- Assemble new document ---
# Find \appendix in existing_appendix_llm_labeled
# Actually \appendix is *before* "\section{Statement on LLM Usage}" — let me check.
# It's in the slice between bib_block and existing_appendix_llm? Or in existing_appendix_llm?
# Original layout: \bibliography...\bibliographystyle...\appendix\n\section{Statement on LLM Usage}
# bib_block runs from \bibliography{} to before \section{Statement on LLM Usage}
# So \appendix is in bib_block.
assert "\\appendix" in bib_block
# Split bib_block at \appendix
bib_pre_appendix, _, bib_after_appendix = bib_block.partition("\\appendix")
assert bib_after_appendix.strip() == ""  # \appendix should be at end of slice (or just whitespace after)

# Build new document
new_text = (
    preamble_and_body_to_4_5  # preamble + body up through end of §4.4
    + new_discussion           # §5 Discussion (pre-Limitations) + new compensating paras + Design Principles
    + new_conclusion           # §6 Conclusion (without Reproducibility/Ethics)
    + limitations_section      # \section*{Limitations}
    + repro_section            # \section*{Reproducibility}
    + ethics_section           # \section*{Ethical Considerations}
    + bib_pre_appendix         # \bibliography + \bibliographystyle
    + "\\appendix\n"
    + appendix_header          # \onecolumn + TOC
    + existing_appendix_llm_labeled.replace("\\end{document}", "")  # Statement on LLM Usage (now App A)
    + appendix_sections_str    # Apps B-K
    + "\n\\end{document}\n"
)

P.write_text(new_text)
print(f"Wrote restructured main.tex ({len(new_text)} chars, was {len(text)})")
print(f"Number of appendix sections moved: {len(parts)}")
