"""Build the nano-dates technical report as a short academic-style PDF (serif, paper layout)."""
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, HRFlowable, KeepTogether)

ASSETS = sys.argv[1]
OUT = sys.argv[2]

# ---- embed real Times New Roman (base-14 Times has no 'ć' glyph) -------------
F = "/System/Library/Fonts/Supplemental/"
pdfmetrics.registerFont(TTFont("TNR", F + "Times New Roman.ttf"))
pdfmetrics.registerFont(TTFont("TNR-B", F + "Times New Roman Bold.ttf"))
pdfmetrics.registerFont(TTFont("TNR-I", F + "Times New Roman Italic.ttf"))
pdfmetrics.registerFont(TTFont("TNR-BI", F + "Times New Roman Bold Italic.ttf"))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")
MONO = "Courier"

INK = colors.HexColor("#1a1a1a"); MUTE = colors.HexColor("#5f5b55")
RULE = colors.HexColor("#d8d5cf"); LIGHT = colors.HexColor("#f5f4f1")
BLUE = colors.HexColor("#1d4ed8"); GREEN = colors.HexColor("#15803d")
AMBER = colors.HexColor("#b45309"); RED = colors.HexColor("#b91c1c")

TITLE = ParagraphStyle("TITLE", fontName="TNR-B", fontSize=19, leading=23, textColor=INK,
                       alignment=TA_LEFT, spaceAfter=4)
BYLINE = ParagraphStyle("BYLINE", fontName="TNR", fontSize=11.5, leading=15, textColor=INK, spaceAfter=2)
ATTRIB = ParagraphStyle("ATTRIB", fontName="TNR", fontSize=9.8, leading=13.5, textColor=INK, spaceAfter=2)
H2 = ParagraphStyle("H2", fontName="TNR-B", fontSize=12.5, leading=15, textColor=INK,
                    spaceBefore=12, spaceAfter=4)
ABS = ParagraphStyle("ABS", fontName="TNR", fontSize=10.2, leading=14.5, textColor=INK,
                     alignment=TA_JUSTIFY, spaceAfter=6, leftIndent=2, rightIndent=2)
BODY = ParagraphStyle("BODY", fontName="TNR", fontSize=10.4, leading=14.8, textColor=INK,
                      alignment=TA_JUSTIFY, spaceAfter=6)
CAP = ParagraphStyle("CAP", fontName="TNR", fontSize=8.8, leading=11.8, textColor=MUTE,
                     alignment=TA_LEFT, spaceBefore=3, spaceAfter=8)
CODE = ParagraphStyle("CODE", fontName=MONO, fontSize=8.6, leading=12, textColor=INK,
                      backColor=LIGHT, leftIndent=7, rightIndent=7, borderPadding=6,
                      spaceBefore=3, spaceAfter=8)
QUOTE = ParagraphStyle("QUOTE", fontName="TNR-I", fontSize=10.6, leading=14.5, textColor=INK,
                       leftIndent=14, rightIndent=10, spaceBefore=4, spaceAfter=9)
NOTE = ParagraphStyle("NOTE", fontName="TNR-I", fontSize=8.8, leading=12.2, textColor=MUTE,
                      alignment=TA_JUSTIFY, spaceAfter=3)
REF = ParagraphStyle("REF", fontName="TNR", fontSize=8.8, leading=12, textColor=INK, spaceAfter=2)

USABLE = letter[0] - 2 * 0.92 * inch

def fig(path, frac=1.0, cap=None):
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    w = USABLE * frac
    items = [Image(path, width=w, height=w * ih / iw)]
    if cap:
        items.append(Paragraph(cap, CAP))
    return KeepTogether(items)

def deco(canvas, doc):
    canvas.saveState()
    canvas.setFont("TNR", 8.5)
    canvas.setFillColor(MUTE)
    canvas.drawCentredString(letter[0] / 2, 0.5 * inch, str(canvas.getPageNumber()))
    canvas.restoreState()

s = []
s.append(Paragraph("Teaching a One-Million-Parameter Model to Read Dates", TITLE))
s.append(Paragraph("Free synthetic data, an honest capability map, and a clean capacity ceiling", BYLINE))
s.append(Spacer(1, 5))
s.append(Paragraph("A Technical Report by <font name='TNR-B'>Vuk Rosi&#263;</font>", BYLINE))
s.append(Paragraph(
    "Subject: <b>nano-dates</b>, an open-source (MIT) 1M-parameter model that converts natural date "
    "phrases into ISO-8601. Weights: huggingface.co/vukrosic/nano-dates &nbsp;&middot;&nbsp; "
    "Code: github.com/vukrosic/nano-dates. June 2026.", ATTRIB))
s.append(Spacer(1, 4))
s.append(HRFlowable(width="100%", thickness=1, color=RULE, spaceAfter=8))

# Abstract
s.append(Paragraph("<b>Abstract.</b> "
    "nano-dates is a 1,016,960-parameter byte-level transformer that turns a natural date phrase into an "
    "ISO-8601 date &mdash; for example, &ldquo;the 3rd of July 2025&rdquo; becomes 2025-07-03. It runs on a "
    "CPU in milliseconds and trains from scratch in about thirty seconds on one GPU. The point of the "
    "project is not the model but the <i>method</i>: every training example is produced by code rather than "
    "scraped or labelled. Because the task is narrow and formally specified, you can sample the answer first "
    "and then render it in many natural surface forms, so the label is correct by construction &mdash; free, "
    "unlimited, and exact. This report describes the recipe, a data-leak bug that made an earlier model look "
    "perfect while it was quietly cheating, and an honest map of exactly where a model this small stops being "
    "able to reason. It is a method demonstration and a capability study, not a production date library.", ABS))
s.append(Spacer(1, 4))

# Figure 1 (headline result, page 1)
s.append(fig(f"{ASSETS}/fig_accuracy_print.png", 1.0,
    "<b>Figure 1.</b> Held-out exact-match accuracy on 2,000 unseen examples (greedy decode), overall "
    "<b>85.4%</b>. Green = solved; amber = partial (variable-count arithmetic); red = the capacity "
    "ceiling. Read the bottom two bars: resolving &ldquo;next friday&rdquo; is the one thing the model "
    "cannot do, and it is the most computationally entangled item in the set."))

# Section 1
s.append(Paragraph("1.&nbsp;&nbsp;Why generate the training data with code", H2))
s.append(Paragraph(
    "A 1M-parameter model cannot be a general assistant, but it can completely nail a task that is narrow and "
    "formally specified. Date&rarr;ISO is exactly that, and that is what makes the data free. The key move is "
    "to <b>sample the answer first</b>: pick a date, then emit it as &ldquo;June 12, 2023&rdquo;, "
    "&ldquo;Jun 12 2023&rdquo;, &ldquo;the 12th of June 2023&rdquo;, or &mdash; for relative phrases &mdash; "
    "pick a reference date and emit &ldquo;tomorrow&rdquo; or &ldquo;in 3 weeks&rdquo;. Because the answer "
    "is where you started, the label is correct by construction.", BODY))
s.append(fig(f"{ASSETS}/fig_recipe_print.png", 1.0,
    "<b>Figure 2.</b> The recipe. Start from the answer date, render it many ways, and the label is simply "
    "the date you began with. No verification, no annotation cost, no distillation from a larger model."))
s.append(Paragraph(
    "This is strictly better than asking a large model to produce training data: there is nothing to verify "
    "(the label <i>is</i> the ground truth you generated), it costs nothing and is unlimited, and you control "
    "the exact distribution of forms and difficulty. The model is handed a reference date (today) at the "
    "start of every prompt, so relative phrases are computable from the input alone &mdash; it never needs a "
    "wall clock. The prompt format, byte for byte, is:", BODY))
s.append(Paragraph("&lt;reference ISO date&gt; | &lt;phrase&gt; =&gt; &lt;answer ISO date&gt;", CODE))
s.append(Paragraph(
    "Seventeen surface renderers, a byte-level tokenizer (vocabulary 256, so every character is one token and "
    "there is no vocabulary file), and prompt-masked supervised fine-tuning in which only the ten-character "
    "ISO answer is scored &mdash; never the phrase the model was handed.", BODY))

# Section 2
s.append(Paragraph("2.&nbsp;&nbsp;A model that looked perfect by cheating", H2))
s.append(Paragraph(
    "The first trained model scored <b>100% on every absolute date form</b>, and I almost shipped it as a "
    "date normalizer. Then I typed an example by hand with a deliberate mismatch, and it failed instantly:", BODY))
s.append(Paragraph("2024-03-10 | Jun 12 2023 =&gt; 2024-03-10 &nbsp;&nbsp;&larr; wrong: it echoed the reference date", CODE))
s.append(Paragraph(
    "The model was not parsing &ldquo;Jun 12 2023&rdquo; at all &mdash; it was <b>copying the reference date</b> "
    "from the front of the prompt. The cause was in the data generator: for absolute forms I had set the "
    "reference date <i>equal to</i> the answer, so during training the answer was always sitting right there "
    "at the start of the prompt, and the model learned the obvious shortcut. It scored 100% on a held-out set "
    "that shared the same leak. The fix is one line of intent: for absolute phrases, draw the reference date "
    "<b>independently</b> of the answer, so the only way to be right is to actually parse the phrase. That single "
    "change also lifted variable-count arithmetic from 32&ndash;45% to 77&ndash;81% &mdash; forcing real "
    "computation everywhere helped.", BODY))
s.append(Paragraph(
    "&ldquo;Correct by construction&rdquo; does not mean &ldquo;not leaking the answer through a side "
    "channel.&rdquo; Your evaluation will happily confirm a model that learned your bug; the only thing that "
    "caught this one was feeding it an input the generator could never produce.", QUOTE))

# Section 3
s.append(Paragraph("3.&nbsp;&nbsp;Where a one-million-parameter model actually breaks", H2))
s.append(Paragraph(
    "After the fix, Figure 1 is the honest capability map. Everything the model can do, it does reliably; the "
    "interesting part is the bottom row. Resolving a weekday phrase means mapping an arbitrary date to its "
    "day-of-week and then doing modular-7 arithmetic to find the next matching day. That day-of-week function "
    "is the most globally entangled computation in the whole task &mdash; it depends on the full date, including "
    "century and leap-year structure &mdash; and a 1M-parameter model simply does not have the capacity to "
    "represent it. Adding training did not move the number: it is a capacity ceiling, not undertraining, and it "
    "is a clean, legible one.", BODY))
rows = [
    ["Capability", "Examples", "Accuracy"],
    ["Parse absolute dates", "June 12, 2023  ·  the 12th of June 2023", "100%"],
    ["Resolve simple relatives", "today  ·  tomorrow  ·  next/last week  ·  next month  ·  in N months", "98-100%"],
    ["Variable-N day/week math", "in N days  ·  N days ago  ·  in N weeks", "77-81%"],
    ["Weekday resolution", "next friday  ·  last tuesday", "~12%"],
    ["Overall", "mixed held-out set", "85.4%"],
]
t = Table(rows, colWidths=[1.55*inch, 3.65*inch, 1.0*inch])
t.setStyle(TableStyle([
    ("FONTNAME", (0,0), (-1,0), "TNR-B"), ("FONTNAME", (0,1), (-1,-1), "TNR"),
    ("FONTNAME", (0,-1), (-1,-1), "TNR-B"),
    ("BACKGROUND", (0,0), (-1,0), INK), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTSIZE", (0,0), (-1,-1), 8.7),
    ("TEXTCOLOR", (2,1), (2,2), GREEN), ("TEXTCOLOR", (2,3), (2,3), AMBER),
    ("TEXTCOLOR", (2,4), (2,4), RED), ("FONTNAME", (2,1), (2,-1), "TNR-B"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
    ("LINEBELOW", (0,-2), (-1,-2), 0.6, RULE),
    ("BOX", (0,0), (-1,-1), 0.5, RULE), ("LINEBELOW", (0,0), (-1,0), 0.5, INK),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 4.5), ("BOTTOMPADDING", (0,0), (-1,-1), 4.5),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
]))
s.append(KeepTogether([t, Paragraph(
    "<b>Table 1.</b> The same numbers as Figure 1, with example phrases. The honest framing: nano-dates parses "
    "absolute dates and resolves most relative offsets; weekday arithmetic is where it stops. That is a far more "
    "useful and trustworthy thing to publish than &ldquo;100% accurate date parser.&rdquo;", CAP)]))

# Section 4
s.append(Paragraph("4.&nbsp;&nbsp;Model and training at a glance", H2))
mrows = [
    ["Parameters", "1,016,960"],
    ["Architecture", "decoder-only transformer, pre-norm: RMSNorm, RoPE, grouped-query attention, SwiGLU"],
    ["Tokenizer", "raw UTF-8 bytes (vocabulary 256, no vocabulary file)"],
    ["Width / depth / heads", "dim 128 / 4 layers / 4 heads (2 KV heads), context 64 bytes"],
    ["Training", "prompt-masked SFT, 12k steps, batch 64, AdamW, cosine LR 3e-3"],
    ["Data", "100,000 code-generated pairs, 17 surface renderers"],
    ["Final validation loss", "0.036"],
]
mt = Table(mrows, colWidths=[1.7*inch, 4.5*inch])
mt.setStyle(TableStyle([
    ("FONTNAME", (0,0), (0,-1), "TNR-B"), ("FONTNAME", (1,0), (1,-1), "TNR"),
    ("FONTSIZE", (0,0), (-1,-1), 8.8),
    ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, LIGHT]),
    ("BOX", (0,0), (-1,-1), 0.5, RULE), ("INNERGRID", (0,0), (-1,-1), 0.4, RULE),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
]))
s.append(KeepTogether([mt, Spacer(1, 6)]))
s.append(Paragraph(
    "Inference needs only <font name='Courier'>torch</font> and <font name='Courier'>safetensors</font> and a "
    "single self-contained model file. The full training, evaluation, and data-generation code &mdash; with "
    "tests that check the labels are correct and the answer is not leaked &mdash; reproduces the 85.4% number "
    "bit-for-bit and is in the public repository.", BODY))

# References
s.append(Paragraph("References", H2))
s.append(Paragraph("[1] V. Rosi&#263;. <i>nano-dates</i> &mdash; model weights and self-contained inference. "
                   "Hugging Face model repository, vukrosic/nano-dates, 2026.", REF))
s.append(Paragraph("[2] V. Rosi&#263;. <i>nano-dates</i> &mdash; training, evaluation, and data-generation code. "
                   "GitHub repository, vukrosic/nano-dates, 2026. MIT license.", REF))
s.append(Spacer(1, 8))
s.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceAfter=5))
s.append(Paragraph(
    "<b>Author&rsquo;s note.</b> This is an independent technical report by Vuk Rosi&#263;, June 2026, on his "
    "own open-source project. It is a method demonstration and a capability study, not a production date "
    "library &mdash; for real software, libraries such as dateutil and chrono are exact and free. The value "
    "here is showing how far a nano-scale model gets on a formal task trained entirely on synthetic data, and "
    "drawing a legible line at where its reasoning breaks. The figures are original to this report. Any "
    "interpretation errors are the author&rsquo;s own.", NOTE))

SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.92*inch, rightMargin=0.92*inch,
                  topMargin=0.8*inch, bottomMargin=0.8*inch,
                  title="nano-dates: A Technical Report", author="Vuk Rosic").build(
    s, onFirstPage=deco, onLaterPages=deco)
print("wrote", OUT)
