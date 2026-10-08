"""What the OCR/HTR pipeline cannot do yet, and where a person has to fetch scans.

One structured source of truth, used by the web app's OCR page
(``/api/ocr/status``) and rendered to ``docs/ocr-gaps.md``:

    python -m ingest.ocr_notes > docs/ocr-gaps.md

Edit the lists below, not the markdown. Counts for manual sources come from a
sample of 2,665 Latin VD17/VD18 records (see ``scripts/`` history, Oct 2026);
they show relative size, not totals.
"""
from __future__ import annotations

from typing import Any, Dict, List

# severity: "blocker" = a whole class of material is unreadable; "limit" = works but
# degraded or partial; "todo" = known work not yet done.
GAPS: List[Dict[str, Any]] = [
    {
        "id": "uncial-papyrus", "severity": "blocker",
        "title": "Uncial, rustic capitals and papyrus are unreadable",
        "detail": ("Every installed handwriting model scores ~30% known words on St Gall Cod. Sang. "
                   "226 (uncial on papyrus), against 75-85% on Carolingian and Gothic hands. "
                   "Tesseract fails too. The pipeline refuses to ingest such output (under 50%)."),
        "next": ("Fine-tune a model on 20-50 hand-corrected pages of the script (eScriptorium / "
                 "`ketos train`), or transcribe by hand. Late-antique and early-medieval material "
                 "(5th-8th c.) is mostly in this class."),
    },
    {
        "id": "translator-quality", "severity": "blocker",
        "title": "The translator invents content on damaged text",
        "detail": ("On the readable St Gall 195 transcription the Latin model produced fluent "
                   "errors (\"tulips\", \"second world war\"). Better transcription helps, but "
                   "the translator is the weaker link for HTR text."),
        "next": "Evaluate a stronger translator on doc 13402 (St Gall 195); keep the original beside any translation.",
    },
    {
        "id": "abbrev-latin-only", "severity": "limit",
        "title": "Abbreviation expansion is Latin-only and rule-based",
        "detail": ("`ingest/abbrev.py` resolves marks against the Latin corpus vocabulary. Old French, "
                   "German and Greek handwriting come out with abbreviations unexpanded, and some "
                   "Latin words stay wrong (\"deustuus\", split fragments like \"si onibus\")."),
        "next": "Per-language expansion, or use an expanding model (Manicule, Frolat `expan`) where it scores well.",
    },
    {
        "id": "untested-models", "severity": "todo",
        "title": "Greek, German-handwriting and Old French models are downloaded but not wired in",
        "detail": ("In `models/htr/`: greek_minuscule_s9-12_NFC, german_handwriting. Not yet "
                   "benchmarked or selectable; Gallicorpora+ (Old French) not downloaded. "
                   "Greek print goes only through Tesseract `grc`."),
        "next": "Add test sets to `scripts/htr_benchmark.py`, then add them to `ingest/htr.py` candidate lists.",
    },
    {
        "id": "walled-sources", "severity": "limit",
        "title": "Several major sources cannot be fetched by script",
        "detail": ("HathiTrust (Cloudflare), ISTC/CERL and Biblissima (bot check), USTC (login), "
                   "Gallica text (ALTCHA), SLUB Dresden and Halle (bot check). The project does not "
                   "defeat these checks."),
        "next": "Download by hand (see the manual-download list below) and ingest the PDF with `ocrimages`.",
    },
    {
        "id": "vd16", "severity": "limit",
        "title": "VD16 is not searchable",
        "detail": "The K10plus SRU endpoint serves VD17 and VD18 only.",
        "next": "Search vd16.de by hand and download the linked scan.",
    },
    {
        "id": "vd-hosts", "severity": "limit",
        "title": "Many VD17/VD18 copies sit on hosts we cannot read",
        "detail": ("In a 2,665-record sample, ~30% of records had a copy our connectors read "
                   "automatically, ~26% had no free link, and ~43% pointed at hosts listed "
                   "below (some now routed: Tübingen, Rostock, Weimar, Berlin)."),
        "next": "Add per-host manifest patterns where a host exposes IIIF (check with a live request first).",
    },
    {
        "id": "layout", "severity": "limit",
        "title": "No real layout analysis",
        "detail": ("Marginalia, interlinear and marginal glosses (glossed Bibles), multi-column pages "
                   "and music notation are read in whatever order the segmenter yields, and neumes "
                   "produce junk lines that are only filtered heuristically."),
        "next": "Region-aware segmentation with a layout model; keep gloss and text separate.",
    },
    {
        "id": "mixed-language", "severity": "limit",
        "title": "Mixed Latin/German books get one language label",
        "detail": ("The pipeline labels a book by its dominant language. German passages inside "
                   "Latin documents are routed per segment at translation time, but a book that is "
                   "mostly German is simply filed as German."),
        "next": "Per-segment language tags at ingest.",
    },
    {
        "id": "tesseract-long-s", "severity": "limit",
        "title": "Tesseract reads long s as f",
        "detail": ("Where Tesseract wins over the Kraken print models it still misreads ſ as f. The "
                   "existing long-s repair tools (`scripts/fix_long_s_ocr*.py`) are separate and not "
                   "run automatically on new ingests."),
        "next": "Run long-s correction in the connector when the engine is Tesseract.",
    },
    {
        "id": "old-ocr-audit", "severity": "todo",
        "title": "Older OCR-derived documents have not been quality-checked with the new metric",
        "detail": "The known-word rate (`abbrev.vocab_hit_rate`) has not been run over the existing library.",
        "next": "Scan the library, list documents under ~60% known words, queue repair or re-OCR.",
    },
    {
        "id": "no-correction-ui", "severity": "todo",
        "title": "No way to correct a transcription in the app",
        "detail": "HTR/OCR output can only be re-run, not edited; corrections are also the training data uncial needs.",
        "next": "A page-image + transcription side-by-side editor that saves corrected lines.",
    },
]

# automated: "no" = a person must fetch the scan; "manifest" = the IIIF manifest URL pattern
# is known (we route it) but an automated page download has not been confirmed end to end.
MANUAL_SOURCES: List[Dict[str, Any]] = [
    {"name": "HathiTrust", "host": "babel.hathitrust.org", "share": None,
     "why": "Cloudflare challenge on catalogue, API and page text.",
     "how": "Open the volume, use its Download / PDF option (public-domain volumes only), save the PDF.",
     "url": "https://babel.hathitrust.org/", "automated": "no"},
    {"name": "Gallica (BnF)", "host": "gallica.bnf.fr", "share": None,
     "why": "Every text and image-viewer endpoint is behind an ALTCHA bot check; the catalogue (SRU) works.",
     "how": "Find a work with the `gallica` connector, open its ark, use Télécharger -> PDF.",
     "url": "https://gallica.bnf.fr/", "automated": "no"},
    {"name": "ISTC (incunabula) / CERL", "host": "data.cerl.org", "share": None,
     "why": "Bot check. ISTC lists incunabula and links to digitized copies; it holds no text.",
     "how": "Search ISTC, follow the link to the holding library's scan, download the PDF.",
     "url": "https://data.cerl.org/istc/", "automated": "no"},
    {"name": "Biblissima portal", "host": "portail.biblissima.fr", "share": None,
     "why": "Bot check. It aggregates manuscripts from many French and European libraries.",
     "how": "Search, follow the link to the holding library; if it publishes IIIF, paste the manifest URL into the `iiif` source.",
     "url": "https://portail.biblissima.fr/", "automated": "no"},
    {"name": "USTC (Universal Short Title Catalogue)", "host": "ustc.ac.uk", "share": None,
     "why": "Login required; finding aid only.",
     "how": "Use it to identify editions, then fetch the digitized copy from the library it names.",
     "url": "https://www.ustc.ac.uk/", "automated": "no"},
    {"name": "SLUB Dresden", "host": "digital.slub-dresden.de", "share": 190,
     "why": "Bot check on the viewer and its manifests.",
     "how": "Open the link from the VD17/18 record, download the work as PDF from the viewer.",
     "url": "https://digital.slub-dresden.de/", "automated": "no"},
    {"name": "Halle (ULB Sachsen-Anhalt) / DOI links", "host": "vd17.bibliothek.uni-halle.de, dx.doi.org, opendata.uni-halle.de", "share": 124,
     "why": "Bot check at opendata.uni-halle.de; DOI links resolve there.",
     "how": "Follow the DOI, download the PDF.",
     "url": "https://opendata.uni-halle.de/", "automated": "no"},
    {"name": "URN resolvers (nbn-resolving.de / .org)", "host": "nbn-resolving.de, nbn-resolving.org", "share": 446,
     "why": "They redirect to whichever library holds the copy (Göttingen, Leipzig, Halle, Bavarian libraries...); the target is not predictable from the URN.",
     "how": "Open the URN; you land on the holding library's viewer. Use its PDF download. If the viewer is one we route (Heidelberg, Göttingen, MDZ), ingest that URL instead.",
     "url": "https://nbn-resolving.org/", "automated": "no"},
    {"name": "Herzog August Bibliothek Wolfenbüttel", "host": "diglib.hab.de", "share": 71,
     "why": "No IIIF manifest found at the URL patterns we tried.",
     "how": "Open the digital copy and use the viewer's download option.",
     "url": "http://diglib.hab.de/", "automated": "no"},
    {"name": "Universitätsbibliothek Freiburg", "host": "dl.ub.uni-freiburg.de", "share": 10,
     "why": "The Heidelberg-style manifest URL returns 404.",
     "how": "Download from the diglit viewer.", "url": "https://dl.ub.uni-freiburg.de/", "automated": "no"},
    {"name": "Hamburg, Darmstadt, Mainz, Stuttgart, Karlsruhe, Jena, BVB and others", "host": "various", "share": 40,
     "why": "Small hosts; no manifest pattern confirmed.",
     "how": "Open the record's link and use the viewer's PDF/image download.",
     "url": None, "automated": "no"},
    {"name": "Archive.org items with no text layer", "host": "archive.org", "share": None,
     "why": "About one item in five is image-only; the text connectors raise for them.",
     "how": "Download the PDF from the item page and ingest it with `ocrimages`.",
     "url": "https://archive.org/", "automated": "no"},
    {"name": "Universitätsbibliothek Tübingen (opendigi)", "host": "opendigi.ub.uni-tuebingen.de", "share": 49,
     "why": "Routed automatically since Oct 2026 (manifest pattern verified live).",
     "how": "Paste the viewer URL into Ingest by identifier (source: iiif).",
     "url": "https://opendigi.ub.uni-tuebingen.de/", "automated": "manifest"},
    {"name": "Rostock (rosdok), Weimar (HAAB), Berlin (SBB)", "host": "purl.uni-rostock.de, haab-digital.klassik-stiftung.de, *.staatsbibliothek-berlin.de", "share": 224,
     "why": "Routed automatically since Oct 2026 (manifest patterns verified live).",
     "how": "Paste the link into Ingest by identifier (source: iiif); if page download fails, fall back to PDF.",
     "url": None, "automated": "manifest"},
]


def render_markdown() -> str:
    out = ["# OCR / HTR: known gaps and manual-download sources", "",
           "_Generated from `ingest/ocr_notes.py` -- edit that file, then run "
           "`python -m ingest.ocr_notes > docs/ocr-gaps.md`._", "", "## Known gaps", ""]
    for g in GAPS:
        out += [f"### {g['title']}  _({g['severity']})_", "", g["detail"], "",
                f"**Next:** {g['next']}", ""]
    out += ["## Sources to download by hand", "",
            "Save the PDF (or images), then ingest it:", "",
            "```", 'python scripts/ingest.py ocrimages "path/to/book.pdf#pages=1-60"', "```", "",
            "or use *OCR → Ingest a downloaded file* in the web app. "
            "`share` = records in a 2,665-record VD17/VD18 sample.", "",
            "| Source | Why not automatic | How to get it | share |", "|---|---|---|---|"]
    for s in MANUAL_SOURCES:
        link = f"[{s['name']}]({s['url']})" if s.get("url") else s["name"]
        out.append(f"| {link} | {s['why']} | {s['how']} | {s['share'] if s['share'] is not None else ''} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(render_markdown())
