"""
Docling-based reader for COMPLEX documents (pdf/docx/...) and TABLES (xlsx).

Customization per design:
  - OCR is DISABLED (do_ocr=False) — embedded/scan images are never force-converted
  - No VLM image description enrichment — images stay images
  - Table structure extraction is ON — tables become TablePart (markdown)
  - Embedded pictures are exported as PNG bytes → ImagePart (sent to vision LLM later)

The document is walked in order, producing TextPart / TablePart / ImagePart
in the sequence they appear in the document.
"""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from file_input.config import MAX_IMAGE_BYTES
from file_input.types import (
    FileKind,
    FilePart,
    FilePartError,
    ImagePart,
    ProcessedFile,
    TablePart,
    TextPart,
)


@lru_cache(maxsize=1)
def _get_converter():
    """One shared DocumentConverter configured to NOT OCR / NOT describe images."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False                # never OCR image content
    pipeline_options.do_table_structure = True     # parse tables as tables
    pipeline_options.generate_picture_images = True  # keep pictures as images
    pipeline_options.images_scale = 2.0

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _item_to_part(item, doc, source_name: str) -> FilePart | None:
    """Map one docling document item to our part types (order-preserving)."""
    from docling_core.types.doc import PictureItem, TableItem, TextItem

    if isinstance(item, TableItem):
        md = item.export_to_markdown(doc)
        if md.strip():
            return TablePart(source_name, md)
        return None

    if isinstance(item, PictureItem):
        pil_image = item.get_image(doc)
        if pil_image is None:
            return None
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        png = buf.getvalue()
        if len(png) > MAX_IMAGE_BYTES:
            # Don't fail the whole document for one huge figure; annotate inline.
            mb = len(png) / (1024 * 1024)
            return TextPart(source_name, f"[embedded image omitted — too large: {mb:.1f} MB]")
        return ImagePart(source_name, png, "image/png")

    if isinstance(item, TextItem):
        text = item.text.strip()
        if text:
            return TextPart(source_name, text)
        return None

    return None  # SectionItem / DocItemLabel decorations etc. — skip


def read_with_docling(path: str | Path, kind: FileKind) -> ProcessedFile:
    """Convert a complex/table file with docling into ordered parts."""
    p = Path(path)
    name = p.name

    if not p.is_file():
        return ProcessedFile(name, kind,
                             [FilePartError(name, f"file not found: {p}")])

    try:
        converter = _get_converter()
        result = converter.convert(str(p))
        doc = result.document
    except Exception as e:  # conversion failure → error part, batch continues
        return ProcessedFile(name, kind, [FilePartError(name, f"docling failed: {e}")])

    parts: list[FilePart] = []
    for item, _level in doc.iterate_items():
        part = _item_to_part(item, doc, name)
        if part is not None:
            parts.append(part)

    if not parts:
        parts = [FilePartError(name, "docling extracted no content "
                                        "(scanned doc without OCR yields images only)")]
    return ProcessedFile(name, kind, parts)
