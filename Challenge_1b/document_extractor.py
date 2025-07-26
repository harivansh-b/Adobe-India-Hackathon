import pymupdf

def extract_sections_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    sections = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text").strip()
        text_blocks = page.get_text("dict")["blocks"]

        section_title = None

        for block in text_blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["flags"] == 20 and len(span["text"].strip()) > 5:
                        section_title = span["text"].strip()
                        break
                if section_title:
                    break
            if section_title:
                break

        if not section_title:
            section_title = (page_text[:50] + "...") if page_text else f"Page {page_num + 1}"

        sections.append({
            "document": pdf_path.split("/")[-1],
            "page_number": page_num + 1,
            "text": page_text,
            "section_title": section_title
        })

    doc.close()
    return sections
