import json
import time

def clean_text(text):
    # Remove newlines, unwanted spaces, and preserve Unicode
    return text.replace("\n", " ").replace("\r", " ").strip()

def generate_output_json(documents, persona, job, ranked_sections):
    output = {
        "metadata": {
            "input_documents": documents,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": time.ctime()
        },
        "extracted_sections": [],
        "sub_section_analysis": []
    }

    for rank, section in enumerate(ranked_sections[:], start=1):
        cleaned_title = clean_text(section["section_title"])
        cleaned_text = clean_text(section["text"][:500])

        output["extracted_sections"].append({
            "document": section["document"],
            "page_number": section["page_number"],
            "section_title": cleaned_title,
            "importance_rank": rank,
            "similarity_score": round(section["score"], 4)
        })

        output["sub_section_analysis"].append({
            "document": section["document"],
            "page_number": section["page_number"],
            "section_title": cleaned_title,
            "refined_text": cleaned_text
        })

    with open("sample_output.json", "w", encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    return output
