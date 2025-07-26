import json
import time

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

    for rank, section in enumerate(ranked_sections[:10], start=1):
        output["extracted_sections"].append({
            "document": section["document"],
            "page_number": section["page_number"],
            "section_title": section["section_title"],
            "importance_rank": rank,
            "similarity_score": round(section["score"], 4)
        })

        output["sub_section_analysis"].append({
            "document": section["document"],
            "page_number": section["page_number"],
            "section_title": section["section_title"],
            "refined_text": section["text"][:500]
        })

    with open(r"sample_output.json", "w") as f:
        json.dump(output, f, indent=4)

    return output
