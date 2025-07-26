from document_extractor import extract_sections_from_pdf
from embeddings import load_embedding_model, compute_embedding
from section_ranker import rank_sections
from output_generator import generate_output_json
import json

# Load persona and job-to-be-done
with open(r"sample_persona_job.json", "r") as f:
    persona_job_data = json.load(f)

persona_text = persona_job_data["persona"]
job_text = persona_job_data["job_to_be_done"]
documents = persona_job_data["documents"]

# Load model
model = load_embedding_model()

# Create persona-job embedding
persona_job_embedding = compute_embedding(model, persona_text + " " + job_text)

# Extract and process documents
all_sections = []
for pdf_path in documents:
    sections = extract_sections_from_pdf(pdf_path)
    all_sections.extend(sections)

# Rank sections
ranked_sections = rank_sections(all_sections, persona_job_embedding, model)

# Generate Output JSON
generate_output_json(documents, persona_text, job_text, ranked_sections)

print("Processing complete. Output written to sample_output.json")
