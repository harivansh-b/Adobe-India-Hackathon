from document_extractor import extract_sections_from_pdf
from embeddings import load_embedding_model, compute_embedding
from section_ranker import rank_sections
from output_generator import generate_output_json
import json
import time
start = time.time()
# Load input
with open("sample_persona_job.json", "r") as f:
    input_data = json.load(f)

# Extract fields from new format
persona_text = input_data["persona"]["role"]
job_text = input_data["job_to_be_done"]["task"]
documents_info = input_data["documents"]
documents = [doc["filename"] for doc in documents_info]  # Extract filenames only

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

# Generate output
generate_output_json(documents_info, persona_text, job_text, ranked_sections)

print("Processing complete. Output written to sample_output.json")
end = time.time()
print("Time elapsed: " + str(end - start))
