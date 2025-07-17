from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def rank_sections(sections,persona_job_embedding,model):
    for section in sections:
        print(section)
        section_emb = model.encode([section["text"]])[0]
        similarity = cosine_similarity(
            [persona_job_embedding],
            [section_emb]
        )[0][0]
        section["score"] = float(similarity)
    ranked = sorted(sections,key=lambda section: section["score"],reverse=True)
    print(ranked)
    return ranked
