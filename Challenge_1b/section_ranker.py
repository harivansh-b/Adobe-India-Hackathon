from sklearn.metrics.pairwise import cosine_similarity


def rank_sections(sections, persona_job_embedding, model, similarity_threshold=0.40):
    ranked_sections = []

    for section in sections:
        section_emb = model.encode([section["text"]])[0]
        similarity = cosine_similarity(
            [persona_job_embedding],
            [section_emb]
        )[0][0]
        section["score"] = float(similarity)

        if similarity >= similarity_threshold:
            ranked_sections.append(section)  # Keep only if above threshold

    ranked = sorted(ranked_sections, key=lambda section: section["score"], reverse=True)

    return ranked
