from medrag import MedRAG

# Örnek soru ve seçenekler
question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
}

# Retrieval-augmented mode (MedRAG)
medrag = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="Textbooks")

# Cevabı alalım. k parametresi, retrieval sırasında kaç snippet alınacağını belirler.
answer, snippets, scores = medrag.answer(question=question, options=options, k=5)

print("Cevap:", answer)
print("Snippet'ler:", snippets)
print("Skorlar:", scores)
