import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def relevance_feedback(vec_docs, vec_queries, sim, gt, n=10):
    # print(np.shape(vec_docs))
    # print(np.shape(sim))
    # print(np.shape(vec_queries))
    
    # print(vec_docs)
    # print(sim)
    # print(vec_queries)
    alpha = 0.4
    beta = 0.15
    for ep in range(3):
        for i in range(sim.shape[1]):
            ranked_documents = np.argsort(-sim[:, i])[:10]
            # print(ranked_documents)
            rel = 0
            nonrel = 0
            for j in gt:
                if j[0] == i+1:
                    if j[1]-1 in ranked_documents:
                        rel+=1
                    else:
                        nonrel+=1
            for j in gt:
                if j[0] == i+1:
                    if j[1]-1 in ranked_documents:
                        vec_queries[i] = vec_queries[i] + (alpha/rel)*(vec_docs[j[1]-1])
                    else:
                        vec_queries[i] = vec_queries[i] - (beta/nonrel)*(vec_docs[j[1]-1])
        sim = cosine_similarity(vec_docs, vec_queries)
    rf_sim = sim
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, gt, tfidf_model, n=10):
    alpha = 0.4
    beta = 0.15
    for ep in range(3):
        for i in range(sim.shape[1]):
            ranked_documents = np.argsort(-sim[:, i])[:10]
            rel = 0
            nonrel = 0
            for j in gt:
                if j[0] == i+1:
                    if j[1]-1 in ranked_documents:
                        rel+=1
                    else:
                        nonrel+=1
            for j in gt:
                if j[0] == i+1:
                    if j[1]-1 in ranked_documents:
                        vec_queries[i] = vec_queries[i] + (alpha/rel)*(vec_docs[j[1]-1])
                    else:
                        vec_queries[i] = vec_queries[i] - (beta/nonrel)*(vec_docs[j[1]-1])
            for j in gt:
                if j[0] == i+1:
                    a = (vec_docs[j[1] - 1].toarray()[0])
                    b = sorted(a, reverse= True)[:5]
                    c = np.argsort(-a)[:5]
                    l = np.zeros(np.shape(vec_queries[i][0]))
                    l[0][c] = a[c]
                    vec_queries[i] = vec_queries[i] + l
        sim = cosine_similarity(vec_docs, vec_queries)
    rf_sim = sim
    return rf_sim