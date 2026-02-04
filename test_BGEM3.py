from FlagEmbedding import BGEM3FlagModel
import torch

if __name__ == "__main__":
    # 显式指定不使用多进程（你的句子很少，根本不需要多进程）
    # 这样可以完全避免进程池相关的清理问题
    model = BGEM3FlagModel(
        'BAAI/bge-m3',
        use_fp16=True,
        device='cuda',           # 显式指定 cuda（可选）
    )

    sentences_1 = [
        "What is BGE M3?",
        "Defination of BM25"
    ]
    sentences_2 = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"
    ]

    # 推荐写法：一次 encode 所有句子，更高效
    all_sentences = sentences_1 + sentences_2
    embeddings = model.encode(
        all_sentences,
        batch_size=12,
        max_length=8192,
        return_dense=True,
        return_sparse=False,            # 如果不需要 sparse 向量就关掉，节省内存
        return_colbert_vecs=False,
    )['dense_vecs']

    # 分割回两组
    emb1 = embeddings[:len(sentences_1)]
    emb2 = embeddings[len(sentences_1):]

    # 计算相似度矩阵
    similarity = emb1 @ emb2.T

    # 美化输出
    print("Similarity matrix:")
    print(similarity.round(4))

    # 如果你想要 cosine similarity（归一化后内积即 cosine）
    print("\nCosine similarities:")
    for i, s1 in enumerate(sentences_1):
        for j, s2 in enumerate(sentences_2):
            print(f"{s1[:30]:<30}  vs  {s2[:30]:<30}  →  {similarity[i,j]:.4f}")