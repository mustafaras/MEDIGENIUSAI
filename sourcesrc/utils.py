from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
import os
import faiss
import json
import torch
import tqdm
import numpy as np

corpus_names = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "MedText": ["textbooks", "statpearls"],
    "MedCorp": ["pubmed", "textbooks", "statpearls", "wikipedia"],
}

retriever_names = {
    "BM25": ["bm25"],
    "Contriever": ["facebook/contriever"],
    "SPECTER": ["allenai/specter"],
    "MedCPT": ["pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"],
    "RRF-2": ["bm25", "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"],
    "RRF-4": ["bm25", "facebook/contriever", "allenai/specter", "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"]
}

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

class CustomizeSentenceTransformer(SentenceTransformer):
    """
    Change the default pooling from "MEAN" to "CLS".
    """
    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        print("No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format(model_name_or_path))
        token = kwargs.get('token', None)
        cache_folder = kwargs.get('cache_folder', None)
        revision = kwargs.get('revision', None)
        trust_remote_code = kwargs.get('trust_remote_code', False)
        if 'token' in kwargs or 'cache_folder' in kwargs or 'revision' in kwargs or 'trust_remote_code' in kwargs:
            transformer_model = Transformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
                tokenizer_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
            )
        else:
            transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]

def embed(chunk_dir, index_dir, model_name, **kwarg):
    save_dir = os.path.join(index_dir, "embedding")

    if "contriever" in model_name:
        model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = CustomizeSentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    fnames = sorted([fname for fname in os.listdir(chunk_dir) if fname.endswith(".jsonl")])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for fname in tqdm.tqdm(fnames):
            fpath = os.path.join(chunk_dir, fname)
            save_path = os.path.join(save_dir, fname.replace(".jsonl", ".npy"))
            if os.path.exists(save_path):
                continue
            if open(fpath).read().strip() == "":
                continue
            print(f"Reading file: {fpath}")
            texts = []
            for item in open(fpath).read().strip().split('\n'):
                try:
                    texts.append(json.loads(item))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in file: {fpath}")
                    continue
            if "specter" in model_name.lower():
                texts = [model.tokenizer.sep_token.join([item["title"], item["content"]]) for item in texts]
            elif "contriever" in model_name.lower():
                texts = [". ".join([item["title"], item["content"]]).replace('..', '.').replace("?.", "?") for item in texts]
            elif "medcpt" in model_name.lower():
                texts = [[item["title"], item["content"]] for item in texts]
            else:
                texts = [concat(item["title"], item["content"]) for item in texts]

            if not texts:  # Skip if no valid texts were found
                print(f"No valid texts found in file: {fpath}, skipping embedding.")
                continue

            embed_chunks = model.encode(texts, **kwarg)
            np.save(save_path, embed_chunks)

        embed_chunks = model.encode([""], **kwarg)
    return embed_chunks.shape[-1]

def construct_index(index_dir, model_name, h_dim=768, HNSW=False, M=32):
    metadata_path = os.path.join(index_dir, "metadatas.jsonl")
    with open(metadata_path, 'w') as f:
        f.write("")

    if HNSW:
        if "specter" in model_name.lower():
            index = faiss.IndexHNSWFlat(h_dim, M)
        else:
            index = faiss.IndexHNSWFlat(h_dim, M)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        if "specter" in model_name.lower():
            index = faiss.IndexFlatL2(h_dim)
        else:
            index = faiss.IndexFlatIP(h_dim)

    embedding_dir = os.path.join(index_dir, "embedding")
    for fname in tqdm.tqdm(sorted(os.listdir(embedding_dir))):
        curr_embed_path = os.path.join(embedding_dir, fname)
        curr_embed = np.load(curr_embed_path)

        # Log the shape of the embedding for debugging
        print(f"Processing file: {fname}, embedding shape: {curr_embed.shape}")

        # Skip empty or malformed embeddings
        if curr_embed.size == 0 or len(curr_embed.shape) != 2:
            print(f"Skipping invalid or empty embedding file: {fname}")
            continue

        index.add(curr_embed)
        with open(metadata_path, 'a+') as f:
            for i in range(len(curr_embed)):
                metadata_entry = {'index': i, 'source': fname.replace(".npy", "")}
                f.write(json.dumps(metadata_entry) + '\n')
                # Log each metadata entry being written
                print(f"Writing metadata entry: {metadata_entry}")

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    return index

class Retriever:
    def __init__(self, retriever_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", corpus_name="textbooks", db_dir="./corpus", HNSW=False, **kwarg):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        self.chunk_dir = os.path.join(self.db_dir, self.corpus_name, "chunk")
        if not os.path.exists(self.chunk_dir):
            print("Cloning the {:s} corpus from Huggingface...".format(self.corpus_name))
            os.system("git clone https://huggingface.co/datasets/MedRAG/{:s} {:s}".format(corpus_name, os.path.join(self.db_dir, self.corpus_name)))
            if self.corpus_name == "statpearls":
                print("Downloading the statpearls corpus from NCBI bookshelf...")
                os.system("wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {:s}".format(os.path.join(self.db_dir, self.corpus_name)))
                os.system("tar -xzvf {:s} -C {:s}".format(os.path.join(self.db_dir, self.corpus_name, "statpearls_NBK430685.tar.gz"), os.path.join(self.db_dir, self.corpus_name)))
                print("Chunking the statpearls corpus...")
                os.system("python src/data/statpearls.py")
        self.index_dir = os.path.join(self.db_dir, self.corpus_name, "index", self.retriever_name.replace("Query-Encoder", "Article-Encoder"))
        if "bm25" in self.retriever_name.lower():
            from pyserini.search.lucene import LuceneSearcher
            self.metadatas = None
            self.embedding_function = None
            if os.path.exists(self.index_dir):
                self.index = LuceneSearcher(os.path.join(self.index_dir))
            else:
                os.system("python -m pyserini.index.lucene --collection JsonCollection --input {:s} --index {:s} --generator DefaultLuceneDocumentGenerator --threads 16".format(self.chunk_dir, self.index_dir))
                self.index = LuceneSearcher(os.path.join(self.index_dir))
        else:
            if os.path.exists(os.path.join(self.index_dir, "faiss.index")):
                self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
                self.metadatas = []
                metadata_path = os.path.join(self.index_dir, "metadatas.jsonl")
                with open(metadata_path, 'r') as f:
                    for line in f:
                        try:
                            self.metadatas.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"Skipping invalid metadata line: {line.strip()}")
                # Log the number of loaded metadata entries
                print(f"Loaded {len(self.metadatas)} metadata entries from {metadata_path}")
            else:
                print("[In progress] Embedding the {:s} corpus with the {:s} retriever...".format(self.corpus_name, self.retriever_name.replace("Query-Encoder", "Article-Encoder")))
                if self.corpus_name in ["textbooks", "pubmed", "wikipedia"] and self.retriever_name in ["allenai/specter", "facebook/contriever", "ncbi/MedCPT-Query-Encoder"] and not os.path.exists(os.path.join(self.index_dir, "embedding")):
                    print("[In progress] Downloading the {:s} embeddings given by the {:s} model...".format(self.corpus_name, self.retriever_name.replace("Query-Encoder", "Article-Encoder")))
                    os.makedirs(self.index_dir, exist_ok=True)
                    if self.corpus_name == "textbooks":
                        if self.retriever_name == "allenai/specter":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EYRRpJbNDyBOmfzCOqfQzrsBwUX0_UT8-j_geDPcVXFnig?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "facebook/contriever":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EQqzldVMCCVIpiFV4goC7qEBSkl8kj5lQHtNq8DvHJdAfw?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "ncbi/MedCPT-Query-Encoder":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EQ8uXe4RiqJJm0Tmnx7fUUkBKKvTwhu9AqecPA3ULUxUqQ?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                    elif self.corpus_name == "pubmed":
                        if self.retriever_name == "allenai/specter":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/Ebz8ySXt815FotxC1KkDbuABNycudBCoirTWkKfl8SEswA?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "facebook/contriever":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EWecRNfTxbRMnM0ByGMdiAsBJbGJOX_bpnUoyXY9Bj4_jQ?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "ncbi/MedCPT-Query-Encoder":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EVCuryzOqy5Am5xzRu6KJz4B6dho7Tv7OuTeHSh3zyrOAw?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                    elif self.corpus_name == "wikipedia":
                        if self.retriever_name == "allenai/specter":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/Ed7zG3_ce-JOmGTbgof3IK0BdD40XcuZ7AGZRcV_5D2jkA?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "facebook/contriever":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/ETKHGV9_KNBPmDM60MWjEdsBXR4P4c7zZk1HLLc0KVaTJw?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "ncbi/MedCPT-Query-Encoder":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EXoxEANb_xBFm6fa2VLRmAcBIfCuTL-5VH6vl4GxJ06oCQ?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                    os.system("unzip {:s} -d {:s}".format(os.path.join(self.index_dir, "embedding.zip"), self.index_dir))
                    os.system("rm {:s}".format(os.path.join(self.index_dir, "embedding.zip")))
                    h_dim = 768
                else:
                    h_dim = embed(chunk_dir=self.chunk_dir, index_dir=self.index_dir, model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"))
    
                print("[In progress] Embedding finished! The dimension of the embeddings is {:d}.".format(h_dim))
                self.index = construct_index(index_dir=self.index_dir, model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"), h_dim=h_dim, HNSW=HNSW)
                print("[Finished] Corpus indexing finished!")
                self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]
            if "contriever" in self.retriever_name.lower():
                self.embedding_function = SentenceTransformer(self.retriever_name, device="cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.embedding_function = CustomizeSentenceTransformer(self.retriever_name, device="cuda" if torch.cuda.is_available() else "cpu")
            self.embedding_function.eval()

    def get_relevant_documents(self, question, k=32, id_only=False, **kwarg):
        assert type(question) == str
        question = [question]

        if "bm25" in self.retriever_name.lower():
            res_ = [[]]
            hits = self.index.search(question[0], k=k)
            res_[0].append(np.array([h.score for h in hits]))
            ids = [h.docid for h in hits]
            indices = [{"source": '_'.join(h.docid.split('_')[:-1]), "index": eval(h.docid.split('_')[-1])} for h in hits]
        else:
            with torch.no_grad():
                query_embed = self.embedding_function.encode(question, **kwarg)
            res_ = self.index.search(query_embed, k=k)

            # Log the search results for debugging
            print(f"Search results: {res_}")
            print(f"Metadata length: {len(self.metadatas)}")

            ids = ['_'.join([self.metadatas[i]["source"], str(self.metadatas[i]["index"])]) for i in res_[1][0]]
            indices = [self.metadatas[i] for i in res_[1][0]]

        scores = res_[0][0].tolist()

        if id_only:
            return [{"id": i} for i in ids], scores
        else:
            return self.idx2txt(indices), scores
    
    def idx2txt(self, indices):
        # Metadata dosyasının yolu index_dir altında metadatas.jsonl olarak belirleniyor.
        metadata_file = os.path.join(self.index_dir, "metadatas.jsonl")
        with open(metadata_file, 'r') as f:
            lines = f.read().strip().split('\n')
        return [json.loads(lines[i["index"]]) for i in indices]

class RetrievalSystem:
    def __init__(self, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", HNSW=False, cache=False):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        assert self.corpus_name in corpus_names
        assert self.retriever_name in retriever_names
        self.retrievers = []
        for retriever in retriever_names[self.retriever_name]:
            self.retrievers.append([])
            for corpus in corpus_names[self.corpus_name]:
                self.retrievers[-1].append(Retriever(retriever, corpus, db_dir, HNSW=HNSW))
        self.cache = cache
        if self.cache:
            self.docExt = DocExtracter(cache=True, corpus_name=self.corpus_name, db_dir=db_dir)
        else:
            self.docExt = None
    
    def retrieve(self, question, k=32, rrf_k=100, id_only=False):
        """
            Given questions, return the relevant snippets from the corpus.
        """
        assert type(question) == str
    
        output_id_only = id_only
        if self.cache:
            id_only = True
    
        texts = []
        scores = []
    
        if "RRF" in self.retriever_name:
            k_ = max(k * 2, 100)
        else:
            k_ = k
        for i in range(len(retriever_names[self.retriever_name])):
            texts.append([])
            scores.append([])
            for j in range(len(corpus_names[self.corpus_name])):
                t, s = self.retrievers[i][j].get_relevant_documents(question, k=k_, id_only=id_only)
                texts[-1].append(t)
                scores[-1].append(s)
        texts, scores = self.merge(texts, scores, k=k, rrf_k=rrf_k)
        if self.cache:
            texts = self.docExt.extract(texts)
        return texts, scores
    
    def merge(self, texts, scores, k, rrf_k):
        # Tamamen düzleştirmek için yardımcı fonksiyon.
        def flatten(lst):
            flat = []
            for el in lst:
                if isinstance(el, list):
                    flat.extend(flatten(el))
                else:
                    flat.append(el)
            return flat
        flat_texts = flatten(texts)
        flat_scores = flatten(scores)
        RRF_dict = {}
        for item, score in zip(flat_texts, flat_scores):
            if isinstance(item, dict):
                doc_id = item.get("id")
                if doc_id is None:
                    doc_id = "{}_{}".format(item.get("source", "unknown"), item.get("index", 0))
            else:
                # Eğer item liste ise, onu yine iterasyona sokuyoruz.
                if isinstance(item, list):
                    for sub_item in item:
                        if isinstance(sub_item, dict):
                            doc_id = sub_item.get("id")
                            if doc_id is None:
                                doc_id = "{}_{}".format(sub_item.get("source", "unknown"), sub_item.get("index", 0))
                            if doc_id in RRF_dict:
                                RRF_dict[doc_id] += score
                            else:
                                RRF_dict[doc_id] = score
                    continue
                else:
                    doc_id = str(item)
            if doc_id in RRF_dict:
                RRF_dict[doc_id] += score
            else:
                RRF_dict[doc_id] = score
        merged = sorted(RRF_dict.items(), key=lambda x: x[1], reverse=True)[:k]
        return merged, flat_scores

class DocExtracter:
    def __init__(self, db_dir="./corpus", cache=False, corpus_name="MedCorp"):
        self.db_dir = db_dir
        self.cache = cache
        print("Initializing the document extracter...")
        for corpus in corpus_names[corpus_name]:
            if not os.path.exists(os.path.join(self.db_dir, corpus, "chunk")):
                print("Cloning the {:s} corpus from Huggingface...".format(corpus))
                os.system("git clone https://huggingface.co/datasets/MedRAG/{:s} {:s}".format(corpus, os.path.join(self.db_dir, corpus)))
                if corpus == "statpearls":
                    print("Downloading the statpearls corpus from NCBI bookshelf...")
                    os.system("wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {:s}".format(os.path.join(self.db_dir, corpus)))
                    os.system("tar -xzvf {:s} -C {:s}".format(os.path.join(self.db_dir, corpus, "statpearls_NBK430685.tar.gz"), os.path.join(self.db_dir, corpus)))
                    print("Chunking the statpearls corpus...")
                    os.system("python src/data/statpearls.py")
        if self.cache:
            id2text_path = os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"]))
            if os.path.exists(id2text_path):
                self.dict = json.load(open(id2text_path))
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    chunk_dir = os.path.join(self.db_dir, corpus, "chunk")
                    for fname in tqdm.tqdm(sorted(os.listdir(chunk_dir))):
                        fpath = os.path.join(chunk_dir, fname)
                        if open(fpath).read().strip() == "":
                            continue
                        for i, line in enumerate(open(fpath).read().strip().split('\n')):
                            item = json.loads(line)
                            _ = item.pop("contents", None)
                            self.dict[item["id"]] = item
                with open(id2text_path, 'w') as f:
                    json.dump(self.dict, f)
        else:
            id2path_path = os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"]))
            if os.path.exists(id2path_path):
                self.dict = json.load(open(id2path_path))
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    chunk_dir = os.path.join(self.db_dir, corpus, "chunk")
                    for fname in tqdm.tqdm(sorted(os.listdir(chunk_dir))):
                        fpath = os.path.join(chunk_dir, fname)
                        if open(fpath).read().strip() == "":
                            continue
                        for i, line in enumerate(open(fpath).read().strip().split('\n')):
                            item = json.loads(line)
                            self.dict[item["id"]] = {"fpath": os.path.join(corpus, "chunk", fname), "index": i}
                with open(id2path_path, 'w') as f:
                    json.dump(self.dict, f, indent=4)
        print("Initialization finished!")
    
    def extract(self, ids):
        if self.cache:
            output = []
            for i in ids:
                item = self.dict[i] if isinstance(i, str) else self.dict[i["id"]]
                output.append(item)
        else:
            output = []
            for i in ids:
                item = self.dict[i] if isinstance(i, str) else self.dict[i["id"]]
                fpath = os.path.join(self.db_dir, item["fpath"])
                with open(fpath, 'r') as f:
                    lines = f.read().strip().split('\n')
                output.append(json.loads(lines[item["index"]]))
        return output
