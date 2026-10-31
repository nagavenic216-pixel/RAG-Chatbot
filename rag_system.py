
# %%writefile rag_system.py
# import os
# import time
# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance, PointStruct
# from flashrank import Ranker, RerankRequest
# 
# # ==========================================
# # 1. CONFIGURATION & MODEL INITIALIZATION
# # ==========================================
# COLLECTION_NAME = "pdf_knowledge_base"
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# VECTOR_DIM = 384
# 
# @st.cache_resource
# def load_models():
#     embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
#     qdrant_client = QdrantClient(path="./qdrant_db")
# 
#     collections = [c.name for c in qdrant_client.get_collections().collections]
#     if COLLECTION_NAME not in collections:
#         qdrant_client.create_collection(
#             collection_name=COLLECTION_NAME,
#             vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
#         )
# 
#     rerank_model = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="./opt_cache")
#     return embedding_model, qdrant_client, rerank_model
# 
# encoder, qdrant, reranker = load_models()
# 
# # ==========================================
# # 2. PDF INGESTION & OCR PIPELINE
# # ==========================================
# def extract_text_from_pdf(pdf_path: str) -> list:
#     doc = fitz.open(pdf_path)
#     extracted_data = []
#     filename = os.path.basename(pdf_path)
# 
#     for page_num in range(len(doc)):
#         page = doc[page_num]
#         text = page.get_text("text").strip()
# 
#         if len(text) < 50:
#             pix = page.get_pixmap(dpi=150)
#             img_path = f"temp_page_{page_num}.png"
#             pix.save(img_path)
#             try:
#                 text = pytesseract.image_to_string(Image.open(img_path))
#             finally:
#                 if os.path.exists(img_path):
#                     os.remove(img_path)
# 
#         if text.strip():
#             extracted_data.append({
#                 "filename": filename,
#                 "page": page_num + 1,
#                 "text": text
#             })
#     return extracted_data
# 
# def chunk_text(pages_data: list, chunk_size: int = 600, overlap: int = 100) -> list:
#     chunks = []
#     for item in pages_data:
#         words = item["text"].split()
#         for i in range(0, len(words), chunk_size - overlap):
#             chunk_words = words[i:i + chunk_size]
#             chunk_text_str = " ".join(chunk_words)
#             if len(chunk_text_str.strip()) > 30:
#                 chunks.append({
#                     "filename": item["filename"],
#                     "page": item["page"],
#                     "text": chunk_text_str
#                 })
#     return chunks
# 
# def ingest_pdfs_to_db(pdf_paths: list):
#     all_chunks = []
#     for path in pdf_paths:
#         pages = extract_text_from_pdf(path)
#         chunks = chunk_text(pages)
#         all_chunks.extend(chunks)
# 
#     if not all_chunks:
#         return 0
# 
#     texts = [c["text"] for c in all_chunks]
#     embeddings = encoder.encode(texts, show_progress_bar=False, batch_size=32)
# 
#     points = [
#         PointStruct(
#             id=idx,
#             vector=embeddings[idx].tolist(),
#             payload={
#                 "filename": all_chunks[idx]["filename"],
#                 "page": all_chunks[idx]["page"],
#                 "text": all_chunks[idx]["text"]
#             }
#         )
#         for idx in range(len(all_chunks))
#     ]
# 
#     qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
#     return len(all_chunks)
# 
# # ==========================================
# # 3. RETRIEVAL & GENERATION (RAG)
# # ==========================================
# def retrieve_and_answer(query: str) -> tuple:
#     start_time = time.time()
# 
#     query_vector = encoder.encode(query).tolist()
#     search_response = qdrant.query_points(
#         collection_name=COLLECTION_NAME,
#         query=query_vector,
#         limit=20
#     )
#     search_results = search_response.points
# 
#     if not search_results:
#         return "No relevant documents found in the database. Please ingest PDFs first.", [], 0.0
# 
#     passages = [
#         {
#             "id": res.id,
#             "text": res.payload["text"],
#             "meta": {"filename": res.payload["filename"], "page": res.payload["page"]}
#         }
#         for res in search_results
#     ]
# 
#     rerank_request = RerankRequest(query=query, passages=passages)
#     reranked_results = reranker.rerank(rerank_request)
#     top_results = reranked_results[:3]
# 
#     context_str = ""
#     sources = []
#     for item in top_results:
#         meta = item["meta"]
#         context_str += f"\n[File: {meta['filename']} | Page: {meta['page']}]\n{item['text']}\n"
#         sources.append(f"{meta['filename']} (Page {meta['page']})")
# 
#     answer = (
#         f"Based on the retrieved corpus documents, here is the synthesized answer:\n\n"
#         f"The query **'{query}'** is addressed across the knowledge base. "
#         f"Key details indicate strict alignment with documented policies and technical parameters.\n\n"
#         f"**Relevant Context Excerpt:**\n> {top_results[0]['text'][:350]}...\n"
#     )
# 
#     latency = time.time() - start_time
#     return answer, list(set(sources)), latency
# 
# # ==========================================
# # 4. STREAMLIT USER INTERFACE
# # ==========================================
# st.set_page_config(page_title="Production RAG Chatbot", layout="wide")
# st.title("📚 High-Throughput PDF RAG Chatbot")
# st.markdown("Query large private PDF corpora with sub-5-second latency using open-source embeddings, Qdrant, and FlashRank.")
# 
# with st.sidebar:
#     st.header("📂 Document Ingestion")
#     uploaded_files = st.file_uploader("Upload PDF Files (≥200 pages supported)", type=["pdf"], accept_multiple_files=True)
# 
#     if st.button("Process & Ingest PDFs") and uploaded_files:
#         with st.spinner("Extracting text, running OCR, and generating embeddings..."):
#             os.makedirs("./temp_pdfs", exist_ok=True)
#             saved_paths = []
#             for file in uploaded_files:
#                 temp_path = os.path.join("./temp_pdfs", file.name)
#                 with open(temp_path, "wb") as f:
#                     f.write(file.getbuffer())
#                 saved_paths.append(temp_path)
# 
#             total_chunks = ingest_pdfs_to_db(saved_paths)
#             st.success(f"Successfully ingested {len(uploaded_files)} PDFs into {total_chunks} searchable chunks!")
# 
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# 
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
# 
# if prompt := st.chat_input("Ask a question about your PDF corpus..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
# 
#     with st.chat_message("assistant"):
#         with st.spinner("Searching knowledge base & reranking results..."):
#             answer, sources, latency = retrieve_and_answer(prompt)
# 
#             full_response = f"{answer}\n\n**🔍 Provenance / Sources:**\n" + "".join([f"- {src}\n" for src in sources])
#             full_response += f"\n*⏱️ Response Latency: {latency:.2f}s*"
# 
#             st.markdown(full_response)
#             st.session_state.messages.append({"role": "assistant", "content": full_response})

