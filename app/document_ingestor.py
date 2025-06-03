import os
import json
import fitz  # PyMuPDF
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DocumentIngestor:
    def __init__(self, docs_dir="chem_docs", index_dir="database"):
        self.docs_dir = docs_dir
        self.index_dir = index_dir
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        os.makedirs(index_dir, exist_ok=True)
        
    def load_documents(self):
        """加载并解析所有技术文档"""
        documents = []
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                path = os.path.join(root, file)
                if file.endswith('.pdf'):
                    text = self._extract_pdf(path)
                elif file.endswith('.docx'):
                    text = self._extract_docx(path)
                else:
                    continue
                
                # 添加文档元数据
                rel_path = os.path.relpath(path, self.docs_dir)
                category = os.path.dirname(rel_path).replace(os.sep, '/')
                documents.append({
                    'path': rel_path,
                    'text': text,
                    'category': category,
                    'title': os.path.splitext(file)[0]
                })
        return documents

    def _extract_pdf(self, file_path):
        """提取PDF文本并保留结构信息"""
        text = ""
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += f"\n\n--- PAGE {page_num+1} ---\n"
                text += page.get_text("text")
        return text

    def _extract_docx(self, file_path):
        """提取DOCX文档结构"""
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            # 保留标题结构
            if para.style.name.startswith('Heading'):
                text += f"\n\n{para.style.name.upper()}: {para.text}\n"
            else:
                text += para.text + "\n"
        return text

    def chunk_documents(self, documents, chunk_size=500, overlap=50):
        """将文档分割为带元数据的文本块"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len
        )
        
        chunks = []
        for doc in documents:
            text_chunks = splitter.split_text(doc['text'])
            for i, chunk in enumerate(text_chunks):
                # 提取块内标题作为上下文
                headers = [line for line in chunk.split('\n') if line.startswith('HEADING')]
                section = headers[-1].split(': ')[1] if headers else "General"
                
                chunks.append({
                    'chunk_id': f"{doc['path']}-{i}",
                    'text': chunk,
                    'source': doc['path'],
                    'title': doc['title'],
                    'category': doc['category'],
                    'section': section,
                    'page': self._estimate_page(chunk, doc['text'])
                })
        return chunks

    def _estimate_page(self, chunk, full_text):
        """估算文本块所在的页码"""
        start_index = full_text.find(chunk[:100])
        if start_index == -1: 
            return "N/A"
        return str(start_index // 2000 + 1)  # 假设每页≈2000字符

    def build_indexes(self, chunks):
        """构建向量和元数据索引"""
        # 生成嵌入向量
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # 保存索引和元数据
        faiss.write_index(index, os.path.join(self.index_dir, "chem_index.faiss"))
        with open(os.path.join(self.index_dir, "metadata.json"), "w") as f:
            json.dump(chunks, f, indent=2)
        
        print(f"索引构建完成: {len(chunks)}个文档块")

# 使用示例
if __name__ == "__main__":
    ingestor = DocumentIngestor()
    documents = ingestor.load_documents()
    chunks = ingestor.chunk_documents(documents)
    ingestor.build_indexes(chunks)