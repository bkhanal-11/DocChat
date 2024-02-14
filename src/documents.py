import os
from typing import List
import logging
import tiktoken

from langchain_community.document_loaders import *
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.indexes import SQLRecordManager, index

FILE_LOADER_MAPPING = {
    "doc": (UnstructuredWordDocumentLoader, {}),
    "docx": (UnstructuredWordDocumentLoader, {}),
    "epub": (UnstructuredEPubLoader, {}),
    "html": (UnstructuredHTMLLoader, {}),
    "md": (UnstructuredMarkdownLoader, {}),
    "pdf": (PyPDFLoader, {}),
    "txt": (TextLoader, {"encoding": "utf8"}),
}

class VectorDB:
    def __init__(self, upload_files, session_id):
        for file in upload_files:
            with open(f'resources/{session_id}/uploads/{file.name}', 'wb') as f:
                f.write(file.read())

    def num_tokens_from_string(self, string: str, encoding_name: str = "cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def load_documents(self, path: str) -> List[Document]:
        """
        Loads documents from a directory.
        """
        loaded_documents = []
            
        for path, _, files in os.walk(os.path.join(path, 'uploads')):
            for file in files:
                ext = os.path.splitext(file)[-1][1:].lower()
                if ext in FILE_LOADER_MAPPING:
                    loader_class, loader_args = FILE_LOADER_MAPPING[ext]
                    # Now create the loader instance with the file path
                    loader = loader_class(os.path.join(path, file), **loader_args)
                else:
                    continue
                    # loader = UnstructuredFileLoader(os.path.join(path, file))
                
                loaded_documents.extend(loader.load())

        return loaded_documents

    def get_texts(self,
        documents: List[Document],
        text_splitter: RecursiveCharacterTextSplitter
    ) -> List[str]:
        """
        Splits the content of the documents into chunks using
        a given text splitter.
        """
        splits = text_splitter.split_documents(documents)
        splits = filter_complex_metadata(splits)
        return splits

    def process_documents(self, path: str) -> List[str]:
        """
        Process all documents in the data directory and
        return their content as chunks of text.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=self.num_tokens_from_string,
        )
        documents: List[Document] = self.load_documents(path)
        return self.get_texts(documents, text_splitter)

    def create_store(self, splits: list, path: str, filename: str):
        """
        Create a vector store from the provided texts.
        """
        try:
            vectorstore = Chroma(
                collection_name="docschat",
                embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
                persist_directory=f"{path}/vectordb",
            )

            record_manager = SQLRecordManager(
                db_url=f"sqlite:///{path}/vectordb/index.sqlite",
                namespace="docschat",
            )

            record_manager.create_schema()

            if filename:
                ids = vectorstore.get(where = {'source': filename})['ids']
                print(ids)
                vectorstore.delete(ids = ids)

            indexing_result = index(
                docs_source=splits,
                record_manager=record_manager,
                vector_store=vectorstore,
                batch_size=1000,
                cleanup="incremental",
                source_id_key="source",
            )

            logging.info(f"Indexing result: {indexing_result}")
        except Exception as e:
            logging.error(f"An error occurred while creating the store: {e}")

    def create_vectors(self, base_path: str, filename: str = None) -> None:
        splits: List[str] = self.process_documents(base_path)
        self.create_store(splits=splits, path=base_path, filename=filename)
    
