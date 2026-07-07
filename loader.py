import os
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(file_paths):
    """
    Load multiple PDF files and extract text along with metadata.
    PyPDFLoader automatically adds 'source' (file path) and 'page' (page number) to metadata.
    """
    documents = []
    for file_path in file_paths:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Ensure the source just contains the basename of the file for cleaner UI display
            for doc in docs:
                doc.metadata['source'] = os.path.basename(doc.metadata.get('source', file_path))
                
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return documents
