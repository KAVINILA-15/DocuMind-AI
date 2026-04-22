from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, strategy="A"):
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    Supports two comparison strategies.
    
    strategy "A": chunk_size = 500, chunk_overlap = 100
    strategy "B": chunk_size = 800, chunk_overlap = 200
    """
    if strategy == "B":
        chunk_size = 800
        chunk_overlap = 200
    else:
        # Default to A
        chunk_size = 500
        chunk_overlap = 100
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # split_documents automatically preserves metadata from the original Document objects
    chunks = text_splitter.split_documents(documents)
    return chunks
