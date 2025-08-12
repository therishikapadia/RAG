import re
import hashlib
from typing import List, Dict, Any
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\[\]{}"\'`-]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def sentence_chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Chunk text into overlapping segments based on sentences.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size for each chunk (in characters)
        overlap: Overlap between chunks (in characters)
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Clean the text
    text = clean_text(text)
    
    # Split into sentences
    try:
        sentences = sent_tokenize(text)
    except:
        # Fallback to simple splitting if NLTK fails
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = ""
    current_length = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed chunk_size
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) > overlap:
                # Find a good breaking point for overlap
                overlap_text = current_chunk[-overlap:]
                # Try to start overlap at sentence boundary
                overlap_sentences = sent_tokenize(overlap_text)
                if len(overlap_sentences) > 1:
                    current_chunk = ' '.join(overlap_sentences[1:])
                else:
                    current_chunk = overlap_text
                current_length = len(current_chunk)
            else:
                current_chunk = ""
                current_length = 0
        
        # Add current sentence
        if current_chunk:
            current_chunk += " " + sentence
        else:
            current_chunk = sentence
        current_length = len(current_chunk)
        
        i += 1
    
    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    return chunks


def word_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Chunk text into overlapping segments based on words.
    
    Args:
        text: Input text to chunk
        chunk_size: Target number of words per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    text = clean_text(text)
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)
        
        # Move start position considering overlap
        start = end - overlap
        
        if start >= len(words):
            break
    
    return chunks


def make_doc_id(source: str, chunk_index: int) -> str:
    """
    Create a unique document ID for a chunk.
    
    Args:
        source: Source document name/path
        chunk_index: Index of the chunk within the document
        
    Returns:
        Unique document ID
    """
    # Create a hash of the source to handle long paths
    source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    return f"{source_hash}_{chunk_index}"


def make_metadata(source: str, chunk_index: int, **kwargs) -> Dict[str, Any]:
    """
    Create metadata dictionary for a document chunk.
    
    Args:
        source: Source document name/path
        chunk_index: Index of the chunk within the document
        **kwargs: Additional metadata fields
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        "source": source,
        "chunk_index": chunk_index,
        "doc_id": make_doc_id(source, chunk_index),
        **kwargs
    }
    return metadata


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text (simple implementation).
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction based on frequency
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'around',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself',
        'we', 'us', 'our', 'ourselves', 'you', 'your', 'yourself', 'yourselves',
        'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
        'it', 'its', 'itself', 'they', 'them', 'their', 'themselves'
    }
    
    # Clean and split text
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        if len(word) > 2 and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in keywords[:max_keywords]]


def split_by_headers(text: str) -> List[Dict[str, str]]:
    """
    Split text by markdown-style headers.
    
    Args:
        text: Input text with headers
        
    Returns:
        List of sections with header and content
    """
    sections = []
    current_header = "Introduction"
    current_content = ""
    
    lines = text.split('\n')
    
    for line in lines:
        # Check if line is a header (starts with #)
        header_match = re.match(r'^(#{1,6})\s+(.+)', line.strip())
        
        if header_match:
            # Save previous section if it has content
            if current_content.strip():
                sections.append({
                    "header": current_header,
                    "content": current_content.strip()
                })
            
            # Start new section
            current_header = header_match.group(2)
            current_content = ""
        else:
            # Add line to current content
            current_content += line + "\n"
    
    # Add final section
    if current_content.strip():
        sections.append({
            "header": current_header,
            "content": current_content.strip()
        })
    
    return sections


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count (assumes ~4 characters per token).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def truncate_text(text: str, max_tokens: int = 4000) -> str:
    """
    Truncate text to approximate token limit.
    
    Args:
        text: Input text
        max_tokens: Maximum number of tokens
        
    Returns:
        Truncated text
    """
    max_chars = max_tokens * 4  # Rough estimate
    if len(text) <= max_chars:
        return text
    
    # Try to truncate at sentence boundary
    truncated = text[:max_chars]
    sentences = sent_tokenize(truncated)
    
    if len(sentences) > 1:
        # Remove the last (potentially incomplete) sentence
        return ' '.join(sentences[:-1])
    else:
        return truncated


def merge_short_chunks(chunks: List[str], min_length: int = 100) -> List[str]:
    """
    Merge chunks that are too short with adjacent chunks.
    
    Args:
        chunks: List of text chunks
        min_length: Minimum length for a chunk
        
    Returns:
        List of merged chunks
    """
    if not chunks:
        return []
    
    merged = []
    current_chunk = chunks[0]
    
    for i in range(1, len(chunks)):
        if len(current_chunk) < min_length:
            # Merge with next chunk
            current_chunk += " " + chunks[i]
        else:
            # Current chunk is long enough, save it and start new one
            merged.append(current_chunk)
            current_chunk = chunks[i]
    
    # Add the last chunk
    if current_chunk:
        merged.append(current_chunk)
    
    return merged


def remove_duplicates(chunks: List[str], similarity_threshold: float = 0.9) -> List[str]:
    """
    Remove duplicate or highly similar chunks.
    
    Args:
        chunks: List of text chunks
        similarity_threshold: Threshold for considering chunks similar
        
    Returns:
        List of unique chunks
    """
    if not chunks:
        return []
    
    unique_chunks = []
    
    for chunk in chunks:
        is_duplicate = False
        
        for existing_chunk in unique_chunks:
            # Simple similarity check based on character overlap
            similarity = calculate_text_similarity(chunk, existing_chunk)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_chunks.append(chunk)
    
    return unique_chunks


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity based on character overlap.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to lowercase and get character sets
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def validate_chunk(chunk: str, min_length: int = 50, max_length: int = 2000) -> bool:
    """
    Validate if a chunk meets quality criteria.
    
    Args:
        chunk: Text chunk to validate
        min_length: Minimum length requirement
        max_length: Maximum length requirement
        
    Returns:
        True if chunk is valid, False otherwise
    """
    if not chunk or not chunk.strip():
        return False
    
    chunk_length = len(chunk.strip())
    
    if chunk_length < min_length or chunk_length > max_length:
        return False
    
    # Check if chunk has meaningful content (not just punctuation/numbers)
    meaningful_chars = re.sub(r'[^\w\s]', '', chunk)
    if len(meaningful_chars) < min_length * 0.7:
        return False
    
    return True