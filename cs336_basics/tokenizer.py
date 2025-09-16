import json
import regex
from typing import Iterable, Iterator, Optional
from pathlib import Path


class Tokenizer:
    """
    A BPE (Byte Pair Encoding) tokenizer that can encode text into token IDs
    and decode token IDs back into text.
    """
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Optional[list[str]] = None):
        """
        Construct from a vocabulary, merges, and optional special tokens.
        
        Args:
            vocab: Mapping from token ID to token bytes
            merges: List of BPE merges as tuples of (token1, token2)
            special_tokens: List of special tokens to add to vocab if missing
        """
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.special_tokens = special_tokens or []
        
        # Create reverse mapping from bytes to token ID
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        
        # Add special tokens to vocab if they don't exist
        if self.special_tokens:
            next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
            for special_token in self.special_tokens:
                special_bytes = special_token.encode("utf-8")
                if special_bytes not in self.token_to_id:
                    self.vocab[next_id] = special_bytes
                    self.token_to_id[special_bytes] = next_id
                    next_id += 1
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]] = None):
        """
        Build from serialized vocab/merges files.
        
        Args:
            vocab_filepath: Path to vocabulary JSON file
            merges_filepath: Path to merges text file
            special_tokens: Optional list of special tokens
        """
        # Load vocabulary
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Convert string keys to integers and string values to bytes
        vocab = {}
        for token_id_str, token_str in vocab_data.items():
            token_id = int(token_id_str)
            token_bytes = token_str.encode('utf-8')
            vocab[token_id] = token_bytes
        
        # Load merges
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split()) == 2:
                    token1_str, token2_str = line.split()
                    token1_bytes = token1_str.encode('utf-8')
                    token2_bytes = token2_str.encode('utf-8')
                    merges.append((token1_bytes, token2_bytes))
        
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode a string into token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        # Handle special tokens first
        tokens = self._preprocess_special_tokens(text)
        
        # Apply BPE merges
        tokens = self._apply_bpe_merges(tokens)
        
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # If token not found, this shouldn't happen with proper BPE
                # but we'll handle it by encoding each byte separately
                for byte_val in token:
                    byte_token = bytes([byte_val])
                    if byte_token in self.token_to_id:
                        token_ids.append(self.token_to_id[byte_token])
        
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, lazily yield token IDs.
        Enables memory-efficient tokenization of large files.
        
        Args:
            iterable: Iterable of strings (e.g., file handle)
            
        Yields:
            Token IDs one at a time
        """
        for line in iterable:
            token_ids = self.encode(line)
            for token_id in token_ids:
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        if not ids:
            return ""
        
        # Convert IDs to bytes
        token_bytes = []
        for token_id in ids:
            if token_id in self.vocab:
                token_bytes.append(self.vocab[token_id])
            else:
                # Handle unknown token ID by treating as byte
                if 0 <= token_id <= 255:
                    token_bytes.append(bytes([token_id]))
        
        # Concatenate all bytes and decode to string
        full_bytes = b''.join(token_bytes)
        try:
            return full_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback: decode with error handling
            return full_bytes.decode('utf-8', errors='replace')
    
    def _preprocess_special_tokens(self, text: str) -> list[bytes]:
        """
        Preprocess text to handle special tokens and perform initial tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of token bytes
        """
        if not self.special_tokens:
            return self._tokenize_chunk(text)
        
        # Sort special tokens by length (longest first) to handle overlapping tokens
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        
        # Split text by special tokens while preserving the special tokens
        pattern = '|'.join(regex.escape(token) for token in sorted_special_tokens)
        parts = regex.split(f'({pattern})', text)
        
        tokens = []
        for part in parts:
            if part in sorted_special_tokens:
                # This is a special token
                tokens.append(part.encode('utf-8'))
            elif part:
                # This is regular text, tokenize it
                tokens.extend(self._tokenize_chunk(part))
        
        return tokens
    
    def _tokenize_chunk(self, chunk: str) -> list[bytes]:
        """
        Tokenize a chunk of text using regex pattern.
        
        Args:
            chunk: Text chunk to tokenize
            
        Returns:
            List of token bytes
        """
        # Use GPT-2 style regex pattern for tokenization
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        tokens = []
        for match in regex.findall(pattern, chunk):
            tokens.append(match.encode('utf-8'))
        
        return tokens
    
    def _apply_bpe_merges(self, tokens: list[bytes]) -> list[bytes]:
        """
        Apply BPE merges to tokenize the input.
        
        Args:
            tokens: List of token bytes
            
        Returns:
            List of merged token bytes
        """
        if not tokens:
            return tokens
        
        # Get special token bytes for comparison
        special_token_bytes = set()
        if self.special_tokens:
            special_token_bytes = {token.encode('utf-8') for token in self.special_tokens}
        
        # Process each token
        result_tokens = []
        for token in tokens:
            # If this is a special token, keep it as-is
            if token in special_token_bytes:
                result_tokens.append(token)
            else:
                # Convert to individual bytes and apply BPE merges
                byte_tokens = [bytes([byte_val]) for byte_val in token]
                
                # Apply merges in order
                for merge_pair in self.merges:
                    new_tokens = []
                    i = 0
                    while i < len(byte_tokens):
                        if (i < len(byte_tokens) - 1 and 
                            byte_tokens[i] == merge_pair[0] and 
                            byte_tokens[i + 1] == merge_pair[1]):
                            # Found a merge pair, combine them
                            new_tokens.append(merge_pair[0] + merge_pair[1])
                            i += 2
                        else:
                            new_tokens.append(byte_tokens[i])
                            i += 1
                    byte_tokens = new_tokens
                
                result_tokens.extend(byte_tokens)
        
        return result_tokens
