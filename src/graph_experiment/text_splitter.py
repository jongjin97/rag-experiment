import re
import itertools
from typing import List, Optional

# Assuming standard langchain import is available, or we can just implement a simple logic wrapper
# If RecursiveCharacterTextSplitter is not available, we can mock or remove inheritance if needed.
# For now, let's assume it's used as a mixin or standalone class.

class TableAwareSplitter:
    """
    A custom text splitter wrapper that:
    1. Takes an existing splitter's chunks (or splits text itself).
    2. Detects table placeholders [TABLE_ID_MAXIDX].
    3. Expands the chunk into multiple versions, one for each table part (0..MAXIDX).
    """
    def __init__(self, base_splitter=None):
        """
        Args:
            base_splitter: An instance of a text splitter (must have .split_text method).
                           If None, assumes input to expand_chunks is already a list of strings.
        """
        self.base_splitter = base_splitter
        # Pattern to match [TABLE_X_Y]
        # Captures: TABLE_ID (alphanumeric), Max Index (digits)
        self.placeholder_pattern = re.compile(r"\[(TABLE_[a-zA-Z0-9_]+)_(\d+)\]")

    def split_text(self, text: str) -> List[str]:
        if not self.base_splitter:
            # If no base splitter, treat whole text as one chunk
            initial_chunks = [text]
        else:
            initial_chunks = self.base_splitter.split_text(text)
            
        return self.expand_chunks(initial_chunks)

    def expand_chunks(self, chunks: List[str]) -> List[str]:
        final_chunks = []
        
        for chunk in chunks:
            # Find all unique placeholders in this chunk
            # matches: list of (table_base_id, max_index_str)
            matches = self.placeholder_pattern.findall(chunk)
            
            if not matches:
                final_chunks.append(chunk)
                continue
            
            # Prepare replacement options for each match
            # replacement_map: position index -> list of replacement strings
            # But regex findall doesn't give positions cleanly for replacement.
            # Better strategy: Identify unique placeholders.
            
            unique_matches = sorted(list(set(matches)))
            
            # Build expansion lists for each unique match
            # e.g. Match ('TABLE_0', '2') -> replaces usage with [TABLE_0_0], [TABLE_0_1], [TABLE_0_2]
            
            replacements_lists = []
            for base_id, max_idx_str in unique_matches:
                max_idx = int(max_idx_str)
                options = []
                for i in range(max_idx + 1):
                    # The format of the expanded placeholder: [TABLE_X_i]
                    # Note: currently user uses [TABLE_X_MAX] to imply Range.
                    # But the final chunks should probably point to specific chunks so the LLM/Loader knows exactly what to look up?
                    # "TABLE_478_0" is the Key in JSON.
                    # So we replace [TABLE_478_2] with [TABLE_478_0], [TABLE_478_1], [TABLE_478_2] respectively in different chunks.
                    options.append(f"[{base_id}_{i}]")
                replacements_lists.append(options)
            
            # Compute Cartesian Product of all expansions
            # e.g. T1 (2 parts), T2 (1 part) -> 2 * 1 = 2 expanded chunks
            
            combinations = list(itertools.product(*replacements_lists))
            
            # Generate expanded chunks
            for combo in combinations:
                # combo is a tuple of replacements corresponding to unique_matches
                new_chunk = chunk
                for idx, match_tuple in enumerate(unique_matches):
                    base_id, max_idx_str = match_tuple
                    original_placeholder = f"[{base_id}_{max_idx_str}]"
                    replacement_str = combo[idx]
                    
                    # Replace ALL occurrences of this specific placeholder
                    new_chunk = new_chunk.replace(original_placeholder, replacement_str)
                    
                final_chunks.append(new_chunk)
                
        return final_chunks
