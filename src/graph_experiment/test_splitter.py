from src.graph_experiment.text_splitter import TableAwareSplitter

def test_expansion():
    splitter = TableAwareSplitter(base_splitter=None)
    
    # test case 1: Single table expansion
    text1 = "Here is some context. [TABLE_10_2] End of context."
    chunks1 = splitter.split_text(text1)
    
    print("--- Test Case 1 ---")
    print(f"Input: {text1}")
    print(f"Output Chunks ({len(chunks1)}):")
    for c in chunks1:
        print(f" > {c}")
        
    assert len(chunks1) == 3
    assert "[TABLE_10_0]" in chunks1[0]
    assert "[TABLE_10_1]" in chunks1[1]
    assert "[TABLE_10_2]" in chunks1[2]

    # test case 2: Multiple tables
    text2 = "Start. [TABLE_A_1] Middle. [TABLE_B_0] End."
    # A has 0,1 (2 parts). B has 0 (1 part). Total 2*1 = 2 versions.
    chunks2 = splitter.split_text(text2)
    
    print("\n--- Test Case 2 ---")
    print(f"Input: {text2}")
    print(f"Output Chunks ({len(chunks2)}):")
    for c in chunks2:
        print(f" > {c}")

    assert len(chunks2) == 2
    # Combinations: (A0, B0), (A1, B0)
    
    # test case 3: No tables
    text3 = "Just plain text."
    chunks3 = splitter.split_text(text3)
    print("\n--- Test Case 3 ---")
    print(f"Output Chunks ({len(chunks3)}): {chunks3}")
    assert len(chunks3) == 1
    assert chunks3[0] == text3

if __name__ == "__main__":
    test_expansion()
