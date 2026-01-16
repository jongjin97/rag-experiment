import networkx as nx
from tqdm import tqdm
from src.light_rag.storage.graph import GraphStorage
from src.light_rag.storage.vector import VectorStorage
from src.light_rag.utils.embedding import get_embedding_function
from src.config import KNOWLEDGE_LIGTH_GRAPH_PATH

async def load_gexf_to_storage(
    graph_storage: GraphStorage,
    entity_vector_storage: VectorStorage,
    relation_vector_storage: VectorStorage
):
    """
    Loads the existing GEXF file and populates:
    1. GraphStorage (already handled by init, but explicit check here)
    2. Entity VectorStorage (Nodes)
    3. Relation VectorStorage (Edges)
    """
    if not KNOWLEDGE_LIGTH_GRAPH_PATH.exists():
        print(f"Error: GEXF file not found at {KNOWLEDGE_LIGTH_GRAPH_PATH}")
        return

    print("Loading GEXF for vectorization...")
    graph = nx.read_gexf(KNOWLEDGE_LIGTH_GRAPH_PATH)
    embedding_fn = get_embedding_function()
    
    # 1. Process Nodes (Entities)
    print("Processing Entities (Nodes)...")
    node_ids = []
    node_embeddings = []
    node_metadatas = []
    
    nodes = list(graph.nodes(data=True))
    batch_size = 100
    
    for i in tqdm(range(0, len(nodes), batch_size), desc="Vectorizing Entities"):
        batch = nodes[i : i + batch_size]
        texts_to_embed = []
        
        for node_id, data in batch:
            # Format: "Name: Description"
            # Some nodes might lack description, handle gracefully
            desc = data.get("description", "")
            label = data.get("label", node_id)
            text = f"{label}: {desc}".strip()
            texts_to_embed.append(text)
            
            node_ids.append(node_id)
            node_metadatas.append({
                "entity_name": label,
                "type": data.get("type", "UNKNOWN"),
                "description": desc,
                "community": str(data.get("community", ""))
            })
            
        # Bulk embed
        embeddings = embedding_fn.embed_documents(texts_to_embed)
        node_embeddings.extend(embeddings)

    # Upsert Nodes to VectorDB
    # VectorStorage.upsert expects single item, but for bulk we might want to iterate
    # Or strict implementation of BaseVectorStorage might need loop.
    # To assume the previous VectorStorage implementation, we loop.
    print("Upserting Entities to VectorDB...")
    for i, nid in enumerate(tqdm(node_ids, desc="Saving Entities")):
        await entity_vector_storage.upsert(
            id=nid,
            embedding=node_embeddings[i],
            metadata=node_metadatas[i]
        )

    # 2. Process Edges (Relations)
    print("Processing Relations (Edges)...")
    edge_ids = []
    edge_embeddings = []
    edge_metadatas = []
    
    edges = list(graph.edges(data=True))
    
    for i in tqdm(range(0, len(edges), batch_size), desc="Vectorizing Relations"):
        batch = edges[i : i + batch_size]
        texts_to_embed = []
        
        for u, v, data in batch:
            relation_id = str(data.get("id", f"{u}-{v}"))
            relation_type = data.get("relation", "RELATED")
            desc = data.get("description", "")
            
            # Format: "Source -[Relation]-> Target: Description"
            text = f"{u} -[{relation_type}]-> {v}: {desc}".strip()
            texts_to_embed.append(text)
            
            edge_ids.append(relation_id)
            edge_metadatas.append({
                "source": u,
                "target": v,
                "relation": relation_type,
                "description": desc
            })
            
        embeddings = embedding_fn.embed_documents(texts_to_embed)
        edge_embeddings.extend(embeddings)

    print("Upserting Relations to VectorDB...")
    for i, eid in enumerate(tqdm(edge_ids, desc="Saving Relations")):
        await relation_vector_storage.upsert(
            id=eid,
            embedding=edge_embeddings[i],
            metadata=edge_metadatas[i]
        )

    print("GEXF Loading Complete!")
