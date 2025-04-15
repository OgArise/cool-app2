# knowledge_graph.py

from neo4j import GraphDatabase, basic_auth
from typing import List, Dict, Any
import config # To get DB credentials
import re # Import regular expressions for sanitization

# Global variable to hold the driver instance (singleton pattern)
driver = None

def get_driver():
    """
    Initializes and returns the Neo4j driver instance.
    Uses a global variable to ensure only one driver is created.
    """
    global driver
    if driver is None:
        if not config.NEO4J_URI or not config.NEO4J_USERNAME or not config.NEO4J_PASSWORD:
            print("ERROR: Neo4j connection details (URI, USERNAME, PASSWORD) missing in config/.env")
            return None
        try:
            print(f"Attempting to connect to Neo4j at: {config.NEO4J_URI}")
            driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=basic_auth(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            )
            driver.verify_connectivity()
            print("Neo4j connection successful.")
        except Exception as e:
            print(f"ERROR: Failed to connect to Neo4j: {e}")
            driver = None
    return driver

def close_driver():
    """Closes the Neo4j driver connection if it exists."""
    global driver
    if driver is not None:
        try:
            driver.close()
            driver = None
            print("Neo4j connection closed.")
        except Exception as e:
             print(f"Error closing Neo4j driver: {e}")

def _sanitize_label_or_type(name: str) -> str:
    """
    Sanitizes a string to be a valid Neo4j Label or Relationship Type.
    - Replaces non-alphanumeric characters with underscores.
    - Ensures it doesn't start with a number (prepends '_').
    - Converts to uppercase for relationship types (convention).
    - Returns 'OTHER' or 'RELATED_TO' if empty after sanitization.
    """
    if not name or not isinstance(name, str):
        return "OTHER" # Default label

    # Replace non-alphanumeric with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    # Prepend underscore if starts with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized

    # If somehow it became empty, return default
    if not sanitized:
        return "OTHER"

    return sanitized

def _run_cypher_query(tx, query: str, params: Dict = None):
    """
    Internal helper function to execute a Cypher query within a transaction.
    """
    if params:
        tx.run(query, **params)
    else:
        tx.run(query)

def update_knowledge_graph(extracted_data: Dict[str, List]):
    """
    Updates the Neo4j graph with extracted entities, risks, and relationships.
    Applies both `:Entity` and specific type labels (e.g., `:COMPANY`) to nodes.
    Uses MERGE operations to avoid duplicates. Assumes driver is initialized.

    Args:
        extracted_data: A dictionary containing 'entities', 'risks', and 'relationships' lists.

    Returns:
        True if the update was successful (or attempted), False otherwise.
    """
    db_driver = get_driver()
    if not db_driver:
        print("ERROR: Neo4j driver not available. Cannot update knowledge graph.")
        return False

    entities = extracted_data.get("entities", [])
    risks = extracted_data.get("risks", [])
    relationships = extracted_data.get("relationships", [])

    try:
        with db_driver.session() as session:
            # --- 1. Merge Entities with Multiple Labels ---
            # Constraints recommended in Neo4j for performance:
            # CREATE CONSTRAINT entity_key IF NOT EXISTS ON (e:Entity) ASSERT (e.name, e.type) IS NODE KEY;
            # CREATE CONSTRAINT risk_key IF NOT EXISTS ON (r:Risk) ASSERT r.description IS NODE KEY;
            print(f"Processing {len(entities)} entities...")
            for entity in entities:
                entity_name = entity.get("name")
                # Use 'OTHER' as default type if not specified or invalid
                entity_type_prop = entity.get("type", "OTHER") if isinstance(entity.get("type"), str) else "OTHER"

                if not entity_name or not isinstance(entity_name, str):
                    print("Warning: Skipping entity with missing or invalid name.")
                    continue

                # Sanitize the type property to use as a valid secondary label
                specific_label = _sanitize_label_or_type(entity_type_prop)

                # Use MERGE on the primary identifier (e.g., :Entity label + name/type props)
                # Then use SET to ensure *both* :Entity and the :SpecificLabel are present.
                # Using SET ensures the label is added even if the node already existed (ON MATCH).
                cypher = f"""
                MERGE (e:Entity {{name: $name, type: $type}})
                ON CREATE SET e.createdAt = timestamp(), e.lastSeen = timestamp(), e:`{specific_label}`
                ON MATCH SET e.lastSeen = timestamp(), e:`{specific_label}`
                """
                # Note: We MERGE on :Entity and properties, then SET the specific label (`e:`SpecificLabel``)
                # This is generally safer than trying to MERGE with dynamic labels directly.

                session.execute_write(
                    _run_cypher_query,
                    cypher,
                    params={"name": entity_name, "type": entity_type_prop}
                )

            # --- 2. Merge Risks and link to Entities ---
            print(f"Processing {len(risks)} risks...")
            for risk in risks:
                risk_desc = risk.get("description")
                risk_severity = risk.get("severity", "MEDIUM")
                if not risk_desc or not isinstance(risk_desc, str):
                    print("Warning: Skipping risk with missing or invalid description.")
                    continue

                # Merge the risk node (only label :Risk)
                session.execute_write(
                    _run_cypher_query,
                    "MERGE (r:Risk {description: $description}) "
                    "ON CREATE SET r.severity = $severity, r.createdAt = timestamp(), r.lastSeen = timestamp() "
                    "ON MATCH SET r.lastSeen = timestamp(), r.severity = $severity",
                    params={"description": risk_desc, "severity": risk_severity}
                )

                # Link risk to related entities (match entities by name/type properties)
                for entity_name in risk.get("related_entities", []):
                    if not entity_name or not isinstance(entity_name, str): continue
                    session.execute_write(
                        _run_cypher_query,
                        # Match entity using the :Entity label and properties
                        "MATCH (r:Risk {description: $riskDesc}) "
                        "MATCH (e:Entity {name: $entityName}) " # Assume name is unique enough for linking here
                        "MERGE (e)-[:ASSOCIATED_WITH]->(r)",
                        params={"riskDesc": risk_desc, "entityName": entity_name}
                    )

            # --- 3. Merge Relationships ---
            print(f"Processing {len(relationships)} relationships...")
            for rel in relationships:
                entity1_name = rel.get("entity1")
                rel_type_raw = rel.get("relationship_type")
                entity2_name = rel.get("entity2")

                if not entity1_name or not rel_type_raw or not entity2_name or \
                   not isinstance(entity1_name, str) or not isinstance(rel_type_raw, str) or not isinstance(entity2_name, str):
                    print(f"Warning: Skipping invalid relationship structure: {rel}")
                    continue

                # Sanitize relationship type (uppercase, alphanumeric/underscore)
                rel_type_sanitized = _sanitize_label_or_type(rel_type_raw).upper() # Use uppercase convention for relationships
                if not rel_type_sanitized or rel_type_sanitized == "OTHER": # Check if sanitization failed or yielded default
                    rel_type_sanitized = "RELATED_TO" # Default relationship type

                # Match entities using :Entity label and name property
                # MERGE the relationship between them
                session.execute_write(
                    _run_cypher_query,
                    f"""
                    MATCH (a:Entity {{name: $entity1}})
                    MATCH (b:Entity {{name: $entity2}})
                    MERGE (a)-[rel:`{rel_type_sanitized}`]->(b)
                    ON CREATE SET rel.createdAt = timestamp(), rel.lastSeen = timestamp()
                    ON MATCH SET rel.lastSeen = timestamp()
                    """,
                    params={"entity1": entity1_name, "entity2": entity2_name}
                )
        print("Knowledge graph update process completed in session.")
        return True
    except Exception as e:
        print(f"ERROR: Failed during knowledge graph update: {e}")
        import traceback
        traceback.print_exc() # Print full traceback to logs for better debugging
        return False