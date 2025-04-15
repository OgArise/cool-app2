# knowledge_graph.py

from neo4j import GraphDatabase, basic_auth
from typing import List, Dict, Any
import config # To get DB credentials

# Global variable to hold the driver instance (singleton pattern)
driver = None

def get_driver():
    """
    Initializes and returns the Neo4j driver instance.
    Uses a global variable to ensure only one driver is created.
    """
    global driver
    if driver is None:
        # Check if configuration details are present
        if not config.NEO4J_URI or not config.NEO4J_USERNAME or not config.NEO4J_PASSWORD:
            print("ERROR: Neo4j connection details (URI, USERNAME, PASSWORD) missing in config/.env")
            return None # Return None if config is missing
        try:
            # Attempt to create the driver instance
            print(f"Attempting to connect to Neo4j at: {config.NEO4J_URI}")
            driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=basic_auth(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            )
            # Verify connection is working
            driver.verify_connectivity()
            print("Neo4j connection successful.")
        except Exception as e:
            print(f"ERROR: Failed to connect to Neo4j: {e}")
            driver = None # Ensure driver is None if connection fails
    return driver

def close_driver():
    """Closes the Neo4j driver connection if it exists."""
    global driver
    if driver is not None:
        try:
            driver.close()
            driver = None # Reset the global variable
            print("Neo4j connection closed.")
        except Exception as e:
             print(f"Error closing Neo4j driver: {e}")


def _run_cypher_query(tx, query: str, params: Dict = None):
    """
    Internal helper function to execute a Cypher query within a transaction.
    Args:
        tx: The Neo4j transaction object.
        query: The Cypher query string.
        params: A dictionary of parameters for the query.
    """
    if params:
        tx.run(query, **params) # Unpack params dictionary
    else:
        tx.run(query)
    # No need to return results for simple MERGE/CREATE in this context
    # If you needed to read data, you would process the result here


def update_knowledge_graph(extracted_data: Dict[str, List]):
    """
    Updates the Neo4j graph with extracted entities, risks, and relationships.
    Uses MERGE operations to avoid duplicates. Assumes driver is initialized.

    Args:
        extracted_data: A dictionary containing 'entities', 'risks', and 'relationships' lists.

    Returns:
        True if the update was successful (or attempted), False otherwise.
    """
    db_driver = get_driver() # Get the initialized driver instance
    if not db_driver:
        print("ERROR: Neo4j driver not available. Cannot update knowledge graph.")
        return False # Cannot proceed without DB connection

    entities = extracted_data.get("entities", [])
    risks = extracted_data.get("risks", [])
    relationships = extracted_data.get("relationships", [])

    # Best practice: Use a session for operations
    try:
        # Use execute_write for operations that modify the graph
        with db_driver.session() as session:
            # --- 1. Merge Entities ---
            # It's highly recommended to create constraints in Neo4j for performance:
            # CREATE CONSTRAINT entity_key IF NOT EXISTS ON (e:Entity) ASSERT (e.name, e.type) IS NODE KEY;
            print(f"Processing {len(entities)} entities...")
            for entity in entities:
                entity_name = entity.get("name")
                entity_type = entity.get("type", "OTHER") # Default type if missing
                if not entity_name:
                    print("Warning: Skipping entity with no name.")
                    continue
                session.execute_write(
                    _run_cypher_query,
                    # Using backticks `` for the label `Entity` allows spaces or special chars if ever needed, though simple names are better
                    "MERGE (e:`Entity` {name: $name, type: $type}) "
                    "ON CREATE SET e.createdAt = timestamp(), e.lastSeen = timestamp() "
                    "ON MATCH SET e.lastSeen = timestamp()",
                    params={"name": entity_name, "type": entity_type}
                )

            # --- 2. Merge Risks and link to Entities ---
            # Consider creating a constraint: CREATE CONSTRAINT risk_key IF NOT EXISTS ON (r:Risk) ASSERT r.description IS NODE KEY;
            print(f"Processing {len(risks)} risks...")
            for risk in risks:
                risk_desc = risk.get("description")
                risk_severity = risk.get("severity", "MEDIUM") # Default severity
                if not risk_desc:
                    print("Warning: Skipping risk with no description.")
                    continue
                # Merge the risk node
                session.execute_write(
                    _run_cypher_query,
                    "MERGE (r:Risk {description: $description}) "
                    "ON CREATE SET r.severity = $severity, r.createdAt = timestamp(), r.lastSeen = timestamp() "
                    "ON MATCH SET r.lastSeen = timestamp(), r.severity = $severity", # Update severity if risk already exists
                    params={"description": risk_desc, "severity": risk_severity}
                )
                # Link risk to related entities mentioned
                for entity_name in risk.get("related_entities", []):
                    if not entity_name: continue # Skip empty names
                    session.execute_write(
                        _run_cypher_query,
                        # Match based on name only here, assumes entity was created above
                        # If types are important for matching, adjust MATCH clause
                        "MATCH (r:Risk {description: $riskDesc}) "
                        "MATCH (e:Entity {name: $entityName}) "
                        "MERGE (e)-[:ASSOCIATED_WITH]->(r)", # Relationship from Entity TO Risk
                        params={"riskDesc": risk_desc, "entityName": entity_name}
                    )

            # --- 3. Merge Relationships ---
            print(f"Processing {len(relationships)} relationships...")
            for rel in relationships:
                entity1 = rel.get("entity1")
                rel_type_raw = rel.get("relationship_type")
                entity2 = rel.get("entity2")

                if not entity1 or not rel_type_raw or not entity2:
                    print(f"Warning: Skipping invalid relationship: {rel}")
                    continue

                # Sanitize relationship type: Convert to UPPERCASE, replace non-alphanumeric with _, ensure it's not empty
                rel_type_sanitized = ''.join(c if c.isalnum() else '_' for c in rel_type_raw.upper())
                if not rel_type_sanitized:
                    print(f"Warning: Skipping relationship with invalid type after sanitization: {rel_type_raw}")
                    continue

                session.execute_write(
                    _run_cypher_query,
                    # Use backticks `` to allow sanitized relationship types
                    f"MATCH (a:Entity {{name: $entity1}}) "
                    f"MATCH (b:Entity {{name: $entity2}}) "
                    f"MERGE (a)-[rel:`{rel_type_sanitized}`]->(b) "
                    f"ON CREATE SET rel.createdAt = timestamp(), rel.lastSeen = timestamp() "
                    f"ON MATCH SET rel.lastSeen = timestamp()",
                    params={"entity1": entity1, "entity2": entity2}
                )
        print("Knowledge graph update process completed in session.")
        return True # Indicate success
    except Exception as e:
        print(f"ERROR: Failed during knowledge graph update: {e}")
        # Optionally re-raise the exception if the caller should handle it more specifically
        # raise e
        return False # Indicate failure
    # Note: The driver itself is closed by calling close_driver() later in the orchestrator's finally block