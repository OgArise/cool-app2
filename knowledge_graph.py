# knowledge_graph.py

from typing import List, Dict, Any
import config
import re
import traceback

# Define variables for optional imports, initialized to None
GraphDatabase = None
basic_auth = None
neo4j_library_available = False

# Setup function to handle imports within a function scope
def _setup_kg_imports():
    global GraphDatabase, basic_auth, neo4j_library_available
    try:
        from neo4j import GraphDatabase, basic_auth
        neo4j_library_available = True
        print("Neo4j library imported successfully.")
        return GraphDatabase, basic_auth, neo4j_library_available
    except ImportError:
        print("Warning: Neo4j library not installed. Knowledge graph functionality disabled.")
        return None, None, False

# Call the setup function at the end of the module to populate module-level variables
GraphDatabase, basic_auth, neo4j_library_available = _setup_kg_imports()

driver = None

def get_driver():
    """
    Initializes and returns the Neo4j driver instance.
    Uses a global variable to ensure only one driver is created.
    Checks for library availability at runtime.
    """
    global driver
    if not neo4j_library_available or GraphDatabase is None or basic_auth is None:
         print("Neo4j library not available.")
         return None

    if driver is None:
        if not config or not getattr(config, 'NEO4J_URI', None) or not getattr(config, 'NEO4J_USERNAME', None) or not getattr(config, 'NEO4J_PASSWORD', None):
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
            traceback.print_exc()
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
        return "OTHER"

    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized

    # Ensure it's not empty after sanitization
    if not sanitized:
        return "OTHER"

    return sanitized.upper() # Convert to uppercase for relationship types by convention

def _run_cypher_query(tx, query: str, params: Dict = None):
    """Internal helper function to execute a Cypher query within a transaction."""
    if params:
        # print(f"Running Cypher (params): {query} with {params}") # Debug print
        tx.run(query, **params)
    else:
        # print(f"Running Cypher (no params): {query}") # Debug print
        tx.run(query)

def update_knowledge_graph(extracted_data: Dict[str, List]):
    """
    Updates the Neo4j graph with extracted entities, risks, and relationships.
    Applies primary labels (e.g., :Company, :Risk, :RegulatoryAgency, :Sanction)
    and potentially secondary type labels. Uses MERGE operations.
    Checks for driver and library availability.
    Uses specific node labels when creating relationships.

    Args:
        extracted_data: A dictionary containing 'entities', 'risks', and 'relationships' lists.

    Returns:
        True if the update was successful (or attempted), False otherwise.
    """
    # Check for library availability at the function entry point
    if not neo4j_library_available:
         print("Neo4j library not available. Cannot update knowledge graph.")
         return False

    db_driver = get_driver()
    if not db_driver:
        print("ERROR: Neo4j driver not available. Cannot update knowledge graph.")
        return False

    entities = extracted_data.get("entities", [])
    risks = extracted_data.get("risks", [])
    relationships = extracted_data.get("relationships", [])

    # Map extracted types to Neo4j primary labels
    type_to_label_map = {
        "COMPANY": "Company",
        "ORGANIZATION": "Organization",
        "REGULATORY_AGENCY": "RegulatoryAgency",
        "SANCTION": "Sanction",
        "RISK": "Risk" # Risks are extracted separately but can have entities linked to them
    }

    try:
        with db_driver.session(database="neo4j") as session:
            # --- 1. Merge Entities with Primary Labels ---
            print(f"Processing {len(entities)} entities for KG update...")
            for entity in entities:
                entity_name = entity.get("name")
                entity_type_raw = entity.get("type", "OTHER") if isinstance(entity.get("type"), str) else "OTHER"
                if not entity_name or not isinstance(entity_name, str): print("Warning: Skipping entity with missing/invalid name."); continue

                primary_label = type_to_label_map.get(entity_type_raw.upper(), "Entity")

                # Use SET to add the primary label and the generic :Entity label
                cypher = f"""
                MERGE (e:{primary_label} {{name: $name}})
                ON CREATE SET e.createdAt = timestamp(), e.lastSeen = timestamp()
                ON MATCH SET e.lastSeen = timestamp()
                SET e.type = $type_raw, e:Entity // Set the type property and add the generic :Entity label for broader matching
                """
                session.execute_write( _run_cypher_query, cypher, params={"name": entity_name, "type_raw": entity_type_raw} )

            # --- 2. Merge Risks and link to Entities ---
            # Risks are merged first, then linked to entities
            print(f"Processing {len(risks)} risks for KG update...")
            for risk in risks:
                risk_desc = risk.get("description"); risk_severity = risk.get("severity", "MEDIUM")
                if not risk_desc or not isinstance(risk_desc, str): print("Warning: Skipping risk with missing/invalid description."); continue

                # Merge Risk node
                session.execute_write( _run_cypher_query, """
                MERGE (r:Risk {description: $description})
                ON CREATE SET r.severity = $severity, r.createdAt = timestamp(), r.lastSeen = timestamp()
                ON MATCH SET r.lastSeen = timestamp(), r.severity = $severity
                """, params={"description": risk_desc, "severity": risk_severity} )

                # Link related Entities to Risk
                # Match entities by name and ANY of the specific extracted labels
                for entity_name in risk.get("related_entities", []):
                    if not entity_name or not isinstance(entity_name, str): continue
                    # Find entities that have one of our specific labels
                    session.execute_write( _run_cypher_query, """
                    MATCH (r:Risk {description: $riskDesc})
                    MATCH (e) WHERE e.name = $entityName AND (e:Company OR e:Organization OR e:RegulatoryAgency OR e:Sanction)
                    MERGE (e)-[:ASSOCIATED_WITH]->(r)
                    """, params={"riskDesc": risk_desc, "entityName": entity_name} )


            # --- 3. Merge Relationships ---
            print(f"Processing {len(relationships)} relationships for KG update...")
            for rel in relationships:
                entity1_name = rel.get("entity1"); rel_type_raw = rel.get("relationship_type"); entity2_name = rel.get("entity2")
                if not entity1_name or not rel_type_raw or not entity2_name or not isinstance(entity1_name, str) or not isinstance(rel_type_raw, str) or not isinstance(entity2_name, str):
                    print(f"Warning: Skipping invalid relationship structure: {rel}"); continue

                rel_type_sanitized = _sanitize_label_or_type(rel_type_raw)
                if not rel_type_sanitized or rel_type_sanitized == "OTHER":
                    print(f"Warning: Skipping relationship with unhandled or empty sanitized type: {rel_type_raw} -> {rel_type_sanitized}"); continue


                # Define specific match patterns based on relationship type
                # This makes KG structure more rigid and correct according to expected types
                if rel_type_sanitized in ["PARENT_COMPANY_OF", "SUBSIDIARY_OF", "AFFILIATE_OF", "ACQUIRED", "RELATED_COMPANY", "JOINT_VENTURE_PARTNER", "INVESTEE"]:
                     # Ownership/Corporate relationships - expect Company or Organization nodes
                     cypher = f"""
                     MATCH (a) WHERE a.name = $entity1 AND (a:Company OR a:Organization)
                     MATCH (b) WHERE b.name = $entity2 AND (b:Company OR b:Organization)
                     MERGE (a)-[rel:`{rel_type_sanitized}`]->(b)
                     ON CREATE SET rel.createdAt = timestamp(), rel.lastSeen = timestamp()
                     ON MATCH SET rel.lastSeen = timestamp()
                     """
                     session.execute_write( _run_cypher_query, cypher, params={"entity1": entity1_name, "entity2": entity2_name} )

                elif rel_type_sanitized == "REGULATED_BY":
                     # Company/Org REGULATED_BY Regulatory Agency
                     cypher = f"""
                     MATCH (a) WHERE a.name = $entity1 AND (a:Company OR a:Organization)
                     MATCH (b:RegulatoryAgency {{name: $entity2}})
                     MERGE (a)-[rel:`REGULATED_BY`]->(b)
                     ON CREATE SET rel.createdAt = timestamp(), rel.lastSeen = timestamp()
                     ON MATCH SET rel.lastSeen = timestamp()
                     """
                     session.execute_write( _run_cypher_query, cypher, params={"entity1": entity1_name, "entity2": entity2_name} )

                elif rel_type_sanitized == "ISSUED_BY":
                     # Sanction ISSUED_BY Regulatory Agency
                     cypher = f"""
                     MATCH (a:Sanction {{name: $entity1}})
                     MATCH (b:RegulatoryAgency {{name: $entity2}})
                     MERGE (a)-[rel:`ISSUED_BY`]->(b)
                     ON CREATE SET rel.createdAt = timestamp(), rel.lastSeen = timestamp()
                     ON MATCH SET rel.lastSeen = timestamp()
                     """
                     session.execute_write( _run_cypher_query, cypher, params={"entity1": entity1_name, "entity2": entity2_name} )

                elif rel_type_sanitized == "SUBJECT_TO":
                     # Company/Org SUBJECT_TO Sanction/RegulatoryAgency
                     cypher = f"""
                     MATCH (a) WHERE a.name = $entity1 AND (a:Company OR a:Organization)
                     MATCH (b) WHERE b.name = $entity2 AND (b:Sanction OR b:RegulatoryAgency)
                     MERGE (a)-[rel:`SUBJECT_TO`]->(b)
                     ON CREATE SET rel.createdAt = timestamp(), rel.lastSeen = timestamp()
                     ON MATCH SET rel.lastSeen = timestamp()
                     """
                     session.execute_write( _run_cypher_query, cypher, params={"entity1": entity1_name, "entity2": entity2_name} )

                elif rel_type_sanitized == "MENTIONED_WITH":
                     # Broader relationship, can be between any two extracted entity types
                     cypher = f"""
                     MATCH (a) WHERE a.name = $entity1 AND (a:Company OR a:Organization OR a:RegulatoryAgency OR a:Sanction)
                     MATCH (b) WHERE b.name = $entity2 AND (b:Company OR b:Organization OR b:RegulatoryAgency OR b:Sanction)
                     MERGE (a)-[rel:`MENTIONED_WITH`]->(b)
                     ON CREATE SET rel.createdAt = timestamp(), rel.lastSeen = timestamp()
                     ON MATCH SET rel.lastSeen = timestamp()
                     """
                     session.execute_write( _run_cypher_query, cypher, params={"entity1": entity1_name, "entity2": entity2_name} )

                else:
                    print(f"Warning: Skipping relationship with unknown or unhandled sanitized type for specific matching: {rel_type_raw} -> {rel_type_sanitized}")
                    # Optional: Fallback to a generic RELATED_TO if needed, but specific is better
                    # cypher = """
                    # MATCH (a {name: $entity1})
                    # MATCH (b {name: $entity2})
                    # MERGE (a)-[rel:RELATED_TO]->(b)
                    # ON CREATE SET rel.createdAt = timestamp(), rel.lastSeen = timestamp(), rel.type = $type_raw
                    # ON MATCH SET rel.lastSeen = timestamp()
                    # """
                    # session.execute_write( _run_cypher_query, cypher, params={"entity1": entity1_name, "entity2": entity2_name, "type_raw": rel_type_raw} )


        print("Knowledge graph update process completed in session.")
        return True
    except Exception as e:
        print(f"ERROR: Failed during knowledge graph update: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n--- Running Local Knowledge Graph Tests ---")
    print("NOTE: Requires Neo4j database configured in .env and running.")

    test_driver = get_driver()

    if test_driver:
        print("\nTesting KG update with sample data (including new types)...")
        sample_extracted_data = {
            "entities": [
                {"name": "TestCompanyA", "type": "COMPANY", "mentions": ["url1", "url2"]},
                {"name": "TestCompanyB", "type": "ORGANIZATION", "mentions": ["url1"]},
                {"name": "TestAgencyC", "type": "REGULATORY_AGENCY", "mentions": ["url3"]},
                {"name": "TestSanctionD", "type": "SANCTION", "mentions": ["url4"]},
                # Added a Risk entity example directly here - normally risks are separate
                # {"name": "Sample Risk Description", "type": "RISK", "description": "Sample Risk Description", "severity": "HIGH", "source_urls": ["url5"], "related_entities": ["TestCompanyA"]}
            ],
            "risks": [
                {"description": "Compliance violation risk", "severity": "SEVERE", "source_urls": ["url1", "url3"], "related_entities": ["TestCompanyA"]},
                {"description": "Environmental penalty risk", "severity": "HIGH", "source_urls": ["url2"], "related_entities": ["TestCompanyB"]}
                 # Test linking risk to Agency
                # {"description": "New Regulation Risk", "severity": "MEDIUM", "source_urls": ["url6"], "related_entities": ["TestAgencyC"]}
            ],
            "relationships": [
                {"entity1": "TestCompanyA", "relationship_type": "SUBSIDIARY_OF", "entity2": "TestCompanyB", "context_urls": ["url1"]},
                {"entity1": "TestCompanyA", "relationship_type": "REGULATED_BY", "entity2": "TestAgencyC", "context_urls": ["url3"]},
                {"entity1": "TestSanctionD", "relationship_type": "ISSUED_BY", "entity2": "TestAgencyC", "context_urls": ["url4"]},
                {"entity1": "TestCompanyA", "relationship_type": "SUBJECT_TO", "entity2": "TestSanctionD", "context_urls": ["url4"]},
                {"entity1": "TestCompanyA", "relationship_type": "MENTIONED_WITH", "entity2": "TestCompanyB", "context_urls": ["url1"]},
                # Example of linking a Company to a Sanction (also covered by SUBJECT_TO, but MENTIONED_WITH is broader)
                {"entity1": "TestCompanyA", "relationship_type": "MENTIONED_WITH", "entity2": "TestSanctionD", "context_urls": ["url7"]},
                # Example of linking an Agency to a Sanction (also covered by ISSUED_BY, but MENTIONED_WITH is broader)
                 {"entity1": "TestAgencyC", "relationship_type": "MENTIONED_WITH", "entity2": "TestSanctionD", "context_urls": ["url8"]}
                # Example of linking an Agency to a Company (could be covered by REGULATED_BY, but MENTIONED_WITH is broader)
                # {"entity1": "TestAgencyC", "relationship_type": "MENTIONED_WITH", "entity2": "TestCompanyA", "context_urls": ["url9"]}
            ]
        }

        update_status = update_knowledge_graph(sample_extracted_data)
        print(f"\nKG Update Status: {'Success' if update_status else 'Failed'}")

        close_driver()
    else:
        print("\nSkipping KG tests: Driver not available.")

    print("\n--- Local Knowledge Graph Tests Complete ---")