CREATE CONSTRAINT kg_term_name IF NOT EXISTS
FOR (term:Term)
REQUIRE term.name IS UNIQUE;

CREATE CONSTRAINT kg_alias_name IF NOT EXISTS
FOR (alias:Alias)
REQUIRE alias.name IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///sample_kg.csv' AS row
WITH row
WHERE row.term IS NOT NULL AND trim(row.term) <> ''
MERGE (term:Term {name: trim(row.term)})
SET term.relation = coalesce(row.relation, ''),
    term.expansion = coalesce(row.expansion, ''),
    term.aliases = CASE
      WHEN row.aliases IS NULL OR trim(row.aliases) = '' THEN []
      ELSE split(row.aliases, '|')
    END;

LOAD CSV WITH HEADERS FROM 'file:///sample_kg.csv' AS row
WITH row
WHERE row.aliases IS NOT NULL AND trim(row.aliases) <> ''
MATCH (term:Term {name: trim(row.term)})
UNWIND split(row.aliases, '|') AS alias_name
WITH term, trim(alias_name) AS alias_name
WHERE alias_name <> ''
MERGE (alias:Alias {name: alias_name})
MERGE (alias)-[:ALIASES]->(term);
