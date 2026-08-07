"""
Turns schema/olist_schema.yaml into one embedding document per table:
table name + description + column names/descriptions, all as one text
block. This is what gets embedded -- not raw data rows.
"""

import yaml


def build_embedding_docs(schema_path: str = "Schema/olist_schema.yaml") -> list[dict]:
    with open(schema_path) as f:
        schema = yaml.safe_load(f)

    docs = []
    for table_name, table_info in schema["tables"].items():
        column_text = "\n".join(
            f"- {col}: {desc}" for col, desc in table_info.get("columns", {}).items()
        )
        content = (
            f"Table: {table_name}\n"
            f"Description: {table_info['description']}\n"
            f"Columns:\n{column_text}"
        )

        docs.append({
            "table_name": table_name,  # metadata, used to map back to real table later
            "content": content,        # the text that actually gets embedded
        })

    return docs


if __name__ == "__main__":
    for doc in build_embedding_docs():
        print(doc["content"])
        print("---")