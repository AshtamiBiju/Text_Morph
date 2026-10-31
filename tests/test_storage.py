import sqlite3

from textmorph.storage import DocumentStore


def test_existing_schema_migrates_heading_and_nullable_evidence_score(tmp_path):
    database = tmp_path / "metadata.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE evidence_records (
            id VARCHAR(36) NOT NULL PRIMARY KEY,
            collection_id VARCHAR(36) NOT NULL,
            document_id VARCHAR(36) NOT NULL,
            chunk_id VARCHAR(36) NOT NULL,
            source_text TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            retrieval_score FLOAT NOT NULL,
            created_at DATETIME NOT NULL
        );
        PRAGMA user_version=1;
        """
    )
    connection.close()

    store = DocumentStore(tmp_path)
    with store.engine.connect() as connection:
        version = connection.exec_driver_sql("PRAGMA user_version").scalar_one()
        chunk_columns = {
            row[1] for row in connection.exec_driver_sql("PRAGMA table_info(chunks)")
        }
        evidence_columns = {
            row[1]: row for row in connection.exec_driver_sql("PRAGMA table_info(evidence_records)")
        }
    assert version == 2
    assert "heading" in chunk_columns
    assert evidence_columns["retrieval_score"][3] == 0
