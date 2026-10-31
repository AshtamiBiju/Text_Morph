import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, create_engine, event, func, inspect, select
from sqlalchemy.orm import DeclarativeBase, Mapped, joinedload, mapped_column, relationship, sessionmaker

from textmorph.config import DATA_DIR, MAX_CHUNKS_PER_COLLECTION, MAX_DOCUMENTS_PER_COLLECTION


class Base(DeclarativeBase):
    pass


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    documents: Mapped[list["Document"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan", passive_deletes=True
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("collection_id", "sha256", name="uq_document_collection_hash"),
        Index("ix_documents_collection", "collection_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="indexed_keyword_only", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    collection: Mapped[Collection] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (Index("ix_chunks_collection", "collection_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    paragraph_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    heading: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    document: Mapped[Document] = relationship(back_populates="chunks")


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("collection_id", "name", name="uq_entity_collection_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)


class Relationship(Base):
    __tablename__ = "relationships"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    source_entity_id: Mapped[str] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    target_entity_id: Mapped[str] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    relationship_type: Mapped[str] = mapped_column(String(64), nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    evidence_sentence: Mapped[str] = mapped_column(Text, nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(64), nullable=False)


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    analysis_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class QAHistory(Base):
    __tablename__ = "qa_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_ms: Mapped[float] = mapped_column(nullable=False)
    generation_ms: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class EvidenceRecord(Base):
    __tablename__ = "evidence_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class DocumentStore:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        database_path = self.data_dir / "metadata.sqlite"
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{database_path}", connect_args={"check_same_thread": False})

        @event.listens_for(self.engine, "connect")
        def _enable_foreign_keys(connection, _record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self._session = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        self._migrate_schema()

    def _migrate_schema(self):
        with self.engine.connect() as connection:
            current_version = int(connection.exec_driver_sql("PRAGMA user_version").scalar_one())
        if current_version > 2:
            raise RuntimeError(
                f"Database schema version {current_version} is newer than this TextMorph version supports."
            )
        inspector = inspect(self.engine)
        chunk_columns = {column["name"] for column in inspector.get_columns("chunks")}
        with self.engine.begin() as connection:
            if "heading" not in chunk_columns:
                connection.exec_driver_sql("ALTER TABLE chunks ADD COLUMN heading VARCHAR(255)")
            evidence_columns = {
                row[1]: row for row in connection.exec_driver_sql("PRAGMA table_info(evidence_records)")
            }
            retrieval_score = evidence_columns.get("retrieval_score")
            if retrieval_score and retrieval_score[3] == 1:
                connection.exec_driver_sql("ALTER TABLE evidence_records RENAME TO evidence_records_v1")
                connection.exec_driver_sql(
                    "CREATE TABLE evidence_records ("
                    "id VARCHAR(36) NOT NULL PRIMARY KEY, "
                    "collection_id VARCHAR(36) NOT NULL REFERENCES collections(id) ON DELETE CASCADE, "
                    "document_id VARCHAR(36) NOT NULL REFERENCES documents(id) ON DELETE CASCADE, "
                    "chunk_id VARCHAR(36) NOT NULL REFERENCES chunks(id) ON DELETE CASCADE, "
                    "source_text TEXT NOT NULL, start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL, "
                    "retrieval_score FLOAT, created_at DATETIME NOT NULL)"
                )
                connection.exec_driver_sql(
                    "INSERT INTO evidence_records "
                    "SELECT id, collection_id, document_id, chunk_id, source_text, start_offset, end_offset, "
                    "retrieval_score, created_at FROM evidence_records_v1"
                )
                connection.exec_driver_sql("DROP TABLE evidence_records_v1")
            connection.exec_driver_sql("PRAGMA user_version=2")

    def create_collection(self, name: str) -> Collection:
        clean_name = " ".join(name.split())
        if not clean_name or len(clean_name) > 120:
            raise ValueError("Collection names must contain 1–120 characters.")
        collection = Collection(name=clean_name)
        with self._session.begin() as session:
            if session.scalar(select(Collection).where(Collection.name == clean_name)):
                raise ValueError("A collection with that name already exists.")
            session.add(collection)
        return collection

    def rename_collection(self, collection_id: str, name: str) -> Collection:
        clean_name = " ".join(name.split())
        if not clean_name or len(clean_name) > 120:
            raise ValueError("Collection names must contain 1–120 characters.")
        with self._session.begin() as session:
            collection = session.get(Collection, collection_id)
            if not collection:
                raise ValueError("Collection not found.")
            duplicate = session.scalar(select(Collection).where(Collection.name == clean_name, Collection.id != collection_id))
            if duplicate:
                raise ValueError("A collection with that name already exists.")
            collection.name = clean_name
        return collection

    def list_collections(self):
        with self._session() as session:
            return list(session.scalars(select(Collection).order_by(Collection.created_at)))

    def get_collection(self, collection_id: str) -> Optional[Collection]:
        with self._session() as session:
            return session.get(Collection, collection_id)

    def document_count(self, collection_id: str) -> int:
        with self._session() as session:
            collection = session.get(Collection, collection_id)
            if not collection:
                raise ValueError("Collection not found.")
            return len(collection.documents)

    def list_documents(self, collection_id: str):
        with self._session() as session:
            if not session.get(Collection, collection_id):
                raise ValueError("Collection not found.")
            return list(
                session.scalars(
                    select(Document)
                    .options(joinedload(Document.chunks))
                    .where(Document.collection_id == collection_id)
                    .order_by(Document.created_at)
                ).unique()
            )

    def get_chunks(self, collection_id: str):
        with self._session() as session:
            return list(
                session.scalars(
                    select(Chunk)
                    .options(joinedload(Chunk.document))
                    .where(Chunk.collection_id == collection_id)
                )
            )

    def chunk_count(self, collection_id: str) -> int:
        with self._session() as session:
            return int(session.scalar(select(func.count()).select_from(Chunk).where(Chunk.collection_id == collection_id)) or 0)

    def register_document(
        self,
        collection_id: str,
        filename: str,
        sha256: str,
        file_path: Path,
        file_type: str,
        page_count: Optional[int],
        chunks: Iterable[Dict],
    ) -> Document:
        with self._session.begin() as session:
            collection = session.get(Collection, collection_id)
            if not collection:
                raise ValueError("Collection not found.")
            chunk_rows = list(chunks)
            if len(collection.documents) >= MAX_DOCUMENTS_PER_COLLECTION:
                raise ValueError(f"This collection has reached the {MAX_DOCUMENTS_PER_COLLECTION}-document limit.")
            existing_chunks = int(
                session.scalar(
                    select(func.count()).select_from(Chunk).where(Chunk.collection_id == collection_id)
                ) or 0
            )
            if existing_chunks + len(chunk_rows) > MAX_CHUNKS_PER_COLLECTION:
                raise ValueError(f"This collection has reached the {MAX_CHUNKS_PER_COLLECTION}-chunk limit.")
            duplicate = session.scalar(
                select(Document).where(
                    Document.collection_id == collection_id,
                    Document.sha256 == sha256,
                )
            )
            if duplicate:
                raise ValueError(f"{filename} is already in this collection.")
            document = Document(
                collection_id=collection_id,
                filename=filename,
                sha256=sha256,
                file_path=str(file_path.resolve()),
                file_type=file_type,
                page_count=page_count,
                status="needs_embedding",
            )
            session.add(document)
            session.flush()
            for item in chunk_rows:
                session.add(
                    Chunk(
                        collection_id=collection_id,
                        document_id=document.id,
                        page_number=item.get("page_number"),
                        paragraph_number=item.get("paragraph_number"),
                        heading=item.get("heading"),
                        start_offset=item["start_offset"],
                        end_offset=item["end_offset"],
                        source_text=item["source_text"],
                    )
                )
        return document

    def set_document_status(self, collection_id: str, document_id: str, status: str):
        with self._session.begin() as session:
            document = session.scalar(
                select(Document).where(Document.collection_id == collection_id, Document.id == document_id)
            )
            if not document:
                raise ValueError("Document not found in this collection.")
            document.status = status

    def save_embeddings(self, collection_id: str, embeddings: Dict[str, Iterable[float]], model_id: str):
        self.save_embedding_batch(collection_id, embeddings, model_id)
        self.mark_collection_indexed(collection_id, model_id)

    def save_embedding_batch(self, collection_id: str, embeddings: Dict[str, Iterable[float]], model_id: str):
        with self._session.begin() as session:
            chunks = session.scalars(
                select(Chunk).where(
                    Chunk.collection_id == collection_id,
                    Chunk.id.in_(embeddings),
                )
            ).all()
            if not chunks or {chunk.id for chunk in chunks} != set(embeddings):
                raise ValueError("Embedding data did not match chunks in the selected collection.")
            for chunk in chunks:
                chunk.embedding_json = encode_embedding(embeddings[chunk.id])
                chunk.embedding_model = model_id

    def mark_collection_indexed(self, collection_id: str, model_id: str):
        with self._session.begin() as session:
            chunks = session.scalars(select(Chunk).where(Chunk.collection_id == collection_id)).all()
            if not chunks or any(
                not chunk.embedding_json or chunk.embedding_model != model_id for chunk in chunks
            ):
                raise ValueError("Every chunk must be indexed with the same model before marking the collection indexed.")
            session.query(Document).filter(Document.collection_id == collection_id).update(
                {Document.status: "indexed"}, synchronize_session=False
            )

    def set_collection_document_status(self, collection_id: str, status: str):
        with self._session.begin() as session:
            if not session.get(Collection, collection_id):
                raise ValueError("Collection not found.")
            session.query(Document).filter(Document.collection_id == collection_id).update(
                {Document.status: status}, synchronize_session=False
            )

    def save_qa(self, collection_id: str, question: str, answer: str, evidence: Iterable[Dict], retrieval_ms: float, generation_ms: float):
        with self._session.begin() as session:
            if not session.get(Collection, collection_id):
                raise ValueError("Collection not found.")
            session.add(QAHistory(
                collection_id=collection_id,
                question=question,
                answer=answer,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
            ))
            for item in evidence:
                self._validate_evidence_record(session, collection_id, item)
                session.add(EvidenceRecord(
                    collection_id=collection_id,
                    document_id=item["document_id"],
                    chunk_id=item["chunk_id"],
                    source_text=item["source_text"],
                    start_offset=item["start_offset"],
                    end_offset=item["end_offset"],
                    retrieval_score=item["retrieval_score"],
                ))

    @staticmethod
    def _validate_evidence_record(session, collection_id: str, item: Dict):
        chunk = session.get(Chunk, item["chunk_id"])
        if (
            chunk is None
            or chunk.collection_id != collection_id
            or chunk.document_id != item["document_id"]
            or chunk.source_text != item["source_text"]
            or chunk.start_offset != item["start_offset"]
            or chunk.end_offset != item["end_offset"]
        ):
            raise ValueError("Evidence does not match a stored chunk in this collection.")

    def save_evidence(self, collection_id: str, records: Iterable[Dict]):
        with self._session.begin() as session:
            if not session.get(Collection, collection_id):
                raise ValueError("Collection not found.")
            for item in records:
                self._validate_evidence_record(session, collection_id, item)
                session.add(EvidenceRecord(
                    collection_id=collection_id,
                    document_id=item["document_id"],
                    chunk_id=item["chunk_id"],
                    source_text=item["source_text"],
                    start_offset=item["start_offset"],
                    end_offset=item["end_offset"],
                    retrieval_score=item.get("retrieval_score"),
                ))

    def save_analysis(self, collection_id: str, analysis_type: str, status: str, result: Dict):
        with self._session.begin() as session:
            if not session.get(Collection, collection_id):
                raise ValueError("Collection not found.")
            session.add(AnalysisRun(
                collection_id=collection_id,
                analysis_type=analysis_type,
                status=status,
                result_json=json.dumps(result, ensure_ascii=False),
            ))

    def save_graph(self, collection_id: str, edges: Iterable[Dict]):
        with self._session.begin() as session:
            if not session.get(Collection, collection_id):
                raise ValueError("Collection not found.")
            session.query(Relationship).filter(Relationship.collection_id == collection_id).delete()
            entities = {
                entity.name.casefold(): entity
                for entity in session.scalars(select(Entity).where(Entity.collection_id == collection_id))
            }
            for edge in edges:
                for name, kind in (
                    (edge["source_entity"], edge["source_type"]),
                    (edge["target_entity"], edge["target_type"]),
                ):
                    if name.casefold() not in entities:
                        entity = Entity(collection_id=collection_id, name=name, entity_type=kind)
                        session.add(entity)
                        session.flush()
                        entities[name.casefold()] = entity
                session.add(Relationship(
                    collection_id=collection_id,
                    source_entity_id=entities[edge["source_entity"].casefold()].id,
                    target_entity_id=entities[edge["target_entity"].casefold()].id,
                    document_id=edge["document_id"],
                    chunk_id=edge["chunk_id"],
                    relationship_type=edge["relationship"],
                    page_number=edge.get("page_number"),
                    evidence_sentence=edge["evidence_sentence"],
                    extraction_method=edge["extraction_method"],
                ))
            used_entity_ids = set(session.scalars(
                select(Relationship.source_entity_id).where(Relationship.collection_id == collection_id)
            )) | set(session.scalars(
                select(Relationship.target_entity_id).where(Relationship.collection_id == collection_id)
            ))
            session.query(Entity).filter(
                Entity.collection_id == collection_id,
                Entity.id.not_in(used_entity_ids),
            ).delete(synchronize_session=False)

    def get_document(self, collection_id: str, document_id: str) -> Optional[Document]:
        with self._session() as session:
            return session.scalar(
                select(Document).where(Document.collection_id == collection_id, Document.id == document_id)
            )

    def delete_document(self, collection_id: str, document_id: str) -> bool:
        with self._session.begin() as session:
            document = session.scalar(
                select(Document).where(Document.collection_id == collection_id, Document.id == document_id)
            )
            if not document:
                return False
            file_path = Path(document.file_path).resolve()
            session.delete(document)
        self._unlink_data_file(file_path)
        return True

    def delete_collection(self, collection_id: str) -> bool:
        with self._session.begin() as session:
            collection = session.get(Collection, collection_id)
            if not collection:
                return False
            paths = [Path(doc.file_path).resolve() for doc in collection.documents]
            session.delete(collection)
        for path in paths:
            self._unlink_data_file(path)
        collection_dir = (self.data_dir / "collections" / collection_id).resolve()
        if collection_dir.is_relative_to(self.data_dir) and collection_dir.exists():
            for path in sorted(collection_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    path.rmdir()
            collection_dir.rmdir()
        return True

    def _unlink_data_file(self, path: Path):
        if path.is_relative_to(self.data_dir) and path.is_file():
            path.unlink()

    def close(self):
        self.engine.dispose()


def decode_embedding(value: Optional[str]):
    if not value:
        return None
    return json.loads(value)


def encode_embedding(values: Iterable[float]) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))
