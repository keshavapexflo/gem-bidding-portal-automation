import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import create_embeddings


class FakeVectors(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def __init__(self, *_args, **_kwargs):
        pass

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, documents, **_kwargs):
        return FakeVectors([[float(len(text)), 1.0] for text in documents])


class FakeCollection:
    def __init__(self):
        self.records = {"existing-id": [1.0, 1.0]}
        self.upserted_ids = []

    def count(self):
        return len(self.records)

    def get(self, **_kwargs):
        return {"ids": list(self.records)}

    def peek(self, **_kwargs):
        return {"embeddings": [next(iter(self.records.values()))]}

    def upsert(self, ids, embeddings, **_kwargs):
        self.upserted_ids.extend(ids)
        self.records.update(dict(zip(ids, embeddings)))

    def delete(self, ids):
        for chunk_id in ids:
            self.records.pop(chunk_id, None)


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def get_or_create_collection(self, **_kwargs):
        return self.collection


class ResumeEmbeddingTests(unittest.TestCase):
    def test_resume_skips_existing_id_and_embeds_missing_id(self):
        collection = FakeCollection()
        chromadb = types.SimpleNamespace(
            PersistentClient=lambda **_kwargs: FakeClient(collection)
        )
        sentence_transformers = types.SimpleNamespace(SentenceTransformer=FakeModel)

        previous_chromadb = sys.modules.get("chromadb")
        previous_transformers = sys.modules.get("sentence_transformers")
        sys.modules["chromadb"] = chromadb
        sys.modules["sentence_transformers"] = sentence_transformers

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = Path(temp_dir) / "bid_chunks.json"
                input_path.write_text(
                    json.dumps({
                        "bids": [{
                            "metadata": {"bid_id": "bid-1"},
                            "chunks": [
                                {"chunk_id": "existing-id", "text": "already stored"},
                                {"chunk_id": "missing-id", "text": "needs embedding"},
                            ],
                        }]
                    }),
                    encoding="utf-8",
                )
                processed = create_embeddings.build_embeddings(Namespace(
                    input=input_path,
                    chroma_path=Path(temp_dir) / "chroma_db",
                    collection="test",
                    model="test-model",
                    model_revision="test-revision",
                    batch_size=2,
                    limit=0,
                    sync_file=None,
                    reset=False,
                    resume=True,
                ))
        finally:
            if previous_chromadb is None:
                sys.modules.pop("chromadb", None)
            else:
                sys.modules["chromadb"] = previous_chromadb
            if previous_transformers is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = previous_transformers

        self.assertEqual(processed, 1)
        self.assertEqual(collection.upserted_ids, ["missing-id"])
        self.assertEqual(collection.count(), 2)


if __name__ == "__main__":
    unittest.main()
