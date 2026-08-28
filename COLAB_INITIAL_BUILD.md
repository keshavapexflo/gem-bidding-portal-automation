# Optional Colab GPU initial build

For a large initial corpus, upload this complete code directory together with
`downloads\` and the PDFs to Google Drive. In a GPU-enabled Colab runtime run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive/LetsBid
!pip install -r requirements.txt
```

For PDFs produced by the current chunker:

```python
!python portal_pipeline.py initialise --skip-download --reset-index --batch-size 256
```

For the legacy corpus used by `embedding_40k.ipynb`, rebuild the chunks first so
repeated ATC clause numbers receive collision-safe IDs:

```python
!python portal_pipeline.py initialise --skip-download --force-rechunk --reset-index --batch-size 256
```

The command writes Chroma directly to `chroma_db\`, creates the durable
boilerplate registry, updates Chroma metadata, and builds the lexical index.
After validation, transfer all three artifacts to the laptop:

```text
bid_chunks.json
downloads/
chroma_db/
```

The corpus and query code use the same pinned BGE model snapshot recorded in
`deployment_manifest.json`.
