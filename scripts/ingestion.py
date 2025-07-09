#!/usr/bin/env python3
"""
scripts/ingest.py

Loads all .txt files from data/texts/, splits them into overlapping chunks,
and writes the chunks (with metadata) to data/chunks.json.
"""

import os
import json
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Determine paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    input_dir = os.path.join(base_dir, os.pardir, "data", "texts")
    output_file = os.path.join(base_dir, os.pardir, "data", "chunks.json")

    logging.info(f"Looking for .txt files in: {input_dir}")

    # Verify input directory exists
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return

    # Collect raw texts
    docs = []
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".txt"):
            logging.debug(f"Skipping non-txt file: {fname}")
            continue
        path = os.path.join(input_dir, fname)
        logging.info(f"Reading file: {path}")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        docs.append({"source": fname, "text": text})

    logging.info(f"Found {len(docs)} document(s)")

    if not docs:
        logging.warning("No documents to process – exiting.")
        return

    # Configure splitter
    chunk_size = 500
    chunk_overlap = 100
    logging.info(f"Splitter configuration: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split into chunks
    chunks = []
    for doc in docs:
        texts = splitter.split_text(doc["text"])
        logging.info(f"Document {doc['source']} → {len(texts)} chunk(s)")
        for i, chunk in enumerate(texts):
            chunks.append({
                "chunk_id": i,
                "source": doc["source"],
                "text": chunk
            })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write out JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved {len(chunks)} total chunk(s) to {output_file}")

if __name__ == "__main__":
    main()
