# VectorDB in Java

Java 17+ port of the C++ VectorDB project (HNSW, KD-Tree, Brute Force, RAG via Ollama).
No external libraries: uses only the JDK (`com.sun.net.httpserver` and `java.net.http`).
The REST API is identical, so the original `index.html` works unchanged.

## Run
1. Install a JDK (17 or newer): https://adoptium.net  (check with `javac -version`)
2. Optional, for Documents / Ask AI tabs: install Ollama, then
   `ollama pull nomic-embed-text` and `ollama pull llama3.2`
3. In this folder:
   - Windows: `run.bat`
   - Linux/macOS: `./run.sh`
4. Open http://localhost:9090

Change the LLM: edit `genModel` in `src/vectordb/OllamaClient.java`.

## Files (src/vectordb)
| File | Role |
|---|---|
| Main.java | HTTP server, routing, all REST endpoints, RAG prompt |
| VectorDB.java | 16-D demo DB, keeps BruteForce + KDTree + HNSW in sync |
| DocumentDB.java | Real-embedding store for the RAG pipeline |
| HNSW.java / KDTree.java / BruteForce.java | The three search algorithms |
| Distance.java | Cosine / Euclidean / Manhattan |
| OllamaClient.java | HTTP client for Ollama embeddings + generation |
| TextChunker.java | 250-word overlapping chunks |
| Json.java | Hand-written JSON helpers |
| DemoData.java | The 20 demo vectors |
| VectorItem.java / Pair.java | Small data records |
