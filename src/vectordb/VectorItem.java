package vectordb;

/** One stored vector: id + human-readable text + category + the embedding itself. */
public record VectorItem(int id, String metadata, String category, float[] emb) { }
