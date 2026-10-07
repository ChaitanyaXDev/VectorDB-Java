package vectordb;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Stores real text chunks + their Ollama embeddings (e.g. 768-D). Used by the RAG pipeline. */
public class DocumentDB {
    public record DocItem(int id, String title, String text, float[] emb) { }
    public record DocHit(float dist, DocItem doc) { }

    private final Map<Integer, DocItem> store = new HashMap<>();
    private final HNSW hnsw = new HNSW(16, 200);
    private final BruteForce bf = new BruteForce();
    private int nextId = 1;
    private int dims = 0;      // learned from the first embedding inserted

    public synchronized int insert(String title, String text, float[] emb) {
        if (dims == 0) dims = emb.length;
        DocItem item = new DocItem(nextId++, title, text, emb);
        store.put(item.id(), item);
        VectorItem vi = new VectorItem(item.id(), title, "doc", emb);
        hnsw.insert(vi, Distance.COSINE);
        bf.insert(vi);
        return item.id();
    }

    /** Top-k most similar chunks; anything with cosine distance > maxDist is dropped as irrelevant. */
    public synchronized List<DocHit> search(float[] q, int k, float maxDist) {
        List<DocHit> out = new ArrayList<>();
        if (store.isEmpty()) return out;
        // tiny collections: exact brute force is both faster and perfectly accurate
        List<Pair> raw = store.size() < 10 ? bf.knn(q, k, Distance.COSINE)
                                           : hnsw.knn(q, k, 50, Distance.COSINE);
        for (Pair p : raw) {
            DocItem d = store.get(p.id());
            if (d != null && p.dist() <= maxDist) out.add(new DocHit(p.dist(), d));
        }
        return out;
    }

    public List<DocHit> search(float[] q, int k) { return search(q, k, 0.7f); }

    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) return false;
        store.remove(id); hnsw.remove(id); bf.remove(id);
        return true;
    }

    public synchronized List<DocItem> all() { return new ArrayList<>(store.values()); }
    public synchronized int size() { return store.size(); }
    public synchronized int getDims() { return dims; }
}
