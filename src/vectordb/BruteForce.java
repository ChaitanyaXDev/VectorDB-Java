package vectordb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Exact search: compare the query with EVERY stored vector. O(N*d). The baseline. */
public class BruteForce {
    private final List<VectorItem> items = new ArrayList<>();

    public void insert(VectorItem v) { items.add(v); }

    public List<Pair> knn(float[] q, int k, Distance dist) {
        List<Pair> r = new ArrayList<>(items.size());
        for (VectorItem v : items) r.add(new Pair(dist.between(q, v.emb()), v.id()));
        Collections.sort(r);
        return r.size() > k ? new ArrayList<>(r.subList(0, k)) : r;
    }

    public void remove(int id) { items.removeIf(v -> v.id() == id); }
}
