package vectordb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

/**
 * K-Dimensional tree: a binary tree where level d splits space on axis (d % dims).
 * Java's garbage collector replaces the manual destroy()/delete from C++.
 */
public class KDTree {
    private static class Node {
        final VectorItem item;
        Node left, right;
        Node(VectorItem item) { this.item = item; }
    }

    private Node root;
    private final int dims;

    public KDTree(int dims) { this.dims = dims; }

    public void insert(VectorItem v) { root = ins(root, v, 0); }

    private Node ins(Node n, VectorItem v, int depth) {
        if (n == null) return new Node(v);
        int ax = depth % dims;
        if (v.emb()[ax] < n.item.emb()[ax]) n.left = ins(n.left, v, depth + 1);
        else n.right = ins(n.right, v, depth + 1);
        return n;
    }

    public List<Pair> knn(float[] q, int k, Distance dist) {
        // max-heap: the WORST of the current best-k sits on top, so we can evict it quickly
        PriorityQueue<Pair> heap = new PriorityQueue<>(Collections.reverseOrder());
        search(root, q, k, 0, dist, heap);
        List<Pair> r = new ArrayList<>(heap);
        Collections.sort(r);
        return r;
    }

    private void search(Node n, float[] q, int k, int depth, Distance dist, PriorityQueue<Pair> heap) {
        if (n == null) return;
        float dn = dist.between(q, n.item.emb());
        if (heap.size() < k || dn < heap.peek().dist()) {
            heap.add(new Pair(dn, n.item.id()));
            if (heap.size() > k) heap.poll();
        }
        int ax = depth % dims;
        float diff = q[ax] - n.item.emb()[ax];
        Node closer  = diff < 0 ? n.left  : n.right;
        Node farther = diff < 0 ? n.right : n.left;
        search(closer, q, k, depth + 1, dist, heap);
        // only visit the far side if the splitting plane is closer than our current k-th best
        if (heap.size() < k || Math.abs(diff) < heap.peek().dist())
            search(farther, q, k, depth + 1, dist, heap);
    }

    public void rebuild(List<VectorItem> items) {
        root = null;
        for (VectorItem v : items) insert(v);
    }
}
