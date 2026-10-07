package vectordb;

/** (distance, id) pair. Natural ordering = smaller distance first. Replaces C++ std::pair<float,int>. */
public record Pair(float dist, int id) implements Comparable<Pair> {
    @Override
    public int compareTo(Pair o) {
        int c = Float.compare(dist, o.dist);
        return c != 0 ? c : Integer.compare(id, o.id);
    }
}
