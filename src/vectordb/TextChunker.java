package vectordb;

import java.util.ArrayList;
import java.util.List;

/** Splits long text into overlapping word windows so each chunk fits the embedding model well. */
public final class TextChunker {
    private TextChunker() { }

    public static List<String> chunk(String text, int chunkWords, int overlapWords) {
        String trimmed = text.trim();
        List<String> chunks = new ArrayList<>();
        if (trimmed.isEmpty()) return chunks;
        String[] words = trimmed.split("\\s+");
        if (words.length <= chunkWords) { chunks.add(text); return chunks; }

        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            chunks.add(String.join(" ", java.util.Arrays.copyOfRange(words, i, end)));
            if (end == words.length) break;
        }
        return chunks;
    }
}
