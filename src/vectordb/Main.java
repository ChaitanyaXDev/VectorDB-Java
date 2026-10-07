package vectordb;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** HTTP server + REST API. Same endpoints as the C++ version, so index.html works unchanged. */
public class Main {
    static final int DIMS = 16;
    static final int PORT = 9090;

    static final VectorDB db = new VectorDB(DIMS);
    static final DocumentDB docDB = new DocumentDB();
    static final OllamaClient ollama = new OllamaClient();

    static final Pattern DELETE_VEC = Pattern.compile("^/delete/(\\d+)$");
    static final Pattern DELETE_DOC = Pattern.compile("^/doc/delete/(\\d+)$");

    public static void main(String[] args) throws IOException {
        DemoData.load(db);

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine (Java) ===");
        System.out.println("http://localhost:" + PORT);
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp)
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);

        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/", Main::handle);
        server.setExecutor(Executors.newFixedThreadPool(8));   // handle several requests at once
        server.start();
    }

    // ------------------------------------------------------------------ routing

    static void handle(HttpExchange ex) throws IOException {
        try {
            ex.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
            ex.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
            ex.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");

            String method = ex.getRequestMethod();
            String path = ex.getRequestURI().getPath();

            if (method.equals("OPTIONS")) { send(ex, 204, "text/plain", ""); return; }

            Map<String, String> q = parseQuery(ex.getRequestURI().getRawQuery());
            String body = method.equals("POST")
                    ? new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8) : "";

            if (method.equals("GET")) {
                switch (path) {
                    case "/search"    -> { search(ex, q); return; }
                    case "/items"     -> { items(ex); return; }
                    case "/benchmark" -> { benchmark(ex, q); return; }
                    case "/hnsw-info" -> { hnswInfo(ex); return; }
                    case "/doc/list"  -> { docList(ex); return; }
                    case "/status"    -> { status(ex); return; }
                    case "/stats"     -> { stats(ex); return; }
                    case "/"          -> { index(ex); return; }
                    default -> { }
                }
            } else if (method.equals("POST")) {
                switch (path) {
                    case "/insert"     -> { insert(ex, body); return; }
                    case "/doc/insert" -> { docInsert(ex, body); return; }
                    case "/doc/search" -> { docSearch(ex, body); return; }
                    case "/doc/ask"    -> { docAsk(ex, body); return; }
                    default -> { }
                }
            } else if (method.equals("DELETE")) {
                Matcher m = DELETE_VEC.matcher(path);
                if (m.matches()) {
                    json(ex, "{\"ok\":" + db.remove(Integer.parseInt(m.group(1))) + "}");
                    return;
                }
                m = DELETE_DOC.matcher(path);
                if (m.matches()) {
                    json(ex, "{\"ok\":" + docDB.remove(Integer.parseInt(m.group(1))) + "}");
                    return;
                }
            }
            send(ex, 404, "application/json", "{\"error\":\"not found\"}");
        } catch (Exception e) {
            e.printStackTrace();
            send(ex, 500, "application/json", "{\"error\":" + Json.str(String.valueOf(e.getMessage())) + "}");
        }
    }

    // ------------------------------------------------------------------ demo vector endpoints

    static void search(HttpExchange ex, Map<String, String> q) throws IOException {
        float[] v = Json.parseVec(q.getOrDefault("v", ""));
        if (v.length != DIMS) { json(ex, "{\"error\":\"need " + DIMS + "D vector\"}"); return; }
        int k = intOr(q.get("k"), 5);
        String metric = orDefault(q.get("metric"), "cosine");
        String algo = orDefault(q.get("algo"), "hnsw");

        VectorDB.SearchOut out = db.search(v, k, metric, algo);
        StringBuilder sb = new StringBuilder("{\"results\":[");
        for (int i = 0; i < out.hits().size(); i++) {
            if (i > 0) sb.append(',');
            VectorDB.Hit h = out.hits().get(i);
            sb.append("{\"id\":").append(h.id())
              .append(",\"metadata\":").append(Json.str(h.meta()))
              .append(",\"category\":").append(Json.str(h.cat()))
              .append(",\"distance\":").append(Json.num(h.dist(), 6))
              .append(",\"embedding\":").append(Json.vec(h.emb())).append('}');
        }
        sb.append("],\"latencyUs\":").append(out.us())
          .append(",\"algo\":").append(Json.str(out.algo()))
          .append(",\"metric\":").append(Json.str(out.metric())).append('}');
        json(ex, sb.toString());
    }

    static void insert(HttpExchange ex, String body) throws IOException {
        String meta = Json.extractStr(body, "metadata");
        String cat = Json.extractStr(body, "category");
        float[] emb = Json.extractArr(body, "embedding");
        if (meta.isEmpty() || emb.length != DIMS) { json(ex, "{\"error\":\"invalid body\"}"); return; }
        int id = db.insert(meta, cat, emb, Distance.COSINE);
        json(ex, "{\"id\":" + id + "}");
    }

    static void items(HttpExchange ex) throws IOException {
        List<VectorItem> items = db.all();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) sb.append(',');
            VectorItem v = items.get(i);
            sb.append("{\"id\":").append(v.id())
              .append(",\"metadata\":").append(Json.str(v.metadata()))
              .append(",\"category\":").append(Json.str(v.category()))
              .append(",\"embedding\":").append(Json.vec(v.emb())).append('}');
        }
        json(ex, sb.append(']').toString());
    }

    static void benchmark(HttpExchange ex, Map<String, String> q) throws IOException {
        float[] v = Json.parseVec(q.getOrDefault("v", ""));
        if (v.length != DIMS) { json(ex, "{\"error\":\"need " + DIMS + "D vector\"}"); return; }
        int k = intOr(q.get("k"), 5);
        String metric = orDefault(q.get("metric"), "cosine");
        VectorDB.BenchOut b = db.benchmark(v, k, metric);
        json(ex, "{\"bruteforceUs\":" + b.bfUs() + ",\"kdtreeUs\":" + b.kdUs()
                + ",\"hnswUs\":" + b.hnswUs() + ",\"itemCount\":" + b.n() + "}");
    }

    static void hnswInfo(HttpExchange ex) throws IOException {
        HNSW.GraphInfo gi = db.hnswInfo();
        StringBuilder sb = new StringBuilder();
        sb.append("{\"topLayer\":").append(gi.topLayer)
          .append(",\"nodeCount\":").append(gi.nodeCount)
          .append(",\"nodesPerLayer\":").append(java.util.Arrays.toString(gi.nodesPerLayer).replace(" ", ""))
          .append(",\"edgesPerLayer\":").append(java.util.Arrays.toString(gi.edgesPerLayer).replace(" ", ""))
          .append(",\"nodes\":[");
        for (int i = 0; i < gi.nodes.size(); i++) {
            if (i > 0) sb.append(',');
            HNSW.NodeView n = gi.nodes.get(i);
            sb.append("{\"id\":").append(n.id()).append(",\"metadata\":").append(Json.str(n.metadata()))
              .append(",\"category\":").append(Json.str(n.category()))
              .append(",\"maxLyr\":").append(n.maxLyr()).append('}');
        }
        sb.append("],\"edges\":[");
        for (int i = 0; i < gi.edges.size(); i++) {
            if (i > 0) sb.append(',');
            HNSW.EdgeView e = gi.edges.get(i);
            sb.append("{\"src\":").append(e.src()).append(",\"dst\":").append(e.dst())
              .append(",\"lyr\":").append(e.lyr()).append('}');
        }
        json(ex, sb.append("]}").toString());
    }

    // ------------------------------------------------------------------ document + RAG endpoints

    /** POST /doc/insert {"title":"...","text":"..."}  -> chunk, embed each chunk, store. */
    static void docInsert(HttpExchange ex, String body) throws IOException {
        String title = Json.extractStr(body, "title");
        String text = Json.extractStr(body, "text");
        if (title.isEmpty() || text.isEmpty()) { json(ex, "{\"error\":\"need title and text\"}"); return; }

        List<String> chunks = TextChunker.chunk(text, 250, 30);
        StringBuilder ids = new StringBuilder();
        for (int i = 0; i < chunks.size(); i++) {
            float[] emb = ollama.embed(chunks.get(i));
            if (emb.length == 0) {
                json(ex, "{\"error\":\"Ollama unavailable. Install from https://ollama.com then run: "
                        + "ollama pull nomic-embed-text && ollama pull llama3.2\"}");
                return;
            }
            String chunkTitle = chunks.size() > 1 ? title + " [" + (i + 1) + "/" + chunks.size() + "]" : title;
            int id = docDB.insert(chunkTitle, chunks.get(i), emb);
            if (i > 0) ids.append(',');
            ids.append(id);
        }
        json(ex, "{\"ids\":[" + ids + "],\"chunks\":" + chunks.size() + ",\"dims\":" + docDB.getDims() + "}");
    }

    static void docList(HttpExchange ex) throws IOException {
        List<DocumentDB.DocItem> docs = docDB.all();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < docs.size(); i++) {
            if (i > 0) sb.append(',');
            DocumentDB.DocItem d = docs.get(i);
            String preview = d.text().length() > 120 ? d.text().substring(0, 120) + "…" : d.text();
            int words = d.text().trim().isEmpty() ? 0 : d.text().trim().split("\\s+").length;
            sb.append("{\"id\":").append(d.id())
              .append(",\"title\":").append(Json.str(d.title()))
              .append(",\"preview\":").append(Json.str(preview))
              .append(",\"words\":").append(words).append('}');
        }
        json(ex, sb.append(']').toString());
    }

    /** POST /doc/search - retrieval only (no LLM), used by the UI to highlight matches fast. */
    static void docSearch(HttpExchange ex, String body) throws IOException {
        String question = Json.extractStr(body, "question");
        int k = Json.extractInt(body, "k", 3);
        if (question.isEmpty()) { json(ex, "{\"error\":\"need question\"}"); return; }
        float[] qEmb = ollama.embed(question);
        if (qEmb.length == 0) { json(ex, "{\"error\":\"Ollama unavailable\"}"); return; }

        List<DocumentDB.DocHit> hits = docDB.search(qEmb, k);
        StringBuilder sb = new StringBuilder("{\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) sb.append(',');
            DocumentDB.DocHit h = hits.get(i);
            sb.append("{\"id\":").append(h.doc().id())
              .append(",\"title\":").append(Json.str(h.doc().title()))
              .append(",\"distance\":").append(Json.num(h.dist(), 4)).append('}');
        }
        json(ex, sb.append("]}").toString());
    }

    /** POST /doc/ask - the full RAG pipeline: embed question -> retrieve -> build prompt -> generate. */
    static void docAsk(HttpExchange ex, String body) throws IOException {
        String question = Json.extractStr(body, "question");
        int k = Json.extractInt(body, "k", 3);
        if (question.isEmpty()) { json(ex, "{\"error\":\"need question\"}"); return; }

        float[] qEmb = ollama.embed(question);                       // 1. embed the question
        if (qEmb.length == 0) { json(ex, "{\"error\":\"Ollama unavailable\"}"); return; }

        List<DocumentDB.DocHit> hits = docDB.search(qEmb, k);        // 2. retrieve top-k chunks

        StringBuilder ctx = new StringBuilder();                     // 3. build the prompt
        for (int i = 0; i < hits.size(); i++)
            ctx.append('[').append(i + 1).append("] ").append(hits.get(i).doc().title()).append(":\n")
               .append(hits.get(i).doc().text()).append("\n\n");
        String prompt =
            "You are a helpful assistant. Answer the user's question directly. "
          + "Use the provided context if it contains relevant information. "
          + "If it doesn't, just use your own general knowledge. "
          + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
          + "Just answer the question naturally.\n\n"
          + "Context:\n" + ctx + "Question: " + question + "\n\nAnswer:";

        String answer = ollama.generate(prompt);                     // 4. generate

        StringBuilder sb = new StringBuilder();                      // 5. return everything
        sb.append("{\"answer\":").append(Json.str(answer))
          .append(",\"model\":").append(Json.str(ollama.genModel))
          .append(",\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) sb.append(',');
            DocumentDB.DocHit h = hits.get(i);
            sb.append("{\"id\":").append(h.doc().id())
              .append(",\"title\":").append(Json.str(h.doc().title()))
              .append(",\"text\":").append(Json.str(h.doc().text()))
              .append(",\"distance\":").append(Json.num(h.dist(), 4)).append('}');
        }
        sb.append("],\"docCount\":").append(docDB.size()).append('}');
        json(ex, sb.toString());
    }

    static void status(HttpExchange ex) throws IOException {
        boolean up = ollama.isAvailable();
        json(ex, "{\"ollamaAvailable\":" + up
                + ",\"embedModel\":" + Json.str(ollama.embedModel)
                + ",\"genModel\":" + Json.str(ollama.genModel)
                + ",\"docCount\":" + docDB.size()
                + ",\"docDims\":" + docDB.getDims()
                + ",\"demoDims\":" + DIMS
                + ",\"demoCount\":" + db.size() + "}");
    }

    static void stats(HttpExchange ex) throws IOException {
        json(ex, "{\"count\":" + db.size() + ",\"dims\":" + DIMS
                + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
                + ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}");
    }

    static void index(HttpExchange ex) throws IOException {
        Path p = Path.of("index.html");
        if (!Files.exists(p)) { send(ex, 404, "text/plain", "index.html not found - run from the project folder"); return; }
        send(ex, 200, "text/html; charset=utf-8", Files.readString(p));
    }

    // ------------------------------------------------------------------ helpers

    static void json(HttpExchange ex, String body) throws IOException {
        send(ex, 200, "application/json", body);
    }

    static void send(HttpExchange ex, int status, String type, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", type);
        if (status == 204) { ex.sendResponseHeaders(204, -1); ex.close(); return; }
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    static Map<String, String> parseQuery(String raw) {
        Map<String, String> m = new HashMap<>();
        if (raw == null || raw.isEmpty()) return m;
        for (String pair : raw.split("&")) {
            int i = pair.indexOf('=');
            String k = i < 0 ? pair : pair.substring(0, i);
            String v = i < 0 ? "" : pair.substring(i + 1);
            m.put(URLDecoder.decode(k, StandardCharsets.UTF_8), URLDecoder.decode(v, StandardCharsets.UTF_8));
        }
        return m;
    }

    static int intOr(String s, int def) {
        try { return Integer.parseInt(s); } catch (Exception e) { return def; }
    }

    static String orDefault(String s, String def) { return (s == null || s.isEmpty()) ? def : s; }
}
