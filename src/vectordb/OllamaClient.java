package vectordb;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/** Talks to a local Ollama server over HTTP (replaces httplib::Client). */
public class OllamaClient {
    private final String base;
    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(3)).build();

    public String embedModel = "nomic-embed-text";
    public String genModel = "llama3.2";

    public OllamaClient() { this("127.0.0.1", 11434); }
    public OllamaClient(String host, int port) { this.base = "http://" + host + ":" + port; }

    public boolean isAvailable() {
        try {
            HttpRequest rq = HttpRequest.newBuilder(URI.create(base + "/api/tags"))
                    .timeout(Duration.ofSeconds(2)).GET().build();
            return http.send(rq, HttpResponse.BodyHandlers.ofString()).statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    private HttpResponse<String> post(String path, String json, int timeoutSec) throws Exception {
        HttpRequest rq = HttpRequest.newBuilder(URI.create(base + path))
                .timeout(Duration.ofSeconds(timeoutSec))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json)).build();
        return http.send(rq, HttpResponse.BodyHandlers.ofString());
    }

    /** Text -> embedding vector. Returns an empty array if Ollama is down or the model is missing. */
    public float[] embed(String text) {
        try {
            String body = "{\"model\":" + Json.str(embedModel) + ",\"prompt\":" + Json.str(text) + "}";
            HttpResponse<String> res = post("/api/embeddings", body, 30);
            if (res.statusCode() != 200) return new float[0];
            return Json.extractArr(res.body(), "embedding");
        } catch (Exception e) {
            return new float[0];
        }
    }

    /** Prompt -> full LLM answer (non-streaming). Returns an ERROR string if Ollama is unavailable. */
    public String generate(String prompt) {
        try {
            String body = "{\"model\":" + Json.str(genModel) + ",\"prompt\":" + Json.str(prompt)
                    + ",\"stream\":false}";
            HttpResponse<String> res = post("/api/generate", body, 180);   // LLMs can be slow
            if (res.statusCode() != 200) return "ERROR: Ollama unavailable. Run: ollama serve";
            return Json.extractStr(res.body(), "response");
        } catch (Exception e) {
            return "ERROR: Ollama unavailable. Run: ollama serve";
        }
    }
}
