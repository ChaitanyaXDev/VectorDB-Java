package vectordb;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Tiny hand-rolled JSON helpers (the original C++ did this by hand too - no JSON library needed). */
public final class Json {
    private Json() { }

    /** Quote + escape a Java string as a JSON string literal. */
    public static String str(String s) {
        StringBuilder o = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"'  -> o.append("\\\"");
                case '\\' -> o.append("\\\\");
                case '\n' -> o.append("\\n");
                case '\r' -> o.append("\\r");
                case '\t' -> o.append("\\t");
                default -> {
                    if (c < 0x20) o.append(String.format("\\u%04x", (int) c));
                    else o.append(c);
                }
            }
        }
        return o.append('"').toString();
    }

    /** float[] -> "[0.1000,0.2000,...]" */
    public static String vec(float[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(String.format(Locale.ROOT, "%.4f", v[i]));
        }
        return sb.append(']').toString();
    }

    public static String num(float f, int decimals) {
        return String.format(Locale.ROOT, "%." + decimals + "f", f);
    }

    /** "0.1,0.2,0.3" -> float[]. Bad tokens are skipped silently (like the C++ version). */
    public static float[] parseVec(String s) {
        List<Float> out = new ArrayList<>();
        if (s != null) {
            for (String t : s.split(",")) {
                try { out.add(Float.parseFloat(t.trim())); } catch (NumberFormatException ignored) { }
            }
        }
        float[] r = new float[out.size()];
        for (int i = 0; i < r.length; i++) r[i] = out.get(i);
        return r;
    }

    /** Pull a string field out of a flat JSON object, handling \" \\ \n \r \t \/ and \\uXXXX escapes. */
    public static String extractStr(String body, String key) {
        int p = body.indexOf('"' + key + '"');
        if (p < 0) return "";
        p = body.indexOf(':', p);
        if (p < 0) return "";
        p++;
        while (p < body.length() && Character.isWhitespace(body.charAt(p))) p++;
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;
        StringBuilder r = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                char e = body.charAt(p);
                switch (e) {
                    case 'n' -> r.append('\n');
                    case 'r' -> r.append('\r');
                    case 't' -> r.append('\t');
                    case 'u' -> {
                        if (p + 4 < body.length()) {
                            try { r.append((char) Integer.parseInt(body.substring(p + 1, p + 5), 16)); p += 4; }
                            catch (NumberFormatException ex) { r.append('u'); }
                        } else r.append('u');
                    }
                    default -> r.append(e);       // covers \" \\ \/
                }
            } else {
                r.append(c);
            }
            p++;
        }
        return r.toString();
    }

    public static int extractInt(String body, String key, int def) {
        int p = body.indexOf('"' + key + '"');
        if (p < 0) return def;
        p = body.indexOf(':', p);
        if (p < 0) return def;
        p++;
        while (p < body.length() && Character.isWhitespace(body.charAt(p))) p++;
        int e = p;
        while (e < body.length() && (Character.isDigit(body.charAt(e)) || (e == p && body.charAt(e) == '-'))) e++;
        try { return Integer.parseInt(body.substring(p, e)); } catch (NumberFormatException ex) { return def; }
    }

    /** Extract a numeric array "key":[...] as float[] (empty if missing). */
    public static float[] extractArr(String body, String key) {
        int p = body.indexOf('"' + key + '"');
        if (p < 0) return new float[0];
        p = body.indexOf('[', p);
        if (p < 0) return new float[0];
        int e = body.indexOf(']', p);
        if (e < 0) return new float[0];
        return parseVec(body.substring(p + 1, e));
    }
}
