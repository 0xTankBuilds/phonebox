using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;

namespace PhoneBox.Agent;

public sealed partial class SessionMonitor
{
    // Pi (pi-coding-agent) writes one JSONL file per session in <agent dir>/sessions/<project slug>/<timestamp>_<id>.jsonl,
    // where the agent dir is ~/.pi/agent unless PI_CODING_AGENT_DIR says otherwise.
    // The monitor keeps only metadata: the "session" record's cwd and id, each "model_change"'s provider/modelId, each
    // "thinking_level_change"'s level, the title from "session_info" (while titles are shared), and, from assistant
    // "message" records, token usage, the stop reason, model, effort and tool names. Prompts, replies and tool
    // arguments are never kept or sent.

    private sealed record PiDetails(string? Id, string? Cwd, string? Provider, string? Model, string? Effort,
        bool HasUsage, bool TokensComplete, long Input, long Output, long CacheRead, long CacheWrite, long Reasoning,
        long LastInputTokens,
        bool TurnOpen, bool TurnClosed, string? Outcome, long? TurnStartedAt, long? LastActivityAt, long? ClosedAt,
        string? LastTool, string? Title, long ReadOffset);

    /// <summary>The first fields of a record, enough to keep its lifecycle meaning without reading its payload.</summary>
    private sealed record PiRecordHead(string? Type, string? Role, long? Timestamp);

    private const int PiMaxReadBytes = 2 * 1024 * 1024;
    private const int PiHeaderBytes = 64 * 1024;
    // A record too long to parse without holding its payload whole: only its own first and last bytes are read.
    private const int PiRecordHeadBytes = 4 * 1024;
    private const int PiRecordTailBytes = 4 * 1024;
    private const int PiScanChunkBytes = 64 * 1024;
    // Where a message record's role can be: the first property of its message object, before any payload.
    private const int PiRoleScanBytes = 1024;
    // Pi writes the failed assistant message before its auto-retry starts. A fresh error therefore stays "busy";
    // once the failed reply is older than this, the retries (including one slow provider attempt) have ended.
    private static readonly TimeSpan PiErrorGrace = TimeSpan.FromMinutes(5);

    private string? _piDiscoveryRoot;
    private long _nextPiDiscovery;
    private string[] _piCandidateFiles = [];
    private readonly ConcurrentDictionary<string, (DateTime LastWrite, long Length, PiDetails Details)> _piDetails = new();

    private static readonly ConcurrentDictionary<string, (long LoadedTick, Dictionary<string, Dictionary<string, long>> Windows)> _piContextWindows = new();
    private const int PiContextWindowRefreshMs = 60_000;

    /// <summary>
    /// Recent Pi agent sessions, from <c>PI_CODING_AGENT_DIR</c>/sessions (or ~/.pi/agent/sessions), or sessionsRoot
    /// for tests. The first complete record is the "session" record with the working directory; until it has been
    /// read the session is not shown.
    /// </summary>
    internal IEnumerable<AgentSession> ReadPi(string? sessionsRoot = null, bool titles = false)
    {
        var agentDir = Environment.GetEnvironmentVariable("PI_CODING_AGENT_DIR");
        if (string.IsNullOrWhiteSpace(agentDir)) agentDir = Path.Combine(Home, ".pi", "agent");
        var root = sessionsRoot ?? Path.Combine(agentDir, "sessions");
        if (!Directory.Exists(root)) yield break;
        var cutoff = DateTime.UtcNow.AddHours(-3);
        if (_piDiscoveryRoot != root || Environment.TickCount64 >= _nextPiDiscovery)
        {
            _piCandidateFiles = new DirectoryInfo(root).EnumerateFiles("*.jsonl", SearchOption.AllDirectories)
                .Where(f => f.LastWriteTimeUtc > cutoff).Select(f => f.FullName).ToArray();
            _piDiscoveryRoot = root;
            _nextPiDiscovery = Environment.TickCount64 + 30_000;
        }
        var active = new HashSet<string>(_piCandidateFiles);
        foreach (var old in _piDetails.Keys.Where(p => !active.Contains(p)))
            _piDetails.TryRemove(old, out _);

        var store = Path.GetFullPath(Path.Combine(root, "..", "models-store.json"));
        foreach (var path in _piCandidateFiles)
        {
            AgentSession? s = null;
            try
            {
                var f = new FileInfo(path);
                if (!f.Exists || f.LastWriteTimeUtc <= cutoff) continue;
                var hasCached = _piDetails.TryGetValue(path, out var cached);
                var unchanged = hasCached && cached.LastWrite == f.LastWriteTimeUtc && cached.Length == f.Length;
                var appended = hasCached && f.Length > cached.Length;
                var details = unchanged
                    ? cached.Details
                    : ReadPiDetails(path, appended ? cached.Details : null,
                        appended ? cached.Details.ReadOffset : 0);
                if (details is null) continue; // no complete record yet: the next pass reads it again
                if (!unchanged) _piDetails[path] = (f.LastWriteTimeUtc, f.Length, details);
                s = BuildPiSession(f, details, titles, store);
            }
            catch { }
            if (s is not null) yield return s;
        }
    }

    private static AgentSession? BuildPiSession(FileInfo f, PiDetails d, bool titles, string contextStore)
    {
        if (d.Cwd is not { Length: > 0 } cwd) return null; // the "session" record has not flushed yet
        var project = Path.GetFileName(cwd.TrimEnd('\\', '/'));
        project = SanitizeAntigravityLabel(project, AgyMaxProjectChars) ?? "pi";
        if (project.Length == 0) project = "pi";
        var id = d.Id ?? Path.GetFileNameWithoutExtension(f.Name);
        // Pi ids are UUIDv7: the start is a timestamp shared by every session for weeks, so the end is what differs.
        var shortId = id.Length <= 4 ? id : id[^4..];
        // A turn stays open while the last record is a user prompt, a tool result, or an assistant reply that carries
        // a tool call, however long the model or a tool takes. A final assistant reply (stopReason "stop", "length",
        // "aborted" or "error") closes it. An error closes it only after Pi's auto-retry grace, measured from the
        // reply that failed: a later title rename rewrites the file without retrying and must not restart it.
        var failedAt = d.ClosedAt ?? new DateTimeOffset(f.LastWriteTimeUtc).ToUnixTimeMilliseconds();
        var retrying = d.TurnClosed && d.Outcome == "failed"
            && DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - failedAt < PiErrorGrace.TotalMilliseconds;
        var open = d.TurnOpen || retrying;
        var contextWindow = ContextWindows(contextStore).TryGetValue(d.Provider ?? "", out var models)
            && models.TryGetValue(d.Model ?? "", out var w) ? (long?)w : null;
        var hasUsage = d.HasUsage && d.TokensComplete;
        var hasInput = hasUsage && (d.Input > 0 || d.CacheRead > 0 || d.CacheWrite > 0);
        return new AgentSession(
            Id: "pi:" + id,
            Tool: "pi",
            Name: $"{project.ToLowerInvariant()}-{shortId}",
            Project: project,
            Status: open ? "busy" : "idle",
            WaitingFor: null,
            Since: d.LastActivityAt ?? new DateTimeOffset(f.LastWriteTimeUtc).ToUnixTimeMilliseconds(),
            StartedAt: new DateTimeOffset(f.CreationTimeUtc).ToUnixTimeMilliseconds(),
            Model: d.Model, Effort: d.Effort,
            // Reasoning is part of Pi's output count, so it is not added again.
            TotalTokens: hasUsage ? d.Input + d.CacheRead + d.CacheWrite + d.Output : null,
            InputTokens: hasInput ? d.Input + d.CacheWrite : null,
            OutputTokens: hasUsage ? d.Output : null,
            CachedInputTokens: hasUsage ? d.CacheRead : null,
            ReasoningTokens: hasUsage && d.Reasoning > 0 ? d.Reasoning : null,
            LastInputTokens: d.LastInputTokens > 0 ? d.LastInputTokens : null,
            ContextWindow: contextWindow,
            TurnStartedAt: open ? d.TurnStartedAt : null,
            Outcome: open ? null : d.TurnClosed ? d.Outcome : null,
            LastTool: d.LastTool,
            Title: titles ? CleanTitle(d.Title) : null);
    }

    /// <summary>
    /// provider → model → context window, from Pi's own model store (~/.pi/agent/models-store.json, one level up from
    /// the sessions root). The store is keyed by provider and holds a <c>models</c> array; only the cached dynamic
    /// catalogs are there, so a built-in model simply has no window. Re-read at most once a minute.
    /// </summary>
    internal static Dictionary<string, Dictionary<string, long>> ContextWindows(string storePath)
    {
        if (_piContextWindows.TryGetValue(storePath, out var cached)
            && Environment.TickCount64 - cached.LoadedTick < PiContextWindowRefreshMs)
            return cached.Windows;
        var map = new Dictionary<string, Dictionary<string, long>>(StringComparer.OrdinalIgnoreCase);
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(storePath));
            // A non-object store is empty, not an error: cache the empty map like any other.
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
            {
                _piContextWindows[storePath] = (Environment.TickCount64, map);
                return map;
            }
            foreach (var provider in doc.RootElement.EnumerateObject())
            {
                if (provider.Value.ValueKind != JsonValueKind.Object
                    || !provider.Value.TryGetProperty("models", out var list)
                    || list.ValueKind != JsonValueKind.Array) continue;
                var models = new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
                foreach (var model in list.EnumerateArray())
                {
                    var id = Str(model, "id");
                    // Long checks the value kind, so null, a string or a fraction is skipped instead of throwing
                    // and hiding every session.
                    var w = Long(model, "contextWindow");
                    if (id is { Length: > 0 } && w is > 0) models[id] = w.Value;
                }
                if (models.Count > 0) map[provider.Name] = models;
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException) { }
        _piContextWindows[storePath] = (Environment.TickCount64, map);
        return map;
    }

    /// <summary>True when the details cache holds this path at the given write time and length (for tests).</summary>
    internal bool HasCachedPiDetails(string path, DateTime lastWriteUtc, long length) =>
        _piDetails.TryGetValue(path, out var cached) && cached.LastWrite == lastWriteUtc && cached.Length == length;

    /// <summary>
    /// Scans metadata once, then follows appended lines. The first read is capped to a tail window; the session
    /// header at the start is read separately so a large session still has its folder. Null while the file has no
    /// complete record yet.
    /// </summary>
    private static PiDetails? ReadPiDetails(string path, PiDetails? previous = null, long previousLength = 0)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        var len = fs.Length;
        // Where the previous read stopped: a record boundary, or the start of the file on the first read.
        var resume = previous is not null ? Math.Min(previousLength, len) : 0L;
        var start = resume;
        var jumped = len - start > PiMaxReadBytes;
        if (jumped) start = Math.Max(0, len - PiMaxReadBytes);
        fs.Seek(start, SeekOrigin.Begin);
        using var buffer = new MemoryStream();
        fs.CopyTo(buffer);
        var bytes = buffer.ToArray();
        var lastNl = Array.LastIndexOf(bytes, (byte)'\n');
        if (lastNl < 0) return previous; // nothing new, or still no complete record
        var complete = lastNl + 1;
        var parseFrom = 0;
        PiRecordHead? skipped = null;
        if (jumped && start > 0)
        {
            // The window starts inside a record that is not parsed. Its head is still read, so a record too large to
            // hold whole (a tool result with a screenshot in it) keeps its meaning: the turn is running.
            skipped = ReadPiRecordHead(fs, start, resume);
            // A tail window can also start in the middle of a record: drop that first partial line.
            parseFrom = Array.IndexOf(bytes, (byte)'\n', 0, complete) + 1;
        }

        var header = previous is null && start > 0 ? ReadPiHeader(path) : null;
        var id = header?.Id ?? previous?.Id;
        var cwd = header?.Cwd ?? previous?.Cwd;
        var provider = previous?.Provider;
        var model = previous?.Model;
        var effort = previous?.Effort;
        var hasUsage = previous?.HasUsage ?? false;
        var tokensComplete = (previous?.TokensComplete ?? true) && !jumped;
        var input = previous?.Input ?? 0;
        var output = previous?.Output ?? 0;
        var cacheRead = previous?.CacheRead ?? 0;
        var cacheWrite = previous?.CacheWrite ?? 0;
        var reasoning = previous?.Reasoning ?? 0;
        var lastInputTokens = previous?.LastInputTokens ?? 0;
        var turnOpen = previous?.TurnOpen ?? false;
        var turnClosed = previous?.TurnClosed ?? false;
        var outcome = previous?.Outcome;
        var turnStartedAt = previous?.TurnStartedAt;
        var lastActivityAt = previous?.LastActivityAt;
        var closedAt = previous?.ClosedAt;
        var lastTool = previous?.LastTool;
        var title = previous?.Title;

        // The skipped record still counts for the turn, but its payload and the records before it stay unread, so
        // only the lifecycle is taken from it: a prompt or a tool result keeps the turn open, a reply can close it.
        if (skipped is { Type: "message", Role: { } skippedRole })
        {
            lastActivityAt = skipped.Timestamp ?? lastActivityAt;
            if (skippedRole is "user" or "toolResult")
            {
                turnOpen = true;
                turnClosed = false;
                outcome = null;
                if (skippedRole == "user") turnStartedAt = skipped.Timestamp ?? turnStartedAt;
            }
            else if (skippedRole == "assistant")
            {
                // The stop reason sits after the payload, so it is the record's own last bytes: the dropped line.
                var skippedStop = PiTailStopReason(bytes, parseFrom - 1);
                if (PiClosedStop(skippedStop))
                {
                    turnOpen = false;
                    turnClosed = true;
                    outcome = PiStopOutcome(skippedStop!);
                    closedAt = skipped.Timestamp ?? closedAt;
                }
                else
                {
                    turnOpen = true;
                    outcome = null;
                }
            }
        }

        foreach (var l in Encoding.UTF8.GetString(bytes, parseFrom, complete - parseFrom).Split('\n'))
        {
            if (l.Length == 0) continue;
            // Tool results and system records carry no metadata we keep, and can be huge (images, file contents).
            // The role comes from the record's own message object, never from the rest of the line, so a tool
            // argument or a file that mentions another role cannot pick the branch or skip an assistant record.
            var lineRole = PiLineRole(l);
            if (lineRole == "toolResult")
            {
                // A tool result still means the turn is running; skip the JSON without parsing it.
                turnOpen = true;
                turnClosed = false;
                outcome = null;
                continue;
            }
            if (lineRole == "system") continue;
            // Cheap skips before any JSON is parsed: most other lines are tool results without metadata.
            if (!l.Contains("\"message\"", StringComparison.Ordinal)
                && !l.Contains("\"session\"", StringComparison.Ordinal)
                && !l.Contains("\"model_change\"", StringComparison.Ordinal)
                && !l.Contains("\"thinking_level_change\"", StringComparison.Ordinal)
                && !l.Contains("\"session_info\"", StringComparison.Ordinal))
                continue;
            try
            {
                using var doc = JsonDocument.Parse(l);
                var root = doc.RootElement;
                switch (Str(root, "type"))
                {
                    case "session":
                        id = Str(root, "id") ?? id;
                        cwd = Str(root, "cwd") ?? cwd;
                        break;
                    case "model_change":
                        provider = SanitizeAntigravityLabel(Str(root, "provider"), AgyMaxMetaChars) ?? provider;
                        model = SanitizeAntigravityLabel(Str(root, "modelId"), AgyMaxMetaChars) ?? model;
                        break;
                    case "thinking_level_change":
                        effort = SanitizeAntigravityLabel(Str(root, "thinkingLevel"), AgyMaxMetaChars) ?? effort;
                        break;
                    case "session_info":
                        title = Str(root, "name") ?? title;
                        break;
                    case "message":
                        var m = root.TryGetProperty("message", out var mm) && mm.ValueKind == JsonValueKind.Object ? mm : default;
                        var role = Str(m, "role");
                        var ts = Ts(root) ?? Ts(m);
                        if (role == "assistant")
                        {
                            model = SanitizeAntigravityLabel(Str(m, "model"), AgyMaxMetaChars) ?? model;
                            provider = SanitizeAntigravityLabel(Str(m, "provider"), AgyMaxMetaChars) ?? provider;
                            effort = SanitizeAntigravityLabel(Str(m, "thinkingLevel"), AgyMaxMetaChars) ?? effort;
                            if (m.TryGetProperty("usage", out var u) && u.ValueKind == JsonValueKind.Object)
                            {
                                hasUsage = true;
                                input += Num(u, "input");
                                output += Num(u, "output");
                                cacheRead += Num(u, "cacheRead");
                                cacheWrite += Num(u, "cacheWrite");
                                reasoning += Num(u, "reasoning");
                                // LastInputTokens is the latest request only, like Grok's and Codex's:
                                // the cumulative counters above stay for InputTokens / TotalTokens.
                                // An error or abort carries Pi's EMPTY_USAGE; it must not wipe the context size.
                                var requestInput = Num(u, "input") + Num(u, "cacheRead") + Num(u, "cacheWrite");
                                if (requestInput > 0) lastInputTokens = requestInput;
                            }
                            var stop = Str(m, "stopReason");
                            if (PiClosedStop(stop))
                            {
                                turnOpen = false;
                                turnClosed = true;
                                // A length stop is Pi's truncation-recovery path, so it can be retried like an error.
                                outcome = PiStopOutcome(stop!);
                                closedAt = ts ?? closedAt;
                            }
                            else
                            {
                                // "toolUse" (or nothing yet): the turn continues, the tool results are coming
                                turnOpen = true;
                                outcome = null;
                            }
                            if (m.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.Array)
                            {
                                foreach (var block in c.EnumerateArray())
                                {
                                    if (block.ValueKind == JsonValueKind.Object && Str(block, "type") == "toolCall")
                                        lastTool = SanitizeAntigravityLabel(Str(block, "name"), AgyMaxMetaChars) ?? lastTool;
                                }
                            }
                        }
                        else if (role == "user" || role == "toolResult")
                        {
                            turnOpen = true;
                            turnClosed = false;
                            outcome = null;
                            if (role == "user") turnStartedAt = ts ?? turnStartedAt;
                        }
                        lastActivityAt = ts ?? lastActivityAt;
                        break;
                }
            }
            catch (JsonException) { }
        }

        return new PiDetails(id, cwd, provider, model, effort, hasUsage, tokensComplete, input, output, cacheRead,
            cacheWrite, reasoning, lastInputTokens,
            turnOpen, turnClosed, outcome, turnStartedAt, lastActivityAt, closedAt, lastTool, title,
            start + complete);
    }

    private static bool PiClosedStop(string? stop) => stop is "stop" or "length" or "aborted" or "error";

    private static string PiStopOutcome(string stop) =>
        stop == "stop" ? "completed" : stop == "aborted" ? "aborted" : "failed";

    /// <summary>
    /// The role a message record carries in its own <c>message</c> object, read from the record's first bytes so a
    /// huge tool result is skipped without parsing it. Only the property right after <c>"message":{</c> counts: a
    /// role-like value in a tool argument or in file contents comes later and cannot be mistaken for it. Null when
    /// the line is not a message record or the role is past the quick-scan window; the caller then parses the line.
    /// </summary>
    private static string? PiLineRole(string line)
    {
        var limit = Math.Min(line.Length, PiRoleScanBytes);
        var role = line.IndexOf("\"role\":\"", 0, limit, StringComparison.Ordinal);
        if (role < 0) return null;
        var head = line[..role].TrimEnd(' ', '\t');
        return head.EndsWith("\"message\":{", StringComparison.Ordinal) ? PiJsonString(line, role + 7) : null;
    }

    /// <summary>
    /// The record that a read window starts inside, identified from its first bytes: enough to keep its lifecycle
    /// meaning (a prompt or a tool result keeps the turn open, a finished reply closes it) without parsing or
    /// holding its payload. <paramref name="floor"/> is the last record boundary before the window, so the record
    /// starts at the last line break before the window (or at the floor when there is none).
    /// </summary>
    private static PiRecordHead? ReadPiRecordHead(FileStream fs, long windowStart, long floor)
    {
        var start = floor;
        var chunk = new byte[PiScanChunkBytes];
        for (var pos = windowStart; pos > floor;)
        {
            var n = (int)Math.Min(chunk.Length, pos - floor);
            var at = pos - n;
            fs.Seek(at, SeekOrigin.Begin);
            var read = 0;
            while (read < n)
            {
                var got = fs.Read(chunk, read, n - read);
                if (got <= 0) break;
                read += got;
            }
            if (read == 0) return null;
            var nl = Array.LastIndexOf(chunk, (byte)'\n', read - 1);
            if (nl >= 0)
            {
                start = at + nl + 1;
                break;
            }
            pos = at;
        }

        var head = new byte[PiRecordHeadBytes];
        fs.Seek(start, SeekOrigin.Begin);
        var readHead = 0;
        while (readHead < head.Length)
        {
            var got = fs.Read(head, readHead, head.Length - readHead);
            if (got <= 0) break;
            readHead += got;
        }
        if (readHead == 0) return null;
        // A head cut inside a string is harmless: the fields read here come before any payload.
        var text = Encoding.UTF8.GetString(head, 0, readHead);
        var ts = PiHeadString(text, "\"timestamp\":");
        return new PiRecordHead(PiHeadString(text, "\"type\":"), PiLineRole(text),
            ts is { Length: > 0 } && DateTimeOffset.TryParse(ts, out var at2) ? at2.ToUnixTimeMilliseconds() : null);
    }

    private static string? PiHeadString(string head, string key)
    {
        var at = head.IndexOf(key, StringComparison.Ordinal);
        if (at < 0) return null;
        var value = at + key.Length;
        while (value < head.Length && head[value] is ' ' or '\t') value++;
        return PiJsonString(head, value);
    }

    /// <summary>
    /// The stop reason of the record a read window starts inside: it ends an assistant record, so it is in the
    /// record's own last bytes (the window's dropped first line) and after every payload.
    /// </summary>
    private static string? PiTailStopReason(byte[] bytes, int length)
    {
        if (length <= 0) return null;
        var from = Math.Max(0, length - PiRecordTailBytes);
        var text = Encoding.UTF8.GetString(bytes, from, length - from);
        var at = text.LastIndexOf("\"stopReason\":\"", StringComparison.Ordinal);
        return at < 0 ? null : PiJsonString(text, at + 13);
    }

    /// <summary>The string value of the JSON string literal that opens at <paramref name="quote"/>.</summary>
    private static string? PiJsonString(string text, int quote)
    {
        if (quote >= text.Length || text[quote] != '"') return null;
        for (var i = quote + 1; i < text.Length; i++)
        {
            if (text[i] == '\\')
            {
                i++;
                continue;
            }
            if (text[i] != '"') continue;
            try
            {
                using var doc = JsonDocument.Parse(text[quote..(i + 1)]);
                return doc.RootElement.GetString();
            }
            catch (JsonException) { return null; }
        }
        return null;
    }

    /// <summary>The session header at the start of a ragged file: its id and working directory.</summary>
    private static (string? Id, string? Cwd)? ReadPiHeader(string path)
    {
        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            var buf = new byte[(int)Math.Min(fs.Length, PiHeaderBytes)];
            var read = 0;
            while (read < buf.Length)
            {
                var n = fs.Read(buf, read, buf.Length - read);
                if (n <= 0) break;
                read += n;
            }
            if (read == 0) return null;
            var end = Array.LastIndexOf(buf, (byte)'\n', read - 1);
            if (end < 0) return null;
            foreach (var line in Encoding.UTF8.GetString(buf, 0, end).Split('\n'))
            {
                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;
                    if (Str(root, "type") == "session") return (Str(root, "id"), Str(root, "cwd"));
                }
                catch (JsonException) { }
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        return null;
    }

    private static long Num(JsonElement e, string p) => Long(e, p) ?? 0;

    private static long? Ts(JsonElement e)
    {
        var s = Str(e, "timestamp");
        return s is { Length: > 0 } && DateTimeOffset.TryParse(s, out var dt) ? dt.ToUnixTimeMilliseconds() : null;
    }
}
