using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;

namespace PhoneBox.Agent;

public sealed partial class SessionMonitor
{
    // Pi (pi-coding-agent) writes one JSONL file per session in ~/.pi/agent/sessions/<project slug>/<timestamp>_<id>.jsonl.
    // The monitor keeps only metadata: the "session" record's cwd and id, each "model_change"'s provider/modelId, each
    // "thinking_level_change"'s level, the title from "session_info" (while titles are shared), and, from assistant
    // "message" records, token usage, the stop reason, model, effort and tool names. Prompts, replies and tool
    // arguments are never kept or sent.

    private sealed record PiDetails(string? Id, string? Cwd, string? Provider, string? Model, string? Effort,
        bool HasUsage, long Input, long Output, long CacheRead, long CacheWrite, long Reasoning,
        long LastInputTokens,
        bool TurnOpen, bool TurnClosed, string? Outcome, long? TurnStartedAt, long? LastActivityAt,
        string? LastTool, string? Title, long ReadOffset);

    private string? _piDiscoveryRoot;
    private long _nextPiDiscovery;
    private string[] _piCandidateFiles = [];
    private readonly ConcurrentDictionary<string, (DateTime LastWrite, long Length, PiDetails Details)> _piDetails = new();

    private static readonly ConcurrentDictionary<string, (long LoadedTick, Dictionary<string, Dictionary<string, long>> Windows)> _piContextWindows = new();
    private const int PiContextWindowRefreshMs = 60_000;

    /// <summary>
    /// Recent Pi agent sessions, from ~/.pi/agent/sessions (or sessionsRoot for tests). The first complete record is
    /// the "session" record with the working directory; until it has been read the session is not shown.
    /// </summary>
    internal IEnumerable<AgentSession> ReadPi(string? sessionsRoot = null, bool titles = false)
    {
        var root = sessionsRoot ?? Path.Combine(Home, ".pi", "agent", "sessions");
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
        if (project.Length == 0) project = "pi";
        var id = d.Id ?? Path.GetFileNameWithoutExtension(f.Name);
        var age = DateTime.UtcNow - f.LastWriteTimeUtc;
        // The turn is open while the last record is a user prompt, a tool result, or an assistant reply that carries
        // a tool call; a final assistant reply (stopReason "stop", "length", "aborted" or "error") closes it.
        var open = d.TurnOpen && age < TimeSpan.FromMinutes(3);
        var contextWindow = ContextWindows(contextStore).TryGetValue(d.Provider ?? "", out var models)
            && models.TryGetValue(d.Model ?? "", out var w) ? (long?)w : null;
        var hasInput = d.HasUsage && (d.Input > 0 || d.CacheRead > 0 || d.CacheWrite > 0);
        return new AgentSession(
            Id: "pi:" + id,
            Tool: "pi",
            Name: $"{project.ToLowerInvariant()}-{id[..Math.Min(4, id.Length)]}",
            Project: project,
            Status: open ? "busy" : "idle",
            WaitingFor: null,
            Since: d.LastActivityAt ?? new DateTimeOffset(f.LastWriteTimeUtc).ToUnixTimeMilliseconds(),
            StartedAt: new DateTimeOffset(f.CreationTimeUtc).ToUnixTimeMilliseconds(),
            Model: d.Model, Effort: d.Effort,
            TotalTokens: d.HasUsage ? d.Input + d.CacheRead + d.CacheWrite + d.Output + d.Reasoning : null,
            InputTokens: hasInput ? d.Input + d.CacheWrite : null,
            OutputTokens: d.HasUsage ? d.Output : null,
            CachedInputTokens: d.HasUsage ? d.CacheRead : null,
            ReasoningTokens: d.HasUsage && d.Reasoning > 0 ? d.Reasoning : null,
            LastInputTokens: d.HasUsage ? d.LastInputTokens : null,
            ContextWindow: contextWindow,
            TurnStartedAt: open ? d.TurnStartedAt : null,
            Outcome: open ? null : d.TurnClosed ? d.Outcome : null,
            LastTool: d.LastTool,
            Title: titles ? PiCleanTitle(d.Title) : null);
    }

    /// <summary>The session title Pi wrote for the chat, trimmed and capped like the others'.</summary>
    internal static string? PiCleanTitle(string? title) =>
        string.IsNullOrWhiteSpace(title) ? null : title.Trim() is { Length: > MaxTitleChars } t ? t[..MaxTitleChars] : title.Trim();

    /// <summary>
    /// provider → model → context window, from Pi's own model store (~/.pi/agent/models-store.json, one level up from
    /// the sessions root); empty for a provider the store does not know. Re-read at most once a minute.
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
                if (provider.Value.ValueKind != JsonValueKind.Object) continue;
                var models = new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
                foreach (var model in provider.Value.EnumerateObject())
                {
                    // TryGetInt64 throws on a non-number (null, string), so check the kind first:
                    // one bad entry must not hide every Pi session.
                    var w = model.Value.ValueKind == JsonValueKind.Object
                        && model.Value.TryGetProperty("contextWindow", out var wEl)
                        && wEl.ValueKind == JsonValueKind.Number
                        && wEl.TryGetInt64(out var v) ? v : 0;
                    if (w > 0) models[model.Name] = w;
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
    /// Scans metadata once, then follows appended lines. Null while the file has no complete record yet.
    /// </summary>
    private static PiDetails? ReadPiDetails(string path, PiDetails? previous = null, long previousLength = 0)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        var start = previous is null ? 0 : previousLength;
        fs.Seek(start, SeekOrigin.Begin);
        using var buffer = new MemoryStream();
        fs.CopyTo(buffer);
        var bytes = buffer.ToArray();
        var complete = Array.LastIndexOf(bytes, (byte)'\n') + 1;
        if (complete <= 0) return previous; // nothing new, or still no complete record

        var id = previous?.Id;
        var cwd = previous?.Cwd;
        var provider = previous?.Provider;
        var model = previous?.Model;
        var effort = previous?.Effort;
        var hasUsage = previous?.HasUsage ?? false;
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
        var lastTool = previous?.LastTool;
        var title = previous?.Title;

        foreach (var l in Encoding.UTF8.GetString(bytes, 0, complete).Split('\n'))
        {
            if (l.Length == 0) continue;
            // Cheap skips before any JSON is parsed: most lines are tool results without metadata.
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
                        provider = Str(root, "provider") ?? provider;
                        model = Str(root, "modelId") ?? model;
                        break;
                    case "thinking_level_change":
                        effort = Str(root, "thinkingLevel") ?? effort;
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
                            model = Str(m, "model") ?? model;
                            provider = Str(m, "provider") ?? provider;
                            effort = Str(m, "thinkingLevel") ?? effort;
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
                                lastInputTokens = Num(u, "input") + Num(u, "cacheRead") + Num(u, "cacheWrite");
                            }
                            var stop = Str(m, "stopReason");
                            if (stop is "stop" or "length" or "aborted" or "error")
                            {
                                turnOpen = false;
                                turnClosed = true;
                                outcome = stop is "stop" or "length" ? "completed" : stop == "aborted" ? "aborted" : "failed";
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
                                        lastTool = Str(block, "name") ?? lastTool;
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

        return new PiDetails(id, cwd, provider, model, effort, hasUsage, input, output, cacheRead, cacheWrite, reasoning,
            lastInputTokens,
            turnOpen, turnClosed, outcome, turnStartedAt, lastActivityAt, lastTool, title, start + complete);
    }

    private static long Num(JsonElement e, string p)
        => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(p, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt64() : 0;

    private static long? Ts(JsonElement e)
    {
        var s = Str(e, "timestamp");
        return s is { Length: > 0 } && DateTimeOffset.TryParse(s, out var dt) ? dt.ToUnixTimeMilliseconds() : null;
    }
}
