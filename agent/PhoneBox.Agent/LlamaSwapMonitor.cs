using System.Globalization;
using System.Text.Json;

namespace PhoneBox.Agent;

/// <summary>
/// One request the router is serving right now. Model and path only - never headers, the remote IP, or anything
/// from the prompt or the answer. Only Id, Model and ElapsedMs reach the phone; Path and RespBytes never leave
/// the server (see the projection in the /api/llamaswap handler).
/// </summary>
public sealed record LlamaSwapInflight(string Id, string Model, string Path, int ElapsedMs, long RespBytes);

/// <summary>
/// One refresh of the llama-swap server: what is connected, which models are loaded, today's counts from its
/// request log, the requests it is serving right now, and the generation speed it measured itself.
/// Counts and model names only - nothing from the prompts or answers ever leaves the server.
/// </summary>
public sealed record LlamaSwapSnapshot(
    bool Connected,
    string? Error,
    string? Version,
    IReadOnlyList<string> LoadedModels,
    double? GenTps50,
    int? TodayRequests,
    long? TodayInputTokens,
    long? TodayOutputTokens,
    // True when the log window (the newest 999 requests) is full and still full of today: the counts above are a
    // lower bound the panel shows as "999+", not a total.
    bool TodayCapped,
    DateTime? LastRequestAt,
    int Inflight,
    // The requests themselves: the elapsed seconds the phone shows for the in-flight count come from here.
    IReadOnlyList<LlamaSwapInflight> InflightRequests);

/// <summary>
/// Polls the local llama-swap server (the OpenAI-compatible LLM router): /api/version, /v1/models,
/// /api/metrics/stats and /api/metrics/activity. Runs only while Settings › General › llama-swap has an address;
/// an empty address polls nothing and the snapshot stays null.
/// </summary>
public sealed class LlamaSwapMonitor(ConfigStore config, ILogger<LlamaSwapMonitor> log) : BackgroundService
{
    internal LlamaSwapMonitor(ILogger<LlamaSwapMonitor> log) : this(null!, log) { }

    /// <summary>For tests: the server address, overriding the setting.</summary>
    internal string? UrlOverride { get; set; }
    /// <summary>For tests: a fake handler instead of the network.</summary>
    internal HttpMessageHandler? HttpHandler { get; init; }
    /// <summary>For tests: the "today" boundary follows this clock.</summary>
    internal Func<DateTime> Now { get; init; } = () => DateTime.Now;
    /// <summary>For tests: the UTC clock the in-flight bookkeeping runs on, so a local DST jump or a timezone
    /// clock change cannot corrupt the elapsed times.</summary>
    internal Func<DateTime> UtcNow { get; init; } = () => DateTime.UtcNow;

    /// <summary>The longest an in-flight request is trusted without its remove event: a dropped event would
    /// otherwise pin the in-flight count - and the phone's fast poll - forever.</summary>
    internal static readonly TimeSpan MaxInflightAge = TimeSpan.FromHours(1);

    /// <summary>The request log window the router keeps and the maximum it answers for (a larger ?limit= is
    /// refused): today's counts are exact only while the newest entries still reach into yesterday.</summary>
    internal const int ActivityWindow = 999;

    /// <summary>The longest one /api/events connection is kept open: a half-dead socket blocks its read forever
    /// and leaves the in-flight list stale until MaxInflightAge, so each connection is dropped after this and
    /// the next one takes a fresh snapshot of what is really in flight.</summary>
    internal static readonly TimeSpan MaxStreamLifetime = TimeSpan.FromMinutes(10);
    /// <summary>For tests: the per-connection lifetime the stream gives up after.</summary>
    internal TimeSpan StreamLifetime { get; init; } = MaxStreamLifetime;

    private HttpClient? _http;
    private HttpClient Http => _http ??= NewHttpClient(HttpHandler);

    // The /api/events stream is long-lived: the poll client's 8-second timeout would hang it up, so it gets its own.
    private HttpClient? _sseHttp;
    private HttpClient SseHttp => _sseHttp ??= NewHttpClient(HttpHandler, infinite: true);

    private static HttpClient NewHttpClient(HttpMessageHandler? handler, bool infinite = false)
    {
        var http = new HttpClient(handler ?? new SocketsHttpHandler()) { Timeout = infinite ? System.Threading.Timeout.InfiniteTimeSpan : TimeSpan.FromSeconds(8) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("PhoneBox/" + AppPaths.Version);
        return http;
    }

    private volatile LlamaSwapSnapshot? _snapshot;
    private readonly SemaphoreSlim _gate = new(1, 1);
    // The active /api/events read, if any: a cleared or changed address cancels it - a quiet old router would
    // otherwise keep the old stream, and its in-flight list, for the rest of the stream's lifetime.
    private volatile CancellationTokenSource? _activeRead;
    private Uri? _lastUrl;

    /// <summary>The last refresh; null while nothing has been polled yet or the address is not set. Also null
    /// the moment the address changes and until the next refresh: the snapshot belongs to the router the old
    /// address pointed at, and it must not keep showing as if it were the new router's.</summary>
    public LlamaSwapSnapshot? Snapshot()
        => HasAddress && _lastUrl is { } last && UrlMatches(last) ? _snapshot : null;

    /// <summary>Whether an address is configured at all. The endpoint reports it separately from the snapshot:
    /// the snapshot is also null before the first refresh has finished, and the widget must say "set the
    /// address", not "offline", in that case.</summary>
    public bool HasAddress
        => Uri.TryCreate(Url(), UriKind.Absolute, out var u) && (u.Scheme == Uri.UriSchemeHttp || u.Scheme == Uri.UriSchemeHttps);

    private readonly object _inflightLock = new();
    private readonly List<LlamaSwapInflight> _inflight = new();
    // When each request first showed up, so its elapsed time keeps growing between events instead of freezing at the
    // last one the router sent.
    private readonly Dictionary<string, DateTime> _firstSeen = new();

    /// <summary>What leaves the machine per in-flight request: the id, the model, and how long the request has
    /// been running. Path and RespBytes stay on the server; the /api/llamaswap handler builds its answer from
    /// this projection only.</summary>
    public sealed record LlamaSwapInflightView(string Id, string Model, int ElapsedMs);
    public static IReadOnlyList<LlamaSwapInflightView> InflightView(IReadOnlyList<LlamaSwapInflight> inflight)
        => inflight.Select(r => new LlamaSwapInflightView(r.Id, r.Model, r.ElapsedMs)).ToList();

    /// <summary>The requests the router is serving right now, from the /api/events stream. ElapsedMs is live: it is
    /// measured against the UTC clock, not frozen at the last event the router sent. Entries that survived a
    /// dropped remove event are dropped after <see cref="MaxInflightAge"/>, so a lost event can never pin the
    /// in-flight count (and the phone's fast poll) forever.</summary>
    public IReadOnlyList<LlamaSwapInflight> Inflight { get {
        lock (_inflightLock)
        {
            var now = UtcNow();
            var stale = _inflight.Where(r => now - _firstSeen[r.Id] > MaxInflightAge).Select(r => r.Id).ToList();
            foreach (var id in stale)
                _firstSeen.Remove(id);
            _inflight.RemoveAll(r => stale.Contains(r.Id));
            return _inflight
                .Select(r => r with { ElapsedMs = (int)Math.Min(Math.Max(0, (now - _firstSeen[r.Id]).TotalMilliseconds), int.MaxValue) })
                .ToList();
        }
    } }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (config is null) return; // tests drive RefreshAsync directly
        await Task.WhenAll(PollLoopAsync(stoppingToken), StreamLoopAsync(stoppingToken));
    }

    private async Task PollLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            await RefreshAsync(ct);
            var seconds = Math.Clamp(config.Get()?.LlamaSwap.RefreshSeconds ?? 60, 10, 3600);
            await Task.Delay(TimeSpan.FromSeconds(seconds), ct);
        }
    }

    // The router pushes its in-flight request list over the same event stream its own web UI subscribes to. The
    // stream is the only view of the requests while they run: the REST endpoints only know the finished ones.
    private async Task StreamLoopAsync(CancellationToken ct)
    {
        var backoff = 1.0;
        while (!ct.IsCancellationRequested)
        {
            var ok = false;
            try
            {
                if (Uri.TryCreate(Url(), UriKind.Absolute, out var baseUri) && (baseUri.Scheme == Uri.UriSchemeHttp || baseUri.Scheme == Uri.UriSchemeHttps))
                {
                    await StreamOnceAsync(baseUri, ct);
                    ok = true;
                }
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
            catch (Exception ex) { log.LogDebug(ex, "llama-swap events stream failed"); }
            backoff = ok ? 1.0 : Math.Min(backoff * 2, 30);
            await Task.Delay(TimeSpan.FromSeconds(backoff), ct);
        }
    }

    /// <summary>Reads one /api/events stream until the server closes it. The router resends the whole in-flight
    /// list as a snapshot on every connection, so the state needs no handshake.</summary>
    internal async Task StreamOnceAsync(Uri baseUri, CancellationToken ct = default)
    {
        using var req = new HttpRequestMessage(HttpMethod.Get, new Uri(baseUri, "api/events"));
        req.Headers.Accept.ParseAdd("text/event-stream");
        using var res = await SseHttp.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
        if (!res.IsSuccessStatusCode) throw new HttpRequestException($"llama-swap /api/events -> {(int)res.StatusCode}");
        ClearInflight();
        using var stream = await res.Content.ReadAsStreamAsync(ct);
        using var reader = new StreamReader(stream);
        try { await ReadLoopAsync(reader, baseUri, ct); }
        catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
    }

    private async Task ReadLoopAsync(StreamReader reader, Uri baseUri, CancellationToken ct)
    {
        using var lifetime = new CancellationTokenSource(StreamLifetime);
        using var readCt = CancellationTokenSource.CreateLinkedTokenSource(ct, lifetime.Token);
        _activeRead = readCt;
        try
        {
            // The loop ends, not just the connection: a cleared or changed address stops the stream - the old
            // router's events would otherwise keep feeding the new one - and the lifetime ends a healthy stream
            // the same way the server closing it does, a normal end, not a failure.
            while (!ct.IsCancellationRequested && UrlMatches(baseUri))
            {
                var line = await reader.ReadLineAsync(readCt.Token);
                if (line is null) break;
                if (!line.StartsWith("data:", StringComparison.Ordinal)) continue;
                var payload = line["data:".Length..].TrimStart();
                JsonDocument doc;
                try { doc = JsonDocument.Parse(payload); } catch (JsonException) { continue; }
                using (doc)
                {
                    if (doc.RootElement.ValueKind != JsonValueKind.Object) continue;
                    if (!doc.RootElement.TryGetProperty("type", out var type) || type.GetString() != "inflight") continue;
                    if (!doc.RootElement.TryGetProperty("data", out var data)) continue;
                    if (data.ValueKind == JsonValueKind.String) ApplyInflightEvent(data.GetString()!);
                    else if (data.ValueKind == JsonValueKind.Object) ApplyInflightEvent(data.GetRawText());
                }
            }
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
        catch (OperationCanceledException)
        {
            // The lifetime, or a cleared or changed address, ended the read: a normal end, so the reconnect loop
            // counts it as one and opens the stream to the current address.
        }
        finally
        {
            if (_activeRead == readCt) _activeRead = null;
        }
    }

    /// <summary>Whether the configured address is still the one this stream came from. Comparing Uris, not
    /// strings: the settings may show <c>http://h:1/</c> where the stream came from <c>http://h:1</c>.</summary>
    private bool UrlMatches(Uri baseUri)
        => Uri.TryCreate(Url(), UriKind.Absolute, out var cur) && cur == baseUri;

    /// <summary>Cancels the /api/events read, if there is one: a cleared or changed address must not keep the
    /// old router's stream - quiet or not - feeding the panel until the stream's lifetime runs out.</summary>
    private void EndStream()
    {
        var read = _activeRead;
        _activeRead = null;
        try { read?.Cancel(); }
        catch (ObjectDisposedException)
        {
            // the read loop already ended and disposed its token source: there is nothing left to cancel
        }
    }

    /// <summary>
    /// Applies one in-flight event from the router: "snapshot" replaces the whole list, "upsert" adds or updates
    /// one request, "remove" drops it. Malformed events and entries are dropped, not fatal: a bad entry must
    /// not wipe the list it arrives in.
    /// </summary>
    internal void ApplyInflightEvent(string payload)
    {
        try
        {
            using var doc = JsonDocument.Parse(payload);
            var op = doc.RootElement.TryGetProperty("operation", out var o) && o.ValueKind == JsonValueKind.String ? o.GetString() : null;
            lock (_inflightLock)
            {
                var now = UtcNow();
                void Add(LlamaSwapInflight r)
                {
                    // a re-upsert of the same request keeps its original start; only a first sighting sets it
                    if (!_firstSeen.ContainsKey(r.Id)) _firstSeen[r.Id] = now - TimeSpan.FromMilliseconds(r.ElapsedMs);
                    _inflight.Add(r);
                }
                if (op == "snapshot")
                {
                    _inflight.Clear();
                    _firstSeen.Clear();
                    if (doc.RootElement.TryGetProperty("requests", out var rs) && rs.ValueKind == JsonValueKind.Array)
                        foreach (var r in rs.EnumerateArray())
                            if (Entry(r) is { } r1) Add(r1);
                }
                else if (op == "upsert" && doc.RootElement.TryGetProperty("request", out var one) && Entry(one) is { } r2)
                {
                    _inflight.RemoveAll(r => r.Id == r2.Id);
                    Add(r2);
                }
                else if (op == "remove" && doc.RootElement.TryGetProperty("id", out var id) && IdOf(id) is { } gone)
                {
                    _inflight.RemoveAll(r => r.Id == gone);
                    _firstSeen.Remove(gone);
                }
            }
        }
        catch (Exception ex)
        {
            log.LogDebug(ex, "llama-swap in-flight event did not parse");
        }
    }

    /// <summary>The id in the one spelling this side keeps: a numbered request is stored as its number, so an
    /// upsert and its remove always meet.</summary>
    private static string? IdOf(JsonElement id)
        => id.ValueKind == JsonValueKind.String ? id.GetString()
         : id.ValueKind == JsonValueKind.Number && id.TryGetInt64(out var n) ? n.ToString(CultureInfo.InvariantCulture)
         : null;

    private static LlamaSwapInflight? Entry(JsonElement e)
    {
        if (e.ValueKind != JsonValueKind.Object || !e.TryGetProperty("id", out var id)) return null;
        // the router may number a request instead of naming it: both ids are opaque to this side, and a remove
        // must find an upsert either way, so the two kinds share one translation
        if (IdOf(id) is not { } idStr) return null;
        static string S(JsonElement p, string k) => p.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString()! : "";
        // absent stays 0 (the router omits a field it has not filled yet); present but not an integer is
        // malformed, and drops the entry, not the list
        static long? N(JsonElement p, string k)
            => !p.TryGetProperty(k, out var v) ? 0
             : v.ValueKind == JsonValueKind.Number && v.TryGetInt64(out var n) ? n
             : (long?)null;
        if (N(e, "elapsed_ms") is not { } elapsed || N(e, "resp_bytes") is not { } bytes) return null;
        return new LlamaSwapInflight(idStr, S(e, "model"), S(e, "req_path"), (int)Math.Min(elapsed, int.MaxValue), bytes);
    }

    private void ClearInflight()
    {
        lock (_inflightLock) { _inflight.Clear(); _firstSeen.Clear(); }
    }

    /// <summary>Reads the four endpoints once. Concurrent refreshes are serialized, not duplicated.</summary>
    internal async Task RefreshAsync(CancellationToken ct = default)
    {
        var url = Url();
        if (!Uri.TryCreate(url, UriKind.Absolute, out var baseUri) || (baseUri.Scheme != Uri.UriSchemeHttp && baseUri.Scheme != Uri.UriSchemeHttps))
        {
            _snapshot = null;
            // No (valid) address: the phone must not keep seeing the last router's requests as in flight.
            ClearInflight();
            EndStream();
            _lastUrl = null;
            return;
        }
        if (_lastUrl is not null && _lastUrl != baseUri)
        {
            // A new address is a new router: the old in-flight list, the stream to the old router, and the old
            // snapshot belong to the old server. Until the new one has answered, the panel says "loading" - the
            // old router's version and counts must not keep showing as if they were the new router's.
            ClearInflight();
            EndStream();
            _snapshot = null;
        }
        _lastUrl = baseUri;
        await _gate.WaitAsync(ct);
        try
        {
            var version = await GetAsync(baseUri, "api/version", ct);
            if (version is null)
            {
                // Only while the stream is not feeding: a healthy stream re-sends the list on its next
                // connection, so one failed REST call must not wipe what it is showing.
                if (_activeRead is null)
                {
                    ClearInflight();
                    _snapshot = Offline();
                }
                return;
            }
            var models = await GetAsync(baseUri, "v1/models", ct);
            var stats = await GetAsync(baseUri, "api/metrics/stats", ct);
            var activity = await GetAsync(baseUri, $"api/metrics/activity?limit={ActivityWindow}", ct);

            var genTps50 = ParseGenTps50(stats);
            var today = ParseToday(activity);
            var inflight = Inflight; // one read: the count and the list it belongs to must agree
            _snapshot = new LlamaSwapSnapshot(
                true, null, ParseVersion(version), ParseLoadedModels(models),
                genTps50, today.requests, today.input, today.output, today.capped,
                today.lastRequestAt, inflight.Count, inflight);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
        catch (Exception ex)
        {
            log.LogDebug(ex, "llama-swap refresh failed");
            if (_activeRead is null)
            {
                ClearInflight();
                _snapshot = Offline();
            }
        }
        finally { _gate.Release(); }
    }

    private LlamaSwapSnapshot Offline() => new(false, "offline", null, [], null, null, null, null, false, null, 0, []);

    private string Url()
    {
        // Config.Normalize keeps the trailing slash when the address has a path, and the relative endpoints
        // (api/version, api/events) must resolve under that path - so it is kept, not trimmed.
        var u = UrlOverride ?? config?.Get()?.LlamaSwap.Url ?? "";
        return u.Trim();
    }

    private async Task<JsonElement?> GetAsync(Uri baseUri, string path, CancellationToken ct)
    {
        var res = await Http.GetAsync(new Uri(baseUri, path), ct);
        if (!res.IsSuccessStatusCode) return null;
        using var doc = await JsonDocument.ParseAsync(await res.Content.ReadAsStreamAsync(ct),
            new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip }, ct);
        return doc.RootElement.Clone();
    }

    private static string? ParseVersion(JsonElement? el)
        => el is { ValueKind: JsonValueKind.Object } o && o.TryGetProperty("version", out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static List<string> ParseLoadedModels(JsonElement? el)
    {
        var list = new List<string>();
        if (el is { ValueKind: JsonValueKind.Object } o && o.TryGetProperty("data", out var data) && data.ValueKind == JsonValueKind.Array)
        {
            foreach (var m in data.EnumerateArray())
            {
                var loaded = m.TryGetProperty("status", out var st)
                    && st.ValueKind == JsonValueKind.Object
                    && st.TryGetProperty("value", out var v)
                    && v.ValueKind == JsonValueKind.String
                    && v.GetString() == "loaded";
                if (!loaded || !m.TryGetProperty("id", out var id) || id.ValueKind != JsonValueKind.String) continue;
                list.Add(id.GetString()!);
            }
        }
        return list;
    }

    /// <summary>The generation speed the router measured itself (p50). The lifetime totals the stats endpoint
    /// also keeps have no reader on the phone, so they stay on the server like the rest of it.</summary>
    private static double? ParseGenTps50(JsonElement? el)
        => el is { ValueKind: JsonValueKind.Object } o && o.TryGetProperty("gen_histogram", out var gh)
            && gh.ValueKind == JsonValueKind.Object && gh.TryGetProperty("p50", out var p50)
            && p50.ValueKind == JsonValueKind.Number ? p50.GetDouble() : null;

    private static long? Num(JsonElement o, string key)
        => o.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt64() : null;

    /// <summary>
    /// Today's counts from the request log (newest first, the newest <see cref="ActivityWindow"/> entries):
    /// the requests made since the local midnight, and their input/output tokens. Null when the log endpoint is
    /// unavailable. When the window is full and its oldest entry is still from today, the true counts are at
    /// least as high, so the panel shows the lower bound with a "+", not a total.
    /// </summary>
    private (int? requests, long? input, long? output, DateTime? lastRequestAt, bool capped) ParseToday(JsonElement? el)
    {
        if (el is not { ValueKind: JsonValueKind.Object } o || !o.TryGetProperty("data", out var data) || data.ValueKind != JsonValueKind.Array)
            return (null, null, null, null, false);
        var day = Now().Date;
        var requests = 0L;
        var input = 0L;
        var output = 0L;
        DateTime? last = null;
        DateTime? oldest = null; // the array is newest-first: the last timestamp seen is the oldest one
        foreach (var entry in data.EnumerateArray())
        {
            DateTime? ts = entry.TryGetProperty("timestamp", out var t) && t.ValueKind == JsonValueKind.String
                && DateTime.TryParse(t.GetString(), CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var dt)
                ? dt.ToLocalTime()
                : null;
            if (ts is not null)
            {
                if (last is null || ts > last) last = ts;
                oldest = ts;
            }
            if (ts is null || ts < day) continue;
            requests++;
            input += entry.TryGetProperty("tokens", out var tk) && tk.ValueKind == JsonValueKind.Object ? Num(tk, "input_tokens") ?? 0 : 0;
            output += entry.TryGetProperty("tokens", out var tk2) && tk2.ValueKind == JsonValueKind.Object ? Num(tk2, "output_tokens") ?? 0 : 0;
        }
        return (checked((int)requests), input, output, last, data.GetArrayLength() == ActivityWindow && oldest is { } oldestTs && oldestTs >= day);
    }
}
