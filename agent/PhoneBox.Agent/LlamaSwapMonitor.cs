using System.Globalization;
using System.Text.Json;

namespace PhoneBox.Agent;

/// <summary>
/// One refresh of the llama-swap server: what is connected, which models are loaded, the totals it keeps, today's
/// counts from its request log, and the generation speed it measured itself. Counts and model names only - nothing
/// from the prompts or answers ever leaves the server.
/// </summary>
public sealed record LlamaSwapSnapshot(
    bool Connected,
    string? Error,
    string? Version,
    IReadOnlyList<string> LoadedModels,
    long TotalRequests,
    long TotalInputTokens,
    long TotalOutputTokens,
    long TotalCacheTokens,
    double? GenTps50,
    int? TodayRequests,
    long? TodayInputTokens,
    long? TodayOutputTokens,
    DateTime? LastRequestAt,
    DateTime CheckedAt);

/// <summary>
/// Polls the local llama-swap server (the OpenAI-compatible LLM router): /api/version, /v1/models,
/// /api/metrics/stats and /api/metrics/activity. Runs only while Settings › General › llama-swap has an address;
/// an empty address polls nothing and the snapshot stays null.
/// </summary>
public sealed class LlamaSwapMonitor(ConfigStore config, ILogger<LlamaSwapMonitor> log) : BackgroundService
{
    internal LlamaSwapMonitor(ILogger<LlamaSwapMonitor> log) : this(null!, log) { }

    /// <summary>For tests: the server address, overriding the setting.</summary>
    internal string? UrlOverride { get; init; }
    /// <summary>For tests: a fake handler instead of the network.</summary>
    internal HttpMessageHandler? HttpHandler { get; init; }
    /// <summary>For tests: the "today" boundary follows this clock.</summary>
    internal Func<DateTime> Now { get; init; } = () => DateTime.Now;

    private HttpClient? _http;
    private HttpClient Http => _http ??= NewHttpClient(HttpHandler);

    private static HttpClient NewHttpClient(HttpMessageHandler? handler)
    {
        var http = new HttpClient(handler ?? new SocketsHttpHandler()) { Timeout = TimeSpan.FromSeconds(8) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("PhoneBox/" + AppPaths.Version);
        return http;
    }

    private volatile LlamaSwapSnapshot? _snapshot;
    private readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>The last refresh; null while nothing has been polled yet or the address is not set.</summary>
    public LlamaSwapSnapshot? Snapshot() => _snapshot;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (config is null) return; // tests drive RefreshAsync directly
        while (!stoppingToken.IsCancellationRequested)
        {
            await RefreshAsync(stoppingToken);
            var seconds = Math.Clamp(config.Get()?.LlamaSwap.RefreshSeconds ?? 60, 10, 3600);
            await Task.Delay(TimeSpan.FromSeconds(seconds), stoppingToken);
        }
    }

    /// <summary>Reads the four endpoints once. Concurrent refreshes are serialized, not duplicated.</summary>
    internal async Task RefreshAsync(CancellationToken ct = default)
    {
        var url = Url();
        if (!Uri.TryCreate(url, UriKind.Absolute, out var baseUri) || (baseUri.Scheme != Uri.UriSchemeHttp && baseUri.Scheme != Uri.UriSchemeHttps))
        {
            _snapshot = null;
            return;
        }
        await _gate.WaitAsync(ct);
        try
        {
            var version = await GetAsync(baseUri, "api/version", ct);
            if (version is null)
            {
                _snapshot = Offline();
                return;
            }
            var models = await GetAsync(baseUri, "v1/models", ct);
            var stats = await GetAsync(baseUri, "api/metrics/stats", ct);
            var activity = await GetAsync(baseUri, "api/metrics/activity?limit=999", ct);

            var (totals, genTps50) = ParseTotals(stats);
            var today = ParseToday(activity);
            _snapshot = new LlamaSwapSnapshot(
                true, null, ParseVersion(version), ParseLoadedModels(models),
                totals.requests, totals.input, totals.output, totals.cache, genTps50,
                today.requests, today.input, today.output, today.lastRequestAt,
                Now());
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
        catch (Exception ex)
        {
            log.LogDebug(ex, "llama-swap refresh failed");
            _snapshot = Offline();
        }
        finally { _gate.Release(); }
    }

    private LlamaSwapSnapshot Offline() => new(false, "offline", null, [], 0, 0, 0, 0, null, null, null, null, null, Now());

    private string Url()
    {
        var u = UrlOverride ?? config?.Get()?.LlamaSwap.Url ?? "";
        return u.Trim().TrimEnd('/');
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

    private static ((long requests, long input, long output, long cache) totals, double? genTps50) ParseTotals(JsonElement? el)
    {
        long requests = 0, input = 0, output = 0, cache = 0;
        double? gen = null;
        if (el is { ValueKind: JsonValueKind.Object } o)
        {
            requests = Num(o, "total_requests");
            input = Num(o, "total_input_tokens");
            output = Num(o, "total_output_tokens");
            cache = Num(o, "total_cache_tokens");
            if (o.TryGetProperty("gen_histogram", out var gh) && gh.ValueKind == JsonValueKind.Object && gh.TryGetProperty("p50", out var p50) && p50.ValueKind == JsonValueKind.Number)
                gen = p50.GetDouble();
        }
        return ((requests, input, output, cache), gen);
    }

    private static long Num(JsonElement o, string key)
        => o.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt64() : 0;

    /// <summary>
    /// Today's counts from the request log (newest first, the last 999 entries): the requests made since the local
    /// midnight, and their input/output tokens. Null when the log endpoint is unavailable.
    /// </summary>
    private (int? requests, long? input, long? output, DateTime? lastRequestAt) ParseToday(JsonElement? el)
    {
        if (el is not { ValueKind: JsonValueKind.Object } o || !o.TryGetProperty("data", out var data) || data.ValueKind != JsonValueKind.Array)
            return (null, null, null, null);
        var day = Now().Date;
        var requests = 0L;
        var input = 0L;
        var output = 0L;
        DateTime? last = null;
        foreach (var entry in data.EnumerateArray())
        {
            DateTime? ts = entry.TryGetProperty("timestamp", out var t) && t.ValueKind == JsonValueKind.String
                && DateTime.TryParse(t.GetString(), CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var dt)
                ? dt.ToLocalTime()
                : null;
            if (ts is not null && (last is null || ts > last)) last = ts;
            if (ts is null || ts < day) continue;
            requests++;
            input += entry.TryGetProperty("tokens", out var tk) && tk.ValueKind == JsonValueKind.Object ? Num(tk, "input_tokens") : 0;
            output += entry.TryGetProperty("tokens", out var tk2) && tk2.ValueKind == JsonValueKind.Object ? Num(tk2, "output_tokens") : 0;
        }
        return (checked((int)requests), input, output, last);
    }
}
