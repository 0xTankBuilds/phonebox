using System.Net;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;
using PhoneBox.Agent;

namespace PhoneBox.Agent.Tests;

public sealed class LlamaSwapMonitorTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("llamaswap-test").FullName;

    [Fact(DisplayName = "The refresh reads the version, the loaded models and the totals")]
    public async Task RefreshReadsServer()
    {
        await using var ls = new FakeLlamaSwap();
        var mon = Monitor(ls.Url);
        await mon.RefreshAsync();
        var s = mon.Snapshot()!;
        Assert.True(s.Connected);
        Assert.Equal("v275", s.Version);
        Assert.Equal(["Qwen3.8-27B-GGUF"], s.LoadedModels); // the unloaded and the status-less model are out
        Assert.Equal(199, s.TotalRequests);
        Assert.Equal(11_723_138, s.TotalInputTokens);
        Assert.Equal(98_605, s.TotalOutputTokens);
        Assert.Equal(11_397_352, s.TotalCacheTokens);
        Assert.Equal(38.53, s.GenTps50!.Value, 2);
    }

    [Fact(DisplayName = "Today's counts come from the request log and stop at the local midnight")]
    public async Task TodayCountsOnlyToday()
    {
        await using var ls = new FakeLlamaSwap();
        var now = DateTime.Now;
        var midnightToday = now.Date;
        ls.ActivityEntries = [
            Entry(midnightToday.AddSeconds(1), 100, 10),
            Entry(now, 50, 5),
            Entry(midnightToday.AddDays(-1).AddHours(23).AddMinutes(59).AddSeconds(59), 999, 999),
        ];
        var mon = Monitor(ls.Url);
        await mon.RefreshAsync();
        var s = mon.Snapshot()!;
        Assert.Equal(2, s.TodayRequests);
        Assert.Equal(150, s.TodayInputTokens);
        Assert.Equal(15, s.TodayOutputTokens);
        Assert.Equal(now, s.LastRequestAt!.Value, TimeSpan.FromSeconds(2));
    }

    [Fact(DisplayName = "An address with nothing listening is offline")]
    public async Task OfflineWhenNothingListens()
    {
        // a port that is free: TestPorts allocates and releases one, and the connection is refused
        var port = TestPorts.Listen();
        var url = port.Prefixes.Single().TrimEnd('/');
        port.Stop();
        port.Close();
        var mon = Monitor(url);
        await mon.RefreshAsync();
        var s = mon.Snapshot()!;
        Assert.False(s.Connected);
        Assert.Equal("offline", s.Error);
        Assert.Null(s.Version);
    }

    [Fact(DisplayName = "An empty address keeps the snapshot null")]
    public async Task EmptyUrlKeepsSnapshotNull()
    {
        var mon = Monitor("");
        await mon.RefreshAsync();
        Assert.Null(mon.Snapshot());
    }

    [Fact(DisplayName = "A missing activity log leaves today's counts empty but the totals intact")]
    public async Task MissingActivityDegraded()
    {
        await using var ls = new FakeLlamaSwap(activity: false);
        var mon = Monitor(ls.Url);
        await mon.RefreshAsync();
        var s = mon.Snapshot()!;
        Assert.True(s.Connected);
        Assert.Null(s.TodayRequests);
        Assert.Equal(199, s.TotalRequests);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static LlamaSwapMonitor Monitor(string url)
        => new(NullLogger<LlamaSwapMonitor>.Instance) { UrlOverride = url };

    private static string Entry(DateTime local, long input, long output)
        => $"{{\"timestamp\":\"{local.ToUniversalTime():yyyy-MM-ddTHH:mm:ss.fffZ}\",\"tokens\":{{\"input_tokens\":{input},\"output_tokens\":{output}}}}}";
}

/// <summary>A llama-swap server: /api/version, /v1/models, /api/metrics/stats, /api/metrics/activity.</summary>
internal sealed class FakeLlamaSwap : IAsyncDisposable
{
    private readonly HttpListener _server = TestPorts.Listen();
    private readonly bool _activity;
    private readonly Task _loop;
    public string Url => _server.Prefixes.Single().TrimEnd('/');
    public volatile string[] ActivityEntries = [];

    public FakeLlamaSwap(bool activity = true)
    {
        _activity = activity;
        _loop = Task.Run(Serve);
    }


    private async Task Serve()
    {
        while (_server.IsListening)
        {
            HttpListenerContext ctx;
            try { ctx = await _server.GetContextAsync(); }
            catch (HttpListenerException) { return; }
            catch (ObjectDisposedException) { return; }
            _ = Task.Run(() => Answer(ctx));
        }
    }

    private void Answer(HttpListenerContext ctx)
    {
        var url = ctx.Request.Url!;
        var path = url.AbsolutePath;
        string body = "{}";
        var status = 200;
        switch (path)
        {
            case "/api/version":
                body = """{"version":"v275","commit":"e0274d5","build_date":"2026-11-20T07:15:47Z"}""";
                break;
            case "/v1/models":
                body = """{"data":[{"id":"Qwen3.8-27B-GGUF","status":{"value":"loaded"}},{"id":"Bonsai-27B-Q1_0","status":{"value":"unloaded"}},{"id":"NoStatus"}]}""";
                break;
            case "/api/metrics/stats":
                body = """{"total_requests":199,"total_input_tokens":11723138,"total_output_tokens":98605,"total_cache_tokens":11397352,"gen_histogram":{"p50":38.53,"p95":50.1}}""";
                break;
            case "/api/metrics/activity" when _activity:
                body = """{"data":[""" + string.Join(",", ActivityEntries) + """]}""";
                break;
            default:
                status = 404;
                break;
        }
        var bytes = System.Text.Encoding.UTF8.GetBytes(body);
        ctx.Response.StatusCode = status;
        ctx.Response.ContentType = "application/json";
        ctx.Response.OutputStream.Write(bytes, 0, bytes.Length);
        ctx.Response.OutputStream.Close();
    }

    public async ValueTask DisposeAsync()
    {
        _server.Stop();
        _server.Close();
        try { await _loop.WaitAsync(TimeSpan.FromSeconds(5)); } catch (Exception) { }
    }
}
