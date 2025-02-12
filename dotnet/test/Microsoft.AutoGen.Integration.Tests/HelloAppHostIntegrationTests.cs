// Copyright (c) Microsoft Corporation. All rights reserved.
// HelloAppHostIntegrationTests.cs

using System.Text.Json;
using Xunit.Abstractions;

namespace Microsoft.AutoGen.Integration.Tests;

public class HelloAppHostIntegrationTests(ITestOutputHelper testOutput)
{
    private const string AppHostAssemblyName = "XlangTests.AppHost";

    [Fact]
    public async Task AppHostRunsCleanly()
    {
        var appHostPath = GetAssemblyPath(AppHostAssemblyName);
        var appHost = await DistributedApplicationTestFactory.CreateAsync(appHostPath, testOutput);
        await using var app = await appHost.BuildAsync().WaitAsync(TimeSpan.FromSeconds(15));

        await app.StartAsync().WaitAsync(TimeSpan.FromSeconds(120));
        await app.WaitForResourcesAsync().WaitAsync(TimeSpan.FromSeconds(120));

        app.EnsureNoErrorsLogged();
        await app.StopAsync().WaitAsync(TimeSpan.FromSeconds(15));
    }

    [Fact]
    public async Task AppHostLogsHelloAgentE2E()
    {
        var testEndpoints = new TestEndpoints(AppHostAssemblyName, new() {
            { "backend", ["/"] }
        });
        var appHostPath = GetAssemblyPath(AppHostAssemblyName);
        var appHost = await DistributedApplicationTestFactory.CreateAsync(appHostPath, testOutput);
        await using var app = await appHost.BuildAsync().WaitAsync(TimeSpan.FromSeconds(15));

        await app.StartAsync().WaitAsync(TimeSpan.FromSeconds(120));
        await app.WaitForResourcesAsync().WaitAsync(TimeSpan.FromSeconds(120));
        if (testEndpoints.WaitForResources?.Count > 0)
        {
            // Wait until each resource transitions to the required state
            var timeout = TimeSpan.FromMinutes(5);
            foreach (var (ResourceName, TargetState) in testEndpoints.WaitForResources)
            {
                await app.WaitForResource(ResourceName, TargetState).WaitAsync(timeout);
            }
        }
        //sleep to make sure the app is running
        await Task.Delay(20000);
        app.EnsureNoErrorsLogged();
        app.EnsureLogContains("HelloAgent said Goodbye");
        app.EnsureLogContains("Wild Hello from Python!");

        await app.StopAsync().WaitAsync(TimeSpan.FromSeconds(15));
    }

    [Fact]
    public async Task AppHostLogsHelloAgentPythonSendsHelloAgentsDotNetReceives()
    {
        //Prepare
        var appHostPath = GetAssemblyPath(AppHostAssemblyName);
        var appHost = await DistributedApplicationTestFactory.CreateAsync(appHostPath, testOutput);
        await using var app = await appHost.BuildAsync().WaitAsync(TimeSpan.FromSeconds(15));
        await app.StartAsync().WaitAsync(TimeSpan.FromSeconds(120));
        await app.WaitForResourcesAsync(new[] { KnownResourceStates.Running }).WaitAsync(TimeSpan.FromSeconds(120));
        
        //Act
        //var backendResourceName = "AgentHost";
        var dotNetResourceName = "HelloAgentTestsDotNET";
        //var pythonResourceName = "HelloAgentTestsPython";
        var expectedMessage = "Hello from Python!";
        var containsExpectedMessage = false;
        app.EnsureNoErrorsLogged();
        containsExpectedMessage = await app.WaitForExpectedMessageInResourceLogs(dotNetResourceName, expectedMessage, TimeSpan.FromSeconds(120));
        await app.StopAsync();

        //Assert
        Assert.True(containsExpectedMessage);
    }

    [Fact]
    public async Task AppHostLogsHelloAgentPythonSendsAgentHostReceives()
    {
        //Prepare
        var appHostPath = GetAssemblyPath(AppHostAssemblyName);
        var appHost = await DistributedApplicationTestFactory.CreateAsync(appHostPath, testOutput);
        await using var app = await appHost.BuildAsync().WaitAsync(TimeSpan.FromSeconds(15));
        await app.StartAsync().WaitAsync(TimeSpan.FromSeconds(120));
        await app.WaitForResourcesAsync(new[] { KnownResourceStates.Running }).WaitAsync(TimeSpan.FromSeconds(120));
        
        //Act
        var backendResourceName = "AgentHost";
        var expectedMessage = "\"source\": \"HelloAgents/python\"";
        var containsExpectedMessage = false;
        app.EnsureNoErrorsLogged();
        containsExpectedMessage = await app.WaitForExpectedMessageInResourceLogs(backendResourceName, expectedMessage, TimeSpan.FromSeconds(120));
        await app.StopAsync();

        //Assert
        Assert.True(containsExpectedMessage);
    }
    [Fact]
    public async Task AppHostLogsHelloAgentsDotNetSendsAgentHostReceives()
    {
        //Prepare
        var appHostPath = GetAssemblyPath(AppHostAssemblyName);
        var appHost = await DistributedApplicationTestFactory.CreateAsync(appHostPath, testOutput);
        await using var app = await appHost.BuildAsync().WaitAsync(TimeSpan.FromSeconds(15));
        await app.StartAsync().WaitAsync(TimeSpan.FromSeconds(120));
        await app.WaitForResourcesAsync(new[] { KnownResourceStates.Running }).WaitAsync(TimeSpan.FromSeconds(120));
        
        //Act
        var backendResourceName = "AgentHost";
        var expectedMessage = "\"source\": \"HelloAgents/dotnet\"";
        var containsExpectedMessage = false;
        app.EnsureNoErrorsLogged();
        containsExpectedMessage = await app.WaitForExpectedMessageInResourceLogs(backendResourceName, expectedMessage, TimeSpan.FromSeconds(120));
        await app.StopAsync();

        //Assert
        Assert.True(containsExpectedMessage);
    }
    private static string GetAssemblyPath(string assemblyName)
    {
        var parentDir = Directory.GetParent(AppContext.BaseDirectory)?.FullName;
        var grandParentDir = parentDir is not null ? Directory.GetParent(parentDir)?.FullName : null;
        var greatGrandParentDir = grandParentDir is not null ? Directory.GetParent(grandParentDir)?.FullName : null 
            ?? AppContext.BaseDirectory;
        var options = new EnumerationOptions { RecurseSubdirectories = true, MatchCasing = MatchCasing.CaseInsensitive };
        if (greatGrandParentDir is not null)
        {
            var foundFile = Directory.GetFiles(greatGrandParentDir, $"{assemblyName}.dll", options).FirstOrDefault();
            if (foundFile is not null)
            {
                return foundFile;
            }
        }
        throw new FileNotFoundException($"Could not find {assemblyName}.dll in {grandParentDir ?? "unknown"} directory");
    }
}

public class TestEndpoints : IXunitSerializable
{
    // Required for deserialization
    public TestEndpoints() { }

    public TestEndpoints(string appHost, Dictionary<string, List<string>> resourceEndpoints)
    {
        AppHost = appHost;
        ResourceEndpoints = resourceEndpoints;
    }

    public string? AppHost { get; set; }

    public List<ResourceWait>? WaitForResources { get; set; }

    public Dictionary<string, List<string>>? ResourceEndpoints { get; set; }

    public void Deserialize(IXunitSerializationInfo info)
    {
        AppHost = info.GetValue<string>(nameof(AppHost));
        WaitForResources = JsonSerializer.Deserialize<List<ResourceWait>>(info.GetValue<string>(nameof(WaitForResources)));
        ResourceEndpoints = JsonSerializer.Deserialize<Dictionary<string, List<string>>>(info.GetValue<string>(nameof(ResourceEndpoints)));
    }

    public void Serialize(IXunitSerializationInfo info)
    {
        info.AddValue(nameof(AppHost), AppHost);
        info.AddValue(nameof(WaitForResources), JsonSerializer.Serialize(WaitForResources));
        info.AddValue(nameof(ResourceEndpoints), JsonSerializer.Serialize(ResourceEndpoints));
    }

    public override string? ToString() => $"{AppHost} ({ResourceEndpoints?.Count ?? 0} resources)";

    public class ResourceWait(string resourceName, string targetState)
    {
        public string ResourceName { get; } = resourceName;

        public string TargetState { get; } = targetState;

        public void Deconstruct(out string resourceName, out string targetState)
        {
            resourceName = ResourceName;
            targetState = TargetState;
        }
    }
}
