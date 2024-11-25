using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Azure;
using Azure.Search.Documents.Indexes;
using Microsoft.SemanticKernel.Connectors.AzureAISearch;
using Microsoft.Extensions.Configuration;
using Azure.Identity;
using Azure.Core;

// get the search index client using Azure Default Credentials or Azure Key Credential with the service secret
var client = GetSearchIndexClient();
var vectorStore = new AzureAISearchVectorStore(searchIndexClient: client);

// get movie list
var movies = vectorStore.GetCollection<string, MovieVector<string>>("movies");
await movies.CreateCollectionIfNotExistsAsync();
var movieData = MovieFactory<string>.GetMovieVectorList();

// get embeddings generator and generate embeddings for movies
IEmbeddingGenerator<string, Embedding<float>> generator =
    new OllamaEmbeddingGenerator(new Uri("http://localhost:11434/"), "all-minilm");
foreach (var movie in movieData)
{
    movie.Vector = await generator.GenerateEmbeddingVectorAsync(movie.Description);
    await movies.UpsertAsync(movie);
}

// perform the search
var query = "A family friendly movie that includes ogres and dragons";
var queryEmbedding = await generator.GenerateEmbeddingVectorAsync(query);

var searchOptions = new VectorSearchOptions()
{
    Top = 2,
    VectorPropertyName = "Vector"
};

var results = await movies.VectorizedSearchAsync(queryEmbedding, searchOptions);
await foreach (var result in results.Results)
{
    Console.WriteLine($"Title: {result.Record.Title}");
    Console.WriteLine($"Description: {result.Record.Description}");
    Console.WriteLine($"Score: {result.Score}");
    Console.WriteLine();
}

SearchIndexClient GetSearchIndexClient()
{
    var config = new ConfigurationBuilder().AddUserSecrets<Program>().Build();
    var azureAISearchUri = config["AZURE_AISEARCH_URI"];

    var credential = new DefaultAzureCredential();
    var client = new SearchIndexClient(new Uri(azureAISearchUri), credential);
    var secret = config["AZURE_AISEARCH_SECRET"];

    if (!string.IsNullOrEmpty(secret))
    {
        client = new SearchIndexClient(new Uri(azureAISearchUri), new AzureKeyCredential(secret));
    }
    
    return client;
}
