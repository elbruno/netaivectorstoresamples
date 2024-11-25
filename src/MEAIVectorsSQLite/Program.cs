using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Microsoft.Data.Sqlite;
using Microsoft.SemanticKernel.Connectors.Sqlite;

var connection = new SqliteConnection("Data Source=:memory:");
connection.LoadExtension("vector-search-extension-name");
connection.Open();

var vectorStore = new SqliteVectorStore(connection);

// get movie list
var movies = vectorStore.GetCollection<string, MovieSQLite<string>>("movies");
await movies.CreateCollectionIfNotExistsAsync();
var movieData = MovieFactory<string>.GetMovieSQLiteList();

// get embeddings generator and generate embeddings for movies
IEmbeddingGenerator<string, Embedding<float>> generator =
    new OllamaEmbeddingGenerator(new Uri("http://localhost:11434/"), "all-minilm");
foreach (var movie in movieData)
{
    movie.DescriptionEmbedding = await generator.GenerateEmbeddingVectorAsync(movie.Description);
    await movies.UpsertAsync(movie);
}

// perform the search
var query = "A family friendly movie that includes ogres and dragons";
var queryEmbedding = await generator.GenerateEmbeddingVectorAsync(query);

var searchOptions = new VectorSearchOptions()
{
    Top = 2,
    VectorPropertyName = "DescriptionEmbedding"
};

var results = await movies.VectorizedSearchAsync(queryEmbedding, searchOptions);
await foreach (var result in results.Results)
{
    Console.WriteLine($"Title: {result.Record.Title}");
    Console.WriteLine($"Description: {result.Record.Description}");
    Console.WriteLine($"Score: {result.Score}");
    Console.WriteLine();
}