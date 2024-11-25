public class MovieFactory<T>
{
    private static T _currentKey;

    static MovieFactory()
    {
        _currentKey = default(T);
    }

    public static List<Movie<T>> GetMovieList()
    {
        var movieData = new List<Movie<T>>()
        {
            new Movie<T>
            {
                Key = KeyGeneratorReturnNext(),
                Title = "Lion King",
                Description = "The Lion King is a classic Disney animated film that tells the story of a young lion named Simba who embarks on a journey to reclaim his throne as the king of the Pride Lands after the tragic death of his father."
            },
            new Movie<T>
            {
                Key = KeyGeneratorReturnNext(),
                Title = "Inception",
                Description = "Inception is a science fiction film directed by Christopher Nolan that follows a group of thieves who enter the dreams of their targets to steal information."
            },
            new Movie<T>
            {
                Key = KeyGeneratorReturnNext(),
                Title = "The Matrix",
                Description = "The Matrix is a science fiction film directed by the Wachowskis that follows a computer hacker named Neo who discovers that the world he lives in is a simulated reality created by machines."
            },
            new Movie<T>
            {
                Key = KeyGeneratorReturnNext(),
                Title = "Shrek",
                Description = "Shrek is an animated film that tells the story of an ogre named Shrek who embarks on a quest to rescue Princess Fiona from a dragon and bring her back to the kingdom of Duloc."
            }
        };
        return movieData;
    }

    public static List<MovieVector<T>> GetMovieVectorList()
    {
        var movieData = GetMovieList();
        var movieVectorData = new List<MovieVector<T>>();
        foreach (var movie in movieData)
        {
            movieVectorData.Add(new MovieVector<T>
            {
                Key = movie.Key,
                Title = movie.Title,
                Description = movie.Description
            });
        }
        return movieVectorData;
    }

    public static List<MovieSQLite<T>> GetMovieSQLiteList()
    {
        var movieData = GetMovieList();
        var movieVectorData = new List<MovieSQLite<T>>();
        foreach (var movie in movieData)
        {
            movieVectorData.Add(new MovieSQLite<T>
            {
                Key = movie.Key,
                Title = movie.Title,
                Description = movie.Description
            });
        }
        return movieVectorData;
    }

    public static T KeyGeneratorReturnNext()
    {
        if (typeof(T) == typeof(int))
        {
            _currentKey = (T)Convert.ChangeType(Convert.ToInt32(_currentKey) + 1, typeof(T));
        }
        else if (typeof(T) == typeof(long))
        {
            _currentKey = (T)Convert.ChangeType(Convert.ToInt64(_currentKey) + 1, typeof(T));
        }
        else if (typeof(T) == typeof(ulong))
        {
            _currentKey = (T)Convert.ChangeType(Convert.ToUInt64(_currentKey) + 1, typeof(T));
        }
        else if (typeof(T) == typeof(string))
        {
            _currentKey = (T)Convert.ChangeType(Guid.NewGuid().ToString(), typeof(T));
        }
        else
        {
            throw new InvalidOperationException("Unsupported key type");
        }

        return _currentKey;
    }
}
