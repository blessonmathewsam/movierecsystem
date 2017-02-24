package com.spark.movierec

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{ concat, lit, sum }
import scala.math.sqrt
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import org.apache.log4j._
import org.apache.spark.broadcast.Broadcast

object MovieRecDF {

  case class Rating(userID: Int, movieID: Int, rating: Double)
  case class Genre(movieId: Int, list: Array[String])

  def ratingMapper(line: String): Rating = {
    val fields = line.split("::")
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
  }

  def genreMapper(line: String): Genre = {
    val fields = line.split("::")
    val genres = fields(2).split('|')
    Genre(fields(0).toInt, genres)
  }

  def main(args: Array[String]) {

    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)

    //Use new SparkSession interface in Spark 2.0
    val spark = SparkSession
      .builder
      .appName("SparkSQL")
      .getOrCreate()

    spark.conf.set("spark.sql.crossJoin.enabled", true)
    val nameDict = spark.sparkContext
      .textFile("s3a://bucket/movies.dat")
      .map(_.split("::"))
      .filter(_.size > 1)
      .map(arr => (arr(0).toInt, arr(1)))
      .collectAsMap()

    val hadoopConf = spark.sparkContext.hadoopConfiguration;
    hadoopConf.set("fs.s3.impl", "org.apache.hadoop.fs.s3native.NativeS3FileSystem")
    hadoopConf.set("fs.s3.awsAccessKeyId", "#################")
    hadoopConf.set("fs.s3.awsSecretAccessKey", "###############################")

    import spark.implicits._
    val lines = spark.sparkContext.textFile("s3a://movierec-blessonm/ratings.dat")

    // Map to Dataset of the form => userID | movieId | rating | time 
    val ratings = lines.map(ratingMapper).toDS()
    val newCols1 = Seq("userID", "movieId1", "rating1", "movieId2", "rating2")

    // Do a self-join => userID | movieId1 | rating1 | movieId2 | rating2 | time 
    val newJoin = ratings.join(ratings, "userId").toDF(newCols1: _*)

    // Filter duplicates
    import org.apache.spark.sql.functions.udf
    val filtered = newJoin.filter($"movieId1" < $"movieId2")

    // Convert dataframe to format => movieId1 | movieId2 | rating1 | rating2 
    val pairs = filtered.select("movieId1", "movieId2", "rating1", "rating2")

    // Generate a key by concatenating movie ids
    val squared = pairs.select(concat($"movieId1", lit("_"), $"movieId2").alias("key"), $"movieId1", $"movieId2", ($"rating1" * $"rating1").alias("ratingXX"), ($"rating2" * $"rating2").alias("ratingYY"), ($"rating1" * $"rating2").alias("ratingXY"))

    // Group by key
    val grouped = squared.groupBy("key", "movieId1", "movieId2").sum("ratingXX", "ratingYY", "ratingXY")

    val cosineSimilarity = udf {
      (ratingXX: Double, ratingYY: Double, ratingXY: Double) =>
        {
          val denom = sqrt(ratingXX) * sqrt(ratingYY)
          val num = ratingXY
          if (denom != 0) (num / denom) else 0.0
        }
    }

    // Calculate consine similarity 
    val finalRatingsDF = grouped.select($"key", $"movieId1", $"movieId2", cosineSimilarity($"sum(ratingXX)", $"sum(ratingYY)", $"sum(ratingXY)").alias("cosinesimilarity"))

    // Read and map movies data
    val movies = spark.sparkContext.textFile("s3a://bucket/movies.dat")
    val genres = movies.map(genreMapper).toDS

    // Self-join
    val genreJoined = genres.join(genres)
    val newCols2 = Seq("movieId1", "genres1", "movieId2", "genres2")
    val newJoined = genreJoined.toDF(newCols2: _*)
    val filteredGenres = newJoined.filter($"movieId1" < $"movieId2")

    import scala.collection.mutable.WrappedArray
    val jaccardCoefficient = udf {
      (Set1: WrappedArray[String], Set2: WrappedArray[String]) =>
        (Set1.toSet.intersect(Set2.toSet)).size.toDouble / (Set1.toSet.union(Set2.toSet)).size.toDouble
    }

    // Compute Jaccard Coefficient
    val withGenre = filteredGenres.withColumn("jaccardcoeff", jaccardCoefficient($"genres1", $"genres2"))
    val finalGenreDF = withGenre.select(concat($"movieId1", lit("_"), $"movieId2").alias("key"), $"jaccardcoeff")

    // Join Ratings and Genres on key
    val joined = finalRatingsDF.join(finalGenreDF, "key")

    // Compute final score by multiplying cosine similarity and jaccard coefficient
    val scored = joined.select($"key", $"movieId1", $"movieId2", ($"jaccardcoeff" * $"cosinesimilarity").alias("score"))

    if (args.length > 0) {
      val movie = args(0).toInt

      // Filter by movieId
      val recommendationsDF = scored.filter($"movieId1" === movie || $"movieId2" === movie)

      // Sort the score of result in descending order
      val resultDF = recommendationsDF.sort($"score".desc).limit(30)

      // Get the id of the movie
      val check = (id1: Int, id2: Int) => if (id1 == movie) nameDict(id2) else nameDict(id1)
      val getName = udf(check)

      // Add movie name to dataframe 
      val results = resultDF.withColumn("movie", getName($"movieId1", $"movieId2"))

      // Show results
      results.show(30)

    }
    spark.stop()
  }
}