import os, sys
from pathlib import Path
from typing import Optional
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, MinHashLSH
from pyspark.sql import SparkSession





class TextSimilarityAnalyzer:
    """
    A class for text similarity analysis using MinHashLSH in PySpark.
    Handles data loading, preprocessing, model building and similarity detection.
    """

    def __init__(self, spark_session=None, hash_buckets=1048576, lsh_tables=64,
                 jaccard_threshold=0.2, app_name="SimilarityReview"):
        """
        Initialize the TextSimilarityAnalyzer.
        
        Args:
            spark_session: Existing SparkSession or None to create new one
            hash_buckets: Number of hash buckets for HashingTF
            lsh_tables: Number of hash tables for MinHashLSH
            jaccard_threshold: Minimum Jaccard similarity threshold
            app_name: Spark application name
        """
        self.spark = spark_session or SparkSession.builder \
            .appName(app_name) \
            .master("local[*]") \
            .getOrCreate()
        
        self.hash_buckets = hash_buckets
        self.lsh_tables = lsh_tables
        self.jaccard_threshold = jaccard_threshold
        self.model = None
        self.pipeline = None

    def build_pipeline(self):
        """Build the text processing and LSH pipeline."""
        tokenizer = RegexTokenizer(inputCol="review_text", outputCol="tokens",
                                   pattern="\\W+", gaps=True, toLowercase=True)
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
        tf = HashingTF(inputCol="filtered_tokens", outputCol="features",
                      numFeatures=self.hash_buckets, binary=True)
        lsh = MinHashLSH(inputCol="features", outputCol="hashes",
                        numHashTables=self.lsh_tables)
        
        self.pipeline = Pipeline(stages=[tokenizer, remover, tf, lsh])
        return self.pipeline
    
    def load_reviews(self, csv_path="./dataset/Books_rating.csv", sample_n=None):
        """
        Load and clean the Books_rating.csv dataset.
        Keeps only (review_id, review_text).
        Optionally subsamples rows.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        # load CSV
        df = (self.spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(str(path)))

        # rename columns if present
        if "review/text" in df.columns:
            df = df.withColumnRenamed("review/text", "review_text")
        
        # keep only valid reviews
        df = (df
            .filter(F.col("review_text").isNotNull() & (F.length("review_text") > 0))
            .select(F.col("Id").cast("string").alias("review_id"), "review_text"))

        # optional subsample
        if sample_n is not None:
            df = df.sample(withReplacement=False, seed = 42, fraction=sample_n / df.count())

        return df
    
    def build_lsh_model(self, df):
        """
        Build and fit a minimal MinHashLSH pipeline on (review_id, review_text).
        Returns (model, df_features) where df_features has:
          review_id, features, set_tokens
        """
        if self.pipeline is None:
            self.build_pipeline()

        self.model = self.pipeline.fit(df)
        
        out = self.model.transform(df)
        # keep only rows with at least 1 token and a nonzero hashed vector
        out = (out
               .withColumn("set_tokens", F.array_distinct(F.col("filtered_tokens")))
               .filter(F.size("set_tokens") >= 5)
               .drop("tokens", "filtered_tokens", "hashes")
            )

        return self.model, out.select("review_id", "review_text", "features", "set_tokens")

    def find_similar_reviews(self, df_feat, topk=None):
        approx_dist = 1.0 - self.jaccard_threshold
        if self.model is None:
            raise ValueError("Model not built. Call build_lsh_model first.")

        a = df_feat.select(
            F.col("review_id").alias("i"),
            F.col("review_text").alias("text_i"),
            F.col("features"),
            F.col("set_tokens").alias("tok_i"),
        )
        b = df_feat.select(
            F.col("review_id").alias("j"),
            F.col("review_text").alias("text_j"),
            F.col("features"),
            F.col("set_tokens").alias("tok_j"),
        )

        lsh_model = self.model.stages[-1]

        cand_raw = lsh_model.approxSimilarityJoin(
            a.select("i", "features"),
            b.select("j", "features"),
            approx_dist,
            distCol="approx_dist",
        )

        # Flatten the struct columns from approxSimilarityJoin
        cand = cand_raw.select(
            F.col("datasetA.i").alias("i"),
            F.col("datasetB.j").alias("j"),
            F.col("approx_dist"),
        )
        pairs = (
            cand
            # de-duplicate symmetric pairs
            .filter(F.col("i") < F.col("j"))
            # join texts + tokens back
            .join(a.select("i", "text_i", "tok_i"), on="i", how="inner")
            .join(b.select("j", "text_j", "tok_j"), on="j", how="inner")
            # exact Jaccard on token sets
            .withColumn("inter_sz", F.size(F.array_intersect("tok_i", "tok_j")))
            .withColumn("union_sz", F.size(F.array_union("tok_i", "tok_j")))
            .withColumn(
                "jaccard",
                F.when(F.col("union_sz") > 0, F.col("inter_sz") / F.col("union_sz"))
                 .otherwise(F.lit(0.0)),
            )
            .filter(F.col("jaccard") >= F.lit(self.jaccard_threshold))
            .select("i", "j", "jaccard", "text_i", "text_j")
            .orderBy(F.col("jaccard").desc())
        )

        if topk is not None:
            pairs = pairs.limit(int(topk))

        return pairs

    
    def lsh_candidates(self, df_feat, sim_threshold=None):
        """
        Generate candidate pairs with LSH (approx Jaccard distance).
        df_feat must have: review_id, features
        Returns (review_id_1, review_id_2, approx_dist)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_lsh_model first.")
            
        if sim_threshold is None:
            sim_threshold = self.jaccard_threshold
            
        approx_dist = float(1.0 - sim_threshold)  # convert similarity -> distance

        a = df_feat.select(F.col("review_id").alias("id_a"),
                           F.col("features"))
        b = df_feat.select(F.col("review_id").alias("id_b"),
                           F.col("features"))

        cand = (self.model.stages[-1].approxSimilarityJoin(a, b, approx_dist, distCol="approx_dist")
                .select(F.col("datasetA.id_a").alias("review_id_1"),
                        F.col("datasetB.id_b").alias("review_id_2"),
                        "approx_dist")
                .filter(F.col("review_id_1") < F.col("review_id_2"))
                .distinct())

        return cand

    def analyze_similarity_pipeline(self, csv_path="./dataset/Books_rating.csv",
                                  sample_n=None, topk= None):
        """
        Complete pipeline: load data, build model, find similarities.
        Returns similar review pairs.
        """
        # Load and preprocess data
        df = self.load_reviews(csv_path, sample_n)
        print(f"Loaded {df.count()} reviews")
        
        # Build LSH model
        model, df_feat = self.build_lsh_model(df)
        print(f"Built LSH model with {df_feat.count()} processed reviews")
        
        # Find similar reviews
        similar_pairs = self.find_similar_reviews(df_feat, topk=topk)
        print(f"Found {similar_pairs.count()} similar pairs")
        
        return similar_pairs
