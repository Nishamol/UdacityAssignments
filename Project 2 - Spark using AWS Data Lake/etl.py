import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import IntegerType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['conf']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['conf']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    song_data = input_data + "/" + "song_data"
    
    # read song data file
    df = spark.read.option("recursiveFileLookup","true").json(song_data)
    df.createOrReplaceTempView("df")
    # extract columns to create songs table
    songs_table = spark.sql(
                             ''' SELECT
                                        song_id,
                                        artist_id,
                                        title,
                                        duration,
                                        year
                                  FROM
                                        df
                            ''')
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.option("header",True) \
        .partitionBy("year","artist_id") \
        .mode("overwrite") \
        .parquet("output_data/songs/songs.parquet")


    # extract columns to create artists table
    artists_table = spark.sql('''
                         SELECT 
                             artist_id,
                             artist_name,
                             artist_location,
                             artist_latitude,
                             artist_longitude
                         FROM df
                          ''')
    
    # write artists table to parquet files
    artists_table.write.option("header",True)\
                .mode("overwrite") \
                .parquet("output_data/artists/artists.parquet")

def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
        
    log_data = input_data + "/" + "log-data"

    # read log data file
    df = spark.read.option("recursiveFileLookup","true").json(log_data)
    
    # filter by actions for song plays
    df = df.filter("page == 'NextSong'")
    df.createOrReplaceTempView("logdf")

    # extract columns for users table    
    users_table = spark.sql('''
                        SELECT
                            distinct
                            userId AS user_id,
                            firstName AS first_name,
                            lastName AS last_name,
                            gender,
                            level
                        FROM
                            logdf                        
                        ''')
    
    # write users table to parquet files
    users_table.write.option("header",True)\
                .mode("overwrite") \
                .parquet("output_data/users/users.parquet")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S.%f'))
    df = df.withColumn("timestampcol", get_timestamp(logdf.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
    df = df.withColumn("datetime", get_datetime(logdf.ts))
    df = df.dropna(how = "any", subset = ["userId", "sessionId"])  #drop na's in userid and session id.
    df = df.filter(logdf2["userId"] != "")  # condition to check if there is any empty user id
    df = df.filter(col("userId").isNotNull()) # condition to check if null user id present. 
    
    # Code to increment the id for time_table and also songplay_id in songplays table 
    global increment
    increment = 0

    def audit(increment):
        increment = increment + 1
        return increment
    
    # extract columns to create time table
    time_table =  spark.sql('''
                            SELECT 
                                    monotonically_increasing_id()+1  AS audit_id,
                                    date_format(datetime, 'HH:mm:ss') AS start_time,
                                    hour(datetime) AS hour,
                                    day(datetime) AS day,
                                    weekofyear(datetime) AS week,
                                    month(datetime) AS month,
                                    year(datetime) AS year, 
                                    dayofmonth(datetime) AS weekday
                                    
                            FROM logdf
                        ''')
    
    # write time table to parquet files partitioned by year and month
    time_table.write.option("header",True)\
                 .partitionBy("year","month") \
                 .mode("overwrite") \
                 .parquet("output_data/time_table/time_table.parquet")

    # read in song data to use for songplays table
    song_data = input_data + "/" + "song-data"
    song_df = spark.read.option("recursiveFileLookup","true").json(song_data)
    
    
    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = spark.sql('''
                                SELECT
                                   monotonically_increasing_id()+1  AS songplay_id, 
                                   a.artist_id,
                                   u.user_id,
                                   u.level,
                                   s.song_id,
                                   t.start_time,
                                   l.location,
                                   l.sessionId,
                                   l.userAgent,
                                   l.song,
                                   t.year AS year,
                                   t.month AS month

                                FROM
                                    logdf l
                                    join artist_table a ON l.artist like a.artist_name
                                    join users_table u ON l.userId = u.user_id
                                    left join songs_table s ON lower(l.song) like lower(s.title)
                                    join time_table t ON l.userId = t.user_id AND l.sessionId = t.session_id

                              ''')

    songplays_table.show(3)

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.option("header",True)\
                    .partitionBy("year","month") \
                    .mode("overwrite") \
                    .parquet("output_data/songplays/songplays.parquet")


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend"
    output_data = "s3a://udacity-dend/publish"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()