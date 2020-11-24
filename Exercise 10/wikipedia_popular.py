import sys
from pyspark.sql import SparkSession, functions, types
import re

spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+


schema = types.StructType([
    types.StructField('language', types.StringType(), False),
    types.StructField('page_name', types.StringType(), False),
    types.StructField('views', types.LongType(), False),
    types.StructField('bytes', types.LongType(), False),
])


def getdate(path):
    result = re.search("([0-9]{8}\-[0-9]{2})", path)
    
    return result.group(1)


def main(in_dir, out_dir):
    
    # Read the csv file
    wiki_page = spark.read.csv(in_dir, schema=schema, sep=' ').withColumn('filename', functions.input_file_name())

    # We need to find the most-viewed page each hour
    data = wiki_page.filter(wiki_page['language']=='en')
    data = data.filter(data['page_name'] != 'Main_Page')
    data = data.filter(data.page_name.startswith('Special:')==False)

    path_to_hour = functions.udf(lambda path: getdate(path), returnType=types.StringTypes())

    data = data.withColumn('date', path_to_hour(data.filename))
    data = data.drop('language', 'bytes', 'filename')

    groups = data.groupBy('date')

    most_viewed = groups.agg(functions.max(data['views']).alias('views'))
    most_viewed.cache()

    cond = ['views', 'date']
    data_join = most_viewed.join(data, cond)

    output = data_join.sort('date', 'page_name')

    output.show()


if __name__=='__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    main(in_dir, out_dir)