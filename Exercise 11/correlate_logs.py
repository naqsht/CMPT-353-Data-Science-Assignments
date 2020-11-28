import sys
from pyspark.sql import SparkSession, functions, types, Row
import re
import math

spark = SparkSession.builder.appName('correlate logs').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        # TODO
        host = m.group(1)
        nbytes = m.group(2)
        return Row(hostname=host, num_bytes=nbytes)
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    # TODO: return an RDD of Row() objects
    rows = log_lines.map(line_to_row)
    rows = rows.filter(not_none)
    
    return rows


def main(in_directory):

    # Get the data out of the files into a DataFrame where you have the hostname and 
    # number of bytes for each request. Do this using an RDD operation: see hints below.
    logs = spark.createDataFrame(create_row_rdd(in_directory))

    # Group by hostname; get the number of requests and sum of bytes transferred, to form a data point.
    logs_hostname = logs.groupby('hostname').count().alias('num_requests')
    logs_nbytes = logs.groupby('hostname').agg(functions.sum('num_bytes').alias('total_bytes'))
    logs_join = logs_nbytes.join(logs_hostname, 'hostname')

    # Produce six values. Add these to get the six sums.
    logs = logs_join.select(logs_join['count'].alias('xi'), logs_join['total_bytes'].alias('yi'), (logs_join['count']*logs_join['total_bytes']).alias('xiyi'), (logs_join['count']*logs_join['count']).alias('xi^2'), (logs_join['total_bytes']*logs_join['total_bytes']).alias('yi^2'))

    n = logs.count()
    sum_xi = logs.agg(functions.sum('xi')).first()[0]
    sum_yi = logs.agg(functions.sum('yi')).first()[0]
    sum_xi2 = logs.agg(functions.sum('xi^2')).first()[0]
    sum_yi2 = logs.agg(functions.sum('yi^2')).first()[0]
    sum_xiyi = logs.agg(functions.sum('xiyi')).first()[0]

    # Calculate the final value of r.
    r = ((n*sum_xiyi)-(sum_xi*sum_yi))/((math.sqrt((n*sum_xi2)-(sum_xi**2)))*(math.sqrt((n*sum_yi2)-(sum_xi**2))))

    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)
