#TODO: transformers needs to be saved!
#TODO: type transformation!!!
def piNone(featurename,dataframe):
    df = dataframe
    return df

from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.ml.linalg import DenseVector
def convertSparseVectortoDenseVector(v):
        v = DenseVector(v)
        new_array = list([float(x) for x in v])
        return new_array

toDenseVectorUdf = F.udf(convertSparseVectortoDenseVector, T.ArrayType(T.FloatType()))

def piStrDoubleCast(featurename,dataframe):
    from pyspark.sql.types import DoubleType
    result = dataframe.withColumn(featurename+"Pre", data_df[featurename].cast(DoubleType()))
    return result

def piStrDateCast(featurename,dataframe):
    from pyspark.sql.functions import from_unixtime
    from pyspark.sql.functions import unix_timestamp
    from pyspark.sql.functions import col
    from pyspark.sql.functions import lit
    df = dataframe
    df = df.withColumn(featurename+"Pre", 
        #df.select(from_unixtime(unix_timestamp(col(featurename), 'yyyy-MM-dd')))
        from_unixtime(unix_timestamp(featurename, 'yyyy-MM-dd'))
    )

    return df

def piStrOneHotEncoding(featurename,dataframe):
    from pyspark.ml.feature import OneHotEncoder
    from pyspark.ml.feature import StringIndexer
    #from pyspark.ml.feature import VectorIndexer
    indexed = dataframe
    indexer = StringIndexer(inputCol=featurename, outputCol=featurename+"HE")
    indexed = indexer.fit(indexed).transform(indexed)
    encoder = OneHotEncoder(inputCols=[featurename+"HE"],
                                outputCols=[featurename+"OHE"])
    indexed = encoder.fit(indexed).transform(indexed)

    indexed = indexed.drop(featurename).drop(featurename+"HE").withColumnRenamed(featurename+"OHE",featurename)
    #indexer = VectorIndexer(inputCol=featurename+"OHE", outputCol=featurename+"tHE", maxCategories=10)
    #indexerModel = indexer.fit(indexed)
    #indexed = indexerModel.transform(indexed)
    
    return indexed

def piNumStandardScaler(featurename,dataframe):
    from pyspark.sql.functions import mean,stddev,max,min
    print("meanstart")
    mean_age, sttdev_age,max_age,min_age = dataframe.select(mean(featurename), stddev(featurename),max(featurename),min(featurename)).first()
    print(mean_age)
    print(sttdev_age)
    print(max_age)
    print(min_age)
    print("meanend")
    #return dataframe.withColumn(featurename, (dataframe[featurename] - mean_age) / sttdev_age) #standardization
    print((dataframe[featurename] - min_age) / (max_age-min_age))
    a = dataframe.withColumn(featurename, (dataframe[featurename] - min_age) / (max_age-min_age)) #min max scale
    a.show()
    return a
    

def piStrCountVectorizer(featurename,dataframe):
    from pyspark.ml.feature import CountVectorizer
    from pyspark.ml.feature import Tokenizer, RegexTokenizer
    from pyspark.sql.functions import col, udf
    from pyspark.sql.types import IntegerType
    df = dataframe
    
    tokenizer = Tokenizer(inputCol=featurename, outputCol=featurename+"Words")
    df = tokenizer.transform(df)
    # fit a CountVectorizerModel from the corpus.
    cv = CountVectorizer(inputCol=featurename+"Words", outputCol=featurename+"Pre", vocabSize=999999, minDF=2.0)

    model = cv.fit(df)

    result = model.transform(df)

    #result = result.withColumn(featurename+"Pre", toDenseVectorUdf(featurename+"Pre2"))

    return result

def indexPipelines():
    pipelines = {}
    pipelines[0] = (0,"None",piNone,None)
    pipelines[1] = (1,"String Indexer & One Hot Encoding",piStrOneHotEncoding,"string")
    pipelines[2] = (2,"Count Vectorizer",piStrCountVectorizer,"string")
    pipelines[3] = (3,"Standard Scaler",piNumStandardScaler,"double")
    pipelines[4] = (4,"DateFormat",piStrDateCast,"string")
    return pipelines

def applytransition(id,featurename,dataframe):
    ip = indexPipelines()
    mytuple = None
    if id in ip.keys():
        mytuple = ip[id]
    else:
        mytuple = ip[0]
    if mytuple is None:
        raise    
    return mytuple[2](featurename,dataframe)

import  pyspark.sql.functions as psf
import pyspark.sql
def customfeaturepipeline(pipeline,featurename,dataframe):
    
    # list of functions is here 
    #https://sparkbyexamples.com/spark/usage-of-spark-sql-string-functions/
    df = dataframe
    pipedfeaturename = featurename+'Piped'
    
    
    try:
        pipeline_qualified = pipeline.replace("self","'"+featurename+"'")

        #for example = psf.regexp_replace(psf.regexp_replace('Erster','\.',''),',','.').cast('double')
        a = eval(pipeline_qualified)
        if isinstance(a,pyspark.sql.Column):
            dataframe = df.withColumn(pipedfeaturename,a) 
        else: 
            raise Exception('WrongSparkSQLException')
    except:
        dataframe = df.withColumn(pipedfeaturename,df[featurename])
    
    
    return dataframe

#<option>String Indexing & OneHotEncoding</option>
#<option>Tokenizer</option>
#<option>StopWordRemover</option>
#<!-- start for double-->
#<option>MinMaxScaler</option>
#<option>QuantileDiscretizer</option>

