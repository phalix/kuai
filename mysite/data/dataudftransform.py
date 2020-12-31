from mysite.models import UDF

class dataudftransform:
    ### TODO: Create parentclass!
    #############
    ### Call me by test = mysite.data.dataudftransform.dataudftransform(dataframe,[],spark)
    #############  test.execute(project).show()
    #########

    dataframe = None
    udfinstructions = None
    sparksession = None
    tablename = 'transform_temp_table'
    
    def __init__(self,dataframe,udfinstructions,sparksession):
        self.dataframe = dataframe
        self.udfinstructions =  udfinstructions
        self.sparksession = sparksession


    def evaluateUDFOnDataFrame(self,project):
        udfs = self.collectUDFSOnProject(project)
        udfs['type'] = col("type") # do not loose type, train test cv
        udflist = list(udfs.values())
        udflist = udflist + self.dataframe.columns
        return self.dataframe.select(udflist)
        
    def collectUDFSOnProject(self,project):
        udfs = UDF.objects.filter(project=project.pk).all()
        from django.core import serializers
        data = serializers.serialize("json", udfs)
        import json
        udfdata = json.loads(data)

        functionarray = {}
        for udf in udfdata:
            import re
            pattern = re.compile("[^A-Za-z]")
            aliasname = pattern.sub("",udf['fields']['input'])
            functionname = aliasname+'_'+str(udf['pk'])
            udfpara = udf['fields']
            udfpara['functionname'] = functionname
            a = self.generateUDFonUDF(udfpara)
            functionarray[a[0]] = a[1]
        
        return functionarray

    def generateUDFonUDF(self,udfdefinition):
        from pyspark.sql.functions import udf
        outputtype = udfdefinition['outputtype']
        left = ""
        right = ""
        for col in outputtype.split(','):
            left = left + 'pyspark.sql.types.'+col+'('
            right = ')'+right
        outputtypeeval = eval(left+right)
        
        udfcode = udfdefinition['udfexecutiontext']
        functionname = udfdefinition['functionname']
        exec('def '+functionname+'(output): \n\t'+udfcode.replace("\n","\n\t"))
        a_udf = eval('udf('+functionname+', outputtypeeval)')
        udfColInput = udfdefinition['input']
        if len(udfColInput.split(","))>1:
            udfColInput = eval("pyspark.sql.functions.array"+str(tuple(udfColInput.split(","))))

        b_udf = a_udf(udfColInput).alias(functionname)
        return [functionname,b_udf]

    def execute(self,project):
        from pyspark.sql import SQLContext
        from pyspark import StorageLevel

        
        sqlContext = SQLContext(self.sparksession)
        if 'transform_temp_table' in sqlContext.tableNames():
            df2 = sqlContext.sql("select * from transform_temp_table")
        else:
            
            try:
                df2 = self.evaluateUDFOnDataFrame(project,self.dataframe)
                df2.registerTempTable("transform_temp_table")
                sqlContext.sql("CACHE TABLE transform_temp_table OPTIONS ('storageLevel' 'DISK_ONLY')")

            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                df2 = self.dataframe
            
        return df2

    def show(self):
        print("hi")
        #### return actual execution text here!