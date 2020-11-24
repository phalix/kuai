def getXandYFromPyArrow(table):
    ### This needs to be done since for multi dim arrays, the numpy shape is lost...
    ### pyarrow to pylist crashed
    ### numpy to_list does not work recursively
    def recResolution(rec):
        if isinstance(rec,np.ndarray):
            rec_2 = rec.tolist()
            return list(map(lambda x: recResolution(x) if isinstance(x,np.ndarray) else x,rec_2))
        else:
            return rec
    columns = table.column_names
    #columns = table.columns
    featurevalues = list(filter(lambda x:x.startswith("feature"),columns))
    x = []
    for feature in featurevalues:
        x_3 = table[feature]
        x_3 = np.asarray(recResolution(x_3.to_numpy()))
        x.append(x_3)
    if len(x) == 1:
        x = x[0]
    targetvalues = list(filter(lambda x:x.startswith("target"),columns))
    y = []
    for target in targetvalues:
        y_3 = table[target]
        y_3 = np.asarray(recResolution(y_3.to_numpy()))
        y.append(y_3)
    if len(y) == 1:
        y = y[0]
    return {"x":x,"y":y}