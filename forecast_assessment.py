from sklearn import linear_model,datasets
import csv
from os import remove, path
from numpy import nan
import pandas
import datetime

class forecaster(object):



def firstday_of_decade(year, decade_of_year):
    assert 0 < decade_of_year < 37, 'decade_of_year is out of range 0 < x < 37'
    month = int((decade_of_year - 1) / 3) + 1
    day_start = ((decade_of_year - 1) % 3) * 10 + 1
    return datetime.date(year, month, day_start)

def load_decadal_csv(filepath):
    assert path.isfile(filepath), filepath + ' is not a file!'
    reader = csv.reader(open(filepath, 'r'))
    intlist = []
    datelist = []
    for row in reader:
        for idx, stringvalue in enumerate(row[1:]):
            try:
                intlist.append(float(stringvalue))
            except:
                intlist.append(nan)
            date = firstday_of_decade(year=int(row[0]),decade_of_year=idx+1)
            datelist.append(date)

    return pandas.Series(data=intlist,index=datelist)


ts=load_decadal_csv("/home/jules/Desktop/Hydromet/decadal_data.csv")
target=ts
feature=ts.shift(1)
df=pandas.DataFrame({'target':target,'feature':feature})
df=df.dropna(how='any')
reg = linear_model.LinearRegression()
reg.fit(df.feature.values.reshape(-1,1),df.target.values.reshape(-1,1))
print(reg.coef_)

