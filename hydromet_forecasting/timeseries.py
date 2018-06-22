import csv
import datetime
from math import floor
from os.path import basename

import pandas
from numpy import nan

from stldecompose import decompose


class FixedIndexTimeseries(object):
    """This class implements a wrapper for 5-day, decadal and monthly timeseries.

    FixedIndex means, that each year has the same number of periods and that every period takes the same position in
    every year, e.g. monthes or semi-monthes etc. It does not work for timeseries with periods, that strictly consist
    of the same  number of days and as such, might overlap New Year. This class is based on pandas.Series objects.

    The timeseries are indexed by the first day of a period, e.g. 2007/5/11 for the 2nd decade in May 2007.
    The annual index is defined as the position of the period within the year, e.g. 5 for the 2nd decade in February
    Timeseries can be loaded from a csv file with the subclass FixedIndexTimeseriesCSV

    Attributes:
        timeseries: a pandas.Series object with data indexed by the first day of a period as datetime.date object.
        label: an optional, custom label for the object.
        mode: The frequency mode of the timeseries. Either p (5-day), d (decadal), m (monthly)
        maxindex: the maximum value that annualindex can have for the mode
    """

    def __init__(self, series, mode, label=None):
        """Initialises an instance of the FixedIndexTimeseries Class

            Args:
                series: A pandas.Series object, where the index is datetime.date objects.
                mode: The frequency that series is expected to have, either: p (5-day), d (decadal), m (monthly)
                label: An optional label for the timeseries. Default is None: uses the label that is found in the series object.

            Returns:
                An instance of FixedIndexTimeseries

            Raises:
                ValueError: When the argument given for mode is not recognized.
                ModeError: when the mode given as argument and the property of series do not fit. Does only recognize if
                        series if of higher frequency than indicated by mode.
            """
        self.mode = mode
        if mode == 'd':
            self.maxindex = 36
            self.period = 10
            self.periodname="decade"
        elif mode == "p":
            self.maxindex = 72
            self.period = 5
            self.periodname="pentade"
        elif mode == "m":
            self.maxindex = 12
            self.period = 30
            self.periodname="month"
        elif mode == "dl":
            self.maxindex = 365
            self.period = 1
            self.periodname="daily"
        else:
            try:
                res = mode.split("-")
                self.begin = datetime.date(1,int(res[0]),1)
                self.end = (datetime.date(1,int(res[1]),1) + datetime.timedelta(32)).replace(day=1) + datetime.timedelta(-1) #Ugly solution to get last day of the month
                #self.yearswitch = False if int(res[1])>int(res[0]) else True #does the season definition overlap new year?
                self.maxindex = 1
                self.period = (self.end - self.begin).days
                self.periodname = "season"
            except:
                raise ValueError("The given mode was not recognized. Check the docstring of the class for details.")

        if self._check_timeseries(series):
            self.timeseries = series.sort_index()
        else:
            raise self.ModeError(
                "The given series can not be recognized as a timeseries with frequency mode %s" % self.mode)

        if label == None:
            self.label = self.timeseries.name
        else:
            self.label = label

        self.mode_order = ['dl','p','d','m']

    class ModeError(Exception):
        pass

    def _check_timeseries(self, series):
        for i, item in series.iteritems():
            date = self.firstday_of_period(i.year, self.convert_to_annual_index(i))
            if not date == i:
                return False
        return True

    def firstday_of_period(self, year, annual_index):
        """Returns the first day of a period given by the year and the annual index of the period

            Decadal: first day of period (2007,3) --> datetime.date(2007,1,21)

            Args:
                year: The year
                annual_index: The index of the period within a year. 0 < annual_index < maxindex (e.g. 5-day: 72)

            Returns:
                datetime.date(y,m,d) of the first day of the period described by the year and annnual index.

            Raises:
                ValueError: When the annual index is invalid or outside the valid range defined by the mode
            """

        if not 0 < annual_index < self.maxindex + 1 or not type(annual_index) == int:
            raise ValueError("Annual index is not valid: 0 < index < %s for mode=%s" % (self.maxindex + 1, self.mode))

        if self.maxindex == 1:
            return datetime.date(year, self.begin.month, self.begin.day)
        elif self.maxindex == 365:
            return datetime.date(year, 1, 1) + datetime.timedelta(annual_index - 1)
        else:
            month = int((annual_index - 1) / (float(self.maxindex) / 12)) + 1
            day_start = int(((annual_index - 1) % (float(self.maxindex) / 12)) * self.period + 1)
            return datetime.date(year, month, day_start)

    def lastday_of_period(self, year, annual_index):
        """Returns the last day of a period given by the year and the annual index of the period

            Decadal: last day of period (2007,3) --> datetime.date(2007,1,31)

            Args:
                year: The year
                annual_index: The index of the period within a year. 0 < annual_index < maxindex (e.g. 5-day: 72)

            Returns:
                datetime.date(y,m,d) of the last day of the period described by the year and annnual index.

            Raises:
                ValueError: When the annual index is invalid or outside the valid range defined by the mode
            """

        if not 0 < annual_index < self.maxindex + 1 or not type(annual_index) == int:
            raise ValueError("Annual index is not valid: 0 < index < %s for mode=%s" % (self.maxindex + 1, self.mode))

        if self.maxindex == 1:
            return datetime.date(year, self.end.month, self.end.day)
        else:
            annual_index = annual_index+1
            if annual_index>self.maxindex:
                annual_index = 1
                year = year + 1
            return self.firstday_of_period(year,annual_index)-datetime.timedelta(1)

    @staticmethod
    def doy(date):
        return date.timetuple().tm_yday

    def convert_to_annual_index(self, date):
        """Returns the annual_index of a datetime.date object

            Decadal: datetime.date(2007,1,21) --> first day of period (2007,3)
                     datetime.date(2007,1,30) --> first day of period (2007,3)
                     datetime.date(2007,2,1)  --> first day of period (2007,4)
            Is the reverse function of firstday_of_period(year,annual_index)

            Args:
                date: A datetime.date object

            Returns:
                int: the annual index of the period that the datetime.date is member of.

            Raises:
                None
            """
        if self.maxindex == 1:
            if self.doy(date) > self.doy(self.end):
                return 2
            else:
                return 1
        elif self.maxindex == 365:
            return self.doy(date)
        else:
            return int((date.month - 1) * (float(self.maxindex) / 12)) + ((min(date.day,30) - 1) / self.period) + 1

    def shift_date_by_period(self, date, shift):
        """Shifts a datetime.date object by the given number of periods.

            E.g. decadal: Shifting datetime.date(2007,1,25)
                          by -3 gives datetime.date(2006,12,21)
            Remark: The input date is fist converter to the first day of the period it is member of.

            Args:
                date: A datetime.date object
                shift: An integer corresponding to the periods that the date should be shifted.
                        Negative value: back in time. Positive value: forward in time

            Returns:
                datetime.date: the shifted date

            Raises:
                None
            """
        newindex = self.convert_to_annual_index(date) + shift
        # Correcting for shifts between years:
        if newindex < 1:
            factor = int(floor((newindex - 1) / self.maxindex))
            return self.firstday_of_period(date.year + 1 * factor, newindex - self.maxindex * factor)
        elif newindex > self.maxindex:
            factor = int(floor((newindex - 1) / self.maxindex))
            return self.firstday_of_period(date.year + int(1 * factor), newindex - self.maxindex * factor)
        else:
            return self.firstday_of_period(date.year, newindex)

    def data_by_index(self, annualindex):

        indexrange = ([annualindex] if type(annualindex) == int else annualindex)

        if not all([(0 < i < self.maxindex+1) for i in indexrange]):
            raise ValueError("The provided annualindex is outside the range %s < annualindex < %s" %(0,self.maxindex+1))
        out = []
        years = range(min(self.timeseries.index).year, max(self.timeseries.index).year + 1)

        for index in indexrange:
            dates = map(self.firstday_of_period, years, len(years) * [index])
            try:
                data = self.timeseries[dates]
                data = data.dropna()
                out.append(data)
            except:
                out.append([])
        if type(annualindex) == int:
            out = out[0]

        return out

    def norm(self, annualindex=None):
        """Given a FixedIndexTimeseries, returns the average (norm) value for each period of the year or the specified period

            Args:
                FixedIndexTimeseriesObj: a FixedIndexTimeseries instance
                annualindex: None (default), or the index or list of indexes of the period(s) for which the norm should be computed.
                            Otherwise the norms for all periods are computed.

            Returns:
                A value or list of values describing the norm in the same order as argument annualindex

            Raises:
                None
            """

        norm = []
        #years = range(min(FixedIndexTimeseriesObj.timeseries.index).year, max(FixedIndexTimeseriesObj.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, self.maxindex + 1)

        for index in indexrange:
            # TODO dates = map(FixedIndexTimeseriesObj.firstday_of_period, years, len(years) * [index])
            #norm.append(FixedIndexTimeseriesObj.timeseries[dates].mean())
            norm.append(self.data_by_index(index).mean())
        if type(annualindex) == int:
            return norm[0]
        else:
            return norm

    def max(self, annualindex=None):
        """Given a FixedIndexTimeseries, returns the max value for each period of the year or the specified period

            Args:
                FixedIndexTimeseriesObj: a FixedIndexTimeseries instance
                annualindex: None (default), or the index or list of indexes of the period(s) for which the max value should be computed.
                            Otherwise the max values for all periods are computed.

            Returns:
                A value or list of values describing the maximum in the same order as argument annualindex

            Raises:
                None
            """

        out = []
        #years = range(min(FixedIndexTimeseries.timeseries.index).year, max(FixedIndexTimeseries.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, self.maxindex + 1)

        for index in indexrange:
            # TODO dates = map(FixedIndexTimeseries.firstday_of_period, years, len(years) * [index])
            #out.append(FixedIndexTimeseries.timeseries[dates].max())
            out.append(self.data_by_index(index).max())
        if type(annualindex) == int:
            return out[0]
        else:
            return out

    def min(self, annualindex=None):
        """Given a FixedIndexTimeseries, returns the min value for each period of the year or the specified period

            Args:
                FixedIndexTimeseriesObj: a FixedIndexTimeseries instance
                annualindex: None (default), or the index or list of indexes of the period(s) for which the min value should be computed.
                            Otherwise the min values for all periods are computed.

            Returns:
                A value or list of values describing the minimum in the same order as argument annualindex

            Raises:
                None
            """

        out = []
        #years = range(min(FixedIndexTimeseries.timeseries.index).year, max(FixedIndexTimeseries.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, self.maxindex + 1)

        for index in indexrange:
            # TODO dates = map(FixedIndexTimeseries.firstday_of_period, years, len(years) * [index])
            #out.append(FixedIndexTimeseries.timeseries[dates].min())
            out.append(self.data_by_index(index).min())
        if type(annualindex) == int:
            return out[0]
        else:
            return out

    def stdev_s(self, annualindex=None):
        """Given a FixedIndexTimeseries, returns the stdev.sample value for each period of the year or the specified period

            Args:
                FixedIndexTimeseriesObj: a FixedIndexTimeseries instance
                annualindex: None (default), or the index or list of indexes of the period(s) for which the stdev.sample value should be computed.
                            Otherwise the stdev.sample values for all periods are computed.

            Returns:
                A value or list of values describing the stdev.sample in the same order as argument annualindex

            Raises:
                None
            """
        out = []
        years = range(min(self.timeseries.index).year, max(self.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, self.maxindex + 1)

        for index in indexrange:
            dates = map(self.firstday_of_period, years, len(years) * [index])
            try:
                out.append(self.timeseries[dates].std())
            except:
                out.append(nan)
        if type(annualindex) == int:
            return out[0]
        else:
            return out

    def trend(self):
        dec = decompose(self.timeseries.values, period=self.maxindex)
        return FixedIndexTimeseries(pandas.Series(dec.trend, index=self.timeseries.index), mode=self.mode)

    def seasonal(self):
        dec = decompose(self.timeseries.values, period=self.maxindex)
        return FixedIndexTimeseries(pandas.Series(dec.seasonal, index=self.timeseries.index), mode=self.mode)

    def residual(self):
        dec = decompose(self.timeseries.values, period=self.maxindex)
        return FixedIndexTimeseries(pandas.Series(dec.resid, index=self.timeseries.index), mode=self.mode)

    def detrend(self):
        dec = decompose(self.timeseries.values, period=self.maxindex)
        return FixedIndexTimeseries(pandas.Series(dec.resid, index=self.timeseries.index)+pandas.Series(dec.seasonal, index=self.timeseries.index), mode=self.mode)

    def derivative(self):
        diff = self.timeseries.diff()
        delta_days = [(x-y).days for x, y in zip(self.timeseries.index, self.timeseries.index[1:])]
        derivative = -diff.drop(diff.index[0])/delta_days
        return FixedIndexTimeseries(derivative, mode=self.mode)

    def downsample(self, mode):
        if len(self.mode) > 2:
            raise ValueError('The timeseries can not be downsampled')
        if len(mode) > 1:
            self.mode_order.append(mode)

        if self.mode_order.index(mode) <= self.mode_order.index(self.mode):
            raise ValueError('The target mode is of same or higher frequency than the source mode. Only downsampling is allowed.')
        else:
            dailyindex = pandas.date_range(self.timeseries.index.values[0], self.timeseries.index.values[-1], freq='D')
            dailytimeseries = self.timeseries.reindex(dailyindex).interpolate('zero')
            dummyInstance = FixedIndexTimeseries(pandas.Series(), mode=mode)
            beginyear = self.timeseries.index.values[0].year
            endyear = self.timeseries.index.values[-1].year
            newindex = [dummyInstance.firstday_of_period(y, i) for y in range(beginyear, endyear+1) for i in range(1, dummyInstance.maxindex+1)]
            values = [nan] * len(newindex)
            for i,date in enumerate(newindex):
                lastday = dummyInstance.lastday_of_period(date.year,dummyInstance.convert_to_annual_index(date))
                try:
                    values[i] = dailytimeseries.reindex(pandas.date_range(date,lastday,freq='D')).mean()
                except:
                    pass
        return FixedIndexTimeseries(pandas.Series(values,newindex),mode=mode)

    def multiply(self, FixedIndexTimeseries_obj):
        if self.mode is not FixedIndexTimeseries_obj.mode:
            raise self.ModeError("Both timeseries must be of the same mode")

        res = self.timeseries.multiply(FixedIndexTimeseries_obj.timeseries)
        return FixedIndexTimeseries(res, mode = self.mode)




class FixedIndexTimeseriesCSV(FixedIndexTimeseries):
    """Is a subclass of FixedIndexTimeseries. Can be initialised with a path of a csv file.

    Description of required csv-file format: rows contain the data of 1 year.
    The first column contains the year of each row. The length of the rows corresponds
    to number of periods of the chosen mode in each year, additional columns will be ignored
    e.g. monthly:
    1995,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12
    1996,...
    Strings are loaded as NaN

    """

    def __init__(self, csv_filepath, mode, label=None):
        self.mode = mode
        if mode == 'd':
            self.maxindex = 36
            self.period = 10
            self.periodname = "decade"
        elif mode == "p":
            self.maxindex = 72
            self.period = 5
            self.periodname = "pentade"
        elif mode == "m":
            self.maxindex = 12
            self.period = 30
            self.periodname = "month"
        elif mode == "dl":
            self.maxindex = 365
            self.period = 1
            self.periodname="daily"
        else:
            try:
                res = mode.split("-")
                self.begin = datetime.date(1, int(res[0]), 1)
                self.end = datetime.date(1, int(res[1]) + 1, 1) - datetime.timedelta(1)
                self.maxindex = 1
                self.period = (self.end - self.begin).days
                self.periodname = "season"
            except:
                raise ValueError("The given mode was not recognized. Check the docstring of the class for details.")


        series = self.load_csv(csv_filepath)
        FixedIndexTimeseries.__init__(self, series, mode, label)

    def load_csv(self, filepath):
        """loads array-like timeseries data from .csv into indexed pandas series

            Description of required csv-file format: rows contain the data of 1 year.
            The first column contains the year of each row. The length of the rows corresponds
            to number of periods of the chosen mode in each year, additional columns will be ignored
            e.g. monthly:
            1995,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12
            1996,...
            Strings are loaded as NaN

            Args:
                filepath: the path to a csv file

            Returns:
                pandas.Series objects

            Raises:
                ValueError: The yearnumber in the first column of the csv could not be recognized.
            """

        reader = csv.reader(open(filepath, 'r'))
        intlist = []
        datelist = []
        for row in reader:
            for i in range(1, self.maxindex + 1):
                try:
                    intlist.append(float(row[i]))
                except:
                    intlist.append(nan)
                try:
                    date = self.firstday_of_period(year=int(row[0]), annual_index=i)
                except ValueError:
                    raise ValueError("CSV format error: The first column must contain years")
                datelist.append(date)

        return pandas.Series(data=intlist, index=datelist, name=basename(filepath))
