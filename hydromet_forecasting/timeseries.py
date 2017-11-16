import csv
import datetime
from math import floor
from os.path import basename

import pandas
from numpy import nan


class FixedIndexTimeseries(object):
    """This class implements a wrapper for 5-day, decadal and monthly timeseries .

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
                model: The frequency that series is expected to have, either: p (5-day), d (decadal), m (monthly)
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
        elif mode == "p":
            self.maxindex = 72
            self.period = 5
        elif mode == "m":
            self.maxindex = 12
            self.period = 30
        else:
            raise ValueError("The given mode was not recognized. Check the docstring of the class.")

        if self._check_timeseries(series):
            self.timeseries = series
        else:
            raise self.ModeError(
                "The given series can not be recognized as a timeseries with frequency mode %s" % self.mode)

        if label == None:
            self.label = self.timeseries.name
        else:
            self.label = label

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

        month = int((annual_index - 1) / (self.maxindex / 12)) + 1
        day_start = ((annual_index - 1) % (self.maxindex / 12)) * self.period + 1
        return datetime.date(year, month, day_start)

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
        return (date.month - 1) * (self.maxindex / 12) + ((date.day - 1) / self.period) + 1

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

    def norm(self, annualindex=None):
        # NEEDS TO BE SHIFTED TO EVALUATOR
        norm = []
        years = range(min(self.timeseries.index).year, max(self.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, self.maxindex + 1)

        for index in indexrange:
            dates = map(self.firstday_of_period, years, len(years) * [index])
            norm.append(self.timeseries[dates].mean())
        if type(annualindex) == int:
            norm = norm[0]

        return norm

    def max(self, annualindex=None):
        # NEEDS TO BE SHIFTED TO EVALUATOR
        out = []
        years = range(min(self.timeseries.index).year, max(self.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, self.maxindex + 1)

        for index in indexrange:
            dates = map(self.firstday_of_period, years, len(years) * [index])
            out.append(self.timeseries[dates].max())
        if type(annualindex) == int:
            out = out[0]

        return out

    def min(self, annualindex=None):
        # NEEDS TO BE SHIFTED TO EVALUATOR
        out = []
        years = range(min(self.timeseries.index).year, max(self.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, self.maxindex + 1)

        for index in indexrange:
            dates = map(self.firstday_of_period, years, len(years) * [index])
            out.append(self.timeseries[dates].min())
        if type(annualindex) == int:
            out = out[0]

        return out

    def stdev_s(self, annualindex=None):
        # NEEDS TO BE SHIFTED TO EVALUATOR
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
            out = out[0]

        return out

    def data_by_index(self, annualindex):
        out = []
        years = range(min(self.timeseries.index).year, max(self.timeseries.index).year + 1)
        indexrange = ([annualindex] if type(annualindex) == int else annualindex)
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
        elif mode == "p":
            self.maxindex = 72
            self.period = 5
        elif mode == "m":
            self.maxindex = 12
            self.period = 30
        else:
            raise ValueError("The given mode was not recognized. Check the docstring of the class.")
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
