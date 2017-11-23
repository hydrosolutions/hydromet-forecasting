from numpy import nan, isnan, full, array, arange
from matplotlib import pyplot as plt
import pandas
from hydromet_forecasting.timeseries import FixedIndexTimeseries
from string import Template
import base64
import tempfile
import pdfkit

class Evaluator(object):
    """UNDER DEVELOPMENT: This class will contain all information and methods for assessing model performance

    It will have a method write_pdf(filename), that generates the assessment report and writes it to "filename".
    When no filename is given, the pdf is stored in a temporary folder.
    Returns: the pathname where the pdf is stored.
    """

    def __init__(self, y, forecast):

        if not y.mode == forecast.mode:
            raise ValueError("The target timeseries is not of the same frequency as the forecasted timeseries.")
        self.y = y
        self.forecast = forecast

        self.y_clean=FixedIndexTimeseries(self.y.timeseries[self.forecast.timeseries.index],mode=self.y.mode)


        # TODO Assert that enough data is contained in timeseries objects

    @staticmethod
    def norm(FixedIndexTimeseries, annualindex=None):
        # NEEDS TO BE SHIFTED TO EVALUATOR
        ts_object=FixedIndexTimeseries
        norm = []
        years = range(min(ts_object.timeseries.index).year, max(ts_object.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, ts_object.maxindex + 1)

        for index in indexrange:
            dates = map(ts_object.firstday_of_period, years, len(years) * [index])
            norm.append(ts_object.timeseries[dates].mean())
        if type(annualindex) == int:
            norm = norm[0]

        return norm

    @staticmethod
    def max(FixedIndexTimeseries, annualindex=None):
        # NEEDS TO BE SHIFTED TO EVALUATOR
        out = []
        years = range(min(FixedIndexTimeseries.timeseries.index).year, max(FixedIndexTimeseries.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, FixedIndexTimeseries.maxindex + 1)

        for index in indexrange:
            dates = map(FixedIndexTimeseries.firstday_of_period, years, len(years) * [index])
            out.append(FixedIndexTimeseries.timeseries[dates].max())
        if type(annualindex) == int:
            out = out[0]

        return out

    @staticmethod
    def min(FixedIndexTimeseries, annualindex=None):
        # NEEDS TO BE SHIFTED TO EVALUATOR
        out = []
        years = range(min(FixedIndexTimeseries.timeseries.index).year, max(FixedIndexTimeseries.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, FixedIndexTimeseries.maxindex + 1)

        for index in indexrange:
            dates = map(FixedIndexTimeseries.firstday_of_period, years, len(years) * [index])
            out.append(FixedIndexTimeseries.timeseries[dates].min())
        if type(annualindex) == int:
            out = out[0]

        return out

    @staticmethod
    def stdev_s(FixedIndexTimeseries, annualindex=None):
        # NEEDS TO BE SHIFTED TO EVALUATOR
        out = []
        years = range(min(FixedIndexTimeseries.timeseries.index).year, max(FixedIndexTimeseries.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, FixedIndexTimeseries.maxindex + 1)

        for index in indexrange:
            dates = map(FixedIndexTimeseries.firstday_of_period, years, len(years) * [index])
            try:
                out.append(FixedIndexTimeseries.timeseries[dates].std())
            except:
                out.append(nan)
        if type(annualindex) == int:
            out = out[0]

        return out


    def computeP(self):
        P = []
        allowed_error = map(lambda x: x * 0.674, self.stdev_s(self.y))
        years = range(min(self.y_clean.timeseries.index).year, max(self.y_clean.timeseries.index).year + 1)
        for index in range(0, self.y_clean.maxindex):
            dates = map(self.y_clean.firstday_of_period, years, len(years) * [index + 1])
            try:
                error = abs(self.forecast.timeseries.reindex(dates)[dates] - self.y_clean.timeseries.reindex(dates)[dates])
                error.dropna()
                good = sum(error <= allowed_error[index])
                P.append(float(good) / len(error.dropna()))
            except:
                P.append(nan)
        return P

    def trainingdata_count(self):
        year_min = self.y_clean.timeseries.index[0].year
        year_max = self.y_clean.timeseries.index[-1].year
        count = [0]*36
        for date in self.y_clean.timeseries.index:
            count[self.y_clean.convert_to_annual_index(date)-1]+=1

        return count



    def prepare_figure(self, width=12, height=3):
        fig, ax = plt.subplots(1, 1)
        fig.set_figwidth(width)
        fig.set_figheight(height)

        nr_bars=self.y.maxindex
        periods_in_month=nr_bars/12.0
        monthly_labels_pos=[p*nr_bars/12.0+(0.0416667*nr_bars-0.5) for p in range(0,12)]
        ax.set_xticks(monthly_labels_pos)
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

        monthly_dividers=[p*nr_bars/12.0+(0.0416667*nr_bars-0.5)-1.5 for p in range(0,13)]
        ax.set_xticks(monthly_dividers, minor=True)

        ax.grid(True, which="minor", axis="x", color="black", linestyle='--')
        ax.tick_params(axis="x", which="major", length=0)
        return fig, ax

    def plot_y_stats(self):
        norm=self.norm(self.y)
        min = self.min(self.y)
        max = self.max(self.y)
        fig, ax = self.prepare_figure()
        ax.plot(norm,label="norm")
        ax.plot(min, label="min")
        ax.plot(max, label="max")
        plt.ylabel(self.y.label)
        return fig

    def plot_P(self):
        P = self.computeP()
        fig, ax = self.prepare_figure()
        ax.bar(range(0, len(P)), P, width=0.7, color="black")
        plt.ylabel("P%")
        ax.set_ylim([0, 1])
        return fig

    def plot_trainingdata(self):
        count = self.trainingdata_count()
        fig, ax = self. prepare_figure()
        ax.bar(range(0, len(count)), count, width=0.7)
        ax.bar(range(0, len(count)), count, width=0.7)
        plt.ylabel("Number of training data")

        return fig

    def plot_ts_comparison(self):
        fig, ax = plt.subplots(1, 1)
        fig.set_figwidth(12)
        fig.set_figheight(3)
        ax.plot(self.y_clean.timeseries, label="observed")
        ax.plot(self.forecast.timeseries, label="predicted")
        ax.set_ylabel(self.y_clean.label)
        return fig

    def table_summary(self):
        data =dict({
            'Number of training data': self.trainingdata_count(),
            'Minimum':self.min(self.y),
            'Norm':self.norm(self.y),
            'Maximum':self.max(self.y),
            '+/- d': self.stdev_s(self.y),
            'P%': self.computeP()
        })
        df=pandas.DataFrame(data)
        return df.to_html()

    def write_html(self, filename="/home/jules/Desktop/test.html"):
        with open('/home/jules/Desktop/Hydromet/hydromet_forecasting/hydromet_forecasting/template', 'r') as htmltemplate:
            page=Template(htmltemplate.read())

        encoded1=self.encode_figure(self.plot_trainingdata())
        encoded2 = self.encode_figure(self.plot_y_stats())
        encoded3 = self.encode_figure(self.plot_P())
        encoded4 = self.encode_figure(self.plot_ts_comparison())

        table = self.table_summary()

        htmlpage = open(filename, 'w')
        htmlpage.write(page.safe_substitute(TABLE=table,IMAGE1=encoded1,IMAGE2=encoded2,IMAGE3=encoded3,IMAGE4=encoded4))
        htmlpage.close()
        return filename

    def encode_figure(self, fig, filename=None):

        with tempfile.TemporaryFile(suffix=".png") as tmpfile:
            fig.savefig(tmpfile, format="png")  # File position is at the end of the file.
            tmpfile.seek(0)  # Rewind the file. (0: the beginning of the file)
            encoded = base64.b64encode(tmpfile.read())
            tmpfile.close()
        return encoded
