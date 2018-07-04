from numpy import nan, isnan, arange, corrcoef, mean
from matplotlib import pyplot as plt
import pandas
from hydromet_forecasting.timeseries import FixedIndexTimeseries
from string import Template
import base64
import tempfile
from os import path

class Evaluator(object):
    """Evaluator class for a predicted timeseries that is given as an FixedIndexTimeseries instance and has annual seasonality.

    This class enables to evaluate the performance of an forecast by means of comparing the observed and the forecasted
    timeseries. Those series need to be instances of the FixedIndexTimeseries class. The Evaluator class implements
    various methods to analyse the series statistically. The method write_html writes a report to a specified path.

    Attributes:
        forecast: the forecasted timeseries
        y_clean: the observed timeseries, reduced to the index of the forecasted timeseries
    """

    def __init__(self, y, forecast):
        """Initialising the Evaluator Instance

            Args:
                y: A FixedIndexTimeseries instance of the observed data
                forecast:  A FixedIndexTimeseries instance of the forecasted data of the same mode as y.

            Returns:
                The Evaluator instance

            Raises:
                ValueError: When the two timeseries are not of the same mode.
                InsufficientData: If the length of the forecasted timeseries is not sufficient for evalaution. At least two
                                    complete years of data must be provided in order to compute standard deviations etc.
            """

        if not y.mode == forecast.mode:
            raise ValueError("The target timeseries is not of the same mode as the forecasted timeseries.")
        self._y = y
        self.forecast = forecast

        self.y_clean=FixedIndexTimeseries(self._y.timeseries[self.forecast.timeseries.index], mode=self._y.mode)

        datagroups = [len(self.forecast.data_by_index(i)) for i in range(1,self.forecast.maxindex+1)]
        if min(datagroups)<2:
            raise self.InsufficientData("The length of the forecasted timeseries is not sufficient.")


    def computeP(self):
        """ Returns the P value (Percentage of forecasts with error/stdev > 0.674)

            Args:
                None

            Returns:
                A list of values for each period of the year of the forecasted timeseries. NaN is retuned of the value could not be determined (e.g. not enough data)

            Raises:
                None
            """
        P = []
        allowed_error = map(lambda x: x * 0.674, self._y.stdev_s())
        years = range(min(self.y_clean.timeseries.index).year, max(self.y_clean.timeseries.index).year + 1)
        for index in range(0, self.y_clean.maxindex):
            dates = map(self.y_clean.firstday_of_period, years, len(years) * [index + 1])
            try:
                error = abs(self.forecast.timeseries.reindex(dates)[dates] - self.y_clean.timeseries.reindex(dates)[dates])
                error = error.dropna()
                good = sum(error <= allowed_error[index])
                P.append(float(good) / len(error))
            except:
                P.append(nan)
        return P

    def computeRelError(self):
        """ Returns the relative error value of the forecast (error / stde

                    Args:
                        None

                    Returns:
                        A list of values for each period of the year of the forecasted timeseries. NaN is retuned if the value could not be determined (e.g. not enough data)

                    Raises:
                        None
                    """
        relerror = []
        stdev = self._y.stdev_s()
        years = range(min(self.y_clean.timeseries.index).year, max(self.y_clean.timeseries.index).year + 1)
        for index in range(0, self.y_clean.maxindex):
            dates = map(self.y_clean.firstday_of_period, years, len(years) * [index + 1])
            try:
                error = abs(self.forecast.timeseries.reindex(dates)[dates] - self.y_clean.timeseries.reindex(dates)[dates])
                error = error.dropna()
                relerror.append(error.values/stdev[index])
            except:
                relerror.append(nan)
        return relerror

    def computeRelError(self):
        """ Returns the relative error value of the forecast (error / stdev.sample)

                    Args:
                        None

                    Returns:
                        A list of values for each period of the year of the forecasted timeseries. NaN is retuned if the value could not be determined (e.g. not enough data)

                    Raises:
                        None
                    """
        relerror = []
        stdev = self._y.stdev_s()
        years = range(min(self.y_clean.timeseries.index).year, max(self.y_clean.timeseries.index).year + 1)
        for index in range(0, self.y_clean.maxindex):
            dates = map(self.y_clean.firstday_of_period, years, len(years) * [index + 1])
            try:
                error = abs(self.forecast.timeseries.reindex(dates)[dates] - self.y_clean.timeseries.reindex(dates)[dates])
                error = error.dropna()
                relerror.append(error.values/stdev[index])
            except:
                relerror.append(nan)
        return relerror

    def trainingdata_count(self):
        """ Returns the number of training data for each period of the year

                    Args:
                        None

                    Returns:
                        A list of values for each period of the year of the forecasted timeseries.
                    Raises:
                        None
                    """
        count = list()
        for index in range(1, self.y_clean.maxindex+1):
            count.append(len(self.forecast.data_by_index(index)))
        return count

    def prepare_figure(self, width=12, height=3):

        fig, ax = plt.subplots(1, 1)
        fig.set_figwidth(width)
        fig.set_figheight(height)

        nr_bars=self._y.maxindex
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
        norm = self._y.max()
        min = self._y.min()
        max = self._y.max()
        stdev = self._y.stdev_s()
        fig, ax = self.prepare_figure()
        ax.plot(norm,label="norm")
        ax.plot(min, label="min")
        ax.plot(max, label="max")
        plt.ylabel(self._y.label)
        return fig

    def plot_P(self):
        P = self.computeP()
        fig, ax = self.prepare_figure()
        ax.bar(range(0, len(P)), P, width=0.7, color="black")
        plt.ylabel("P%")
        ax.set_ylim([0, 1])
        return fig

    def plot_RelError(self):
        relerror = self.computeRelError()
        fig, ax = self.prepare_figure()
        ax.boxplot(relerror)
        ax.plot([0, ax.get_xlim()[1]], [0.674, 0.674], color='red', linestyle='dashed')
        plt.ylabel("Error/STDEV")
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
        fig.set_figwidth(8)
        fig.set_figheight(8)
        ax.scatter(self.y_clean.timeseries,self.forecast.timeseries, marker=".", color='black')
        ax.set_ylabel("predicted")
        ax.set_xlabel("observed")
        maxval = max(ax.get_xlim()[1],ax.get_ylim()[1])
        ax.set_ylim([0,maxval])
        ax.set_xlim([0,maxval])
        ax.plot([0, maxval], [0, maxval],color='green', linestyle='dashed')
        r_corr = round(corrcoef(self.y_clean.timeseries,self.forecast.timeseries)[0,1],3)
        ax.text(0.5 * maxval, 0.9 * maxval, ("R = %s" % (r_corr)))
        return fig

    def table_summary(self):
        data =dict({
            'Number of training data': self.trainingdata_count(),
            'Minimum':self._y.min(),
            'Norm':self._y.norm(),
            'Maximum':self._y.max(),
            '+/- d': self._y.stdev_s(),
            'P%': self.computeP()
        })
        df=pandas.DataFrame(data)
        return df.to_html()

    def write_html(self, filename):
        """ writes an evaluation report to the specified filepath as an html

            Args:
                filename: path to the html file to be created

            Returns:
                None

            Raises:
                None
            """
        templatefilepath = path.join(path.dirname(__file__),'template')
        with open(templatefilepath, 'r') as htmltemplate:
            page=Template(htmltemplate.read())

        encoded1=self.encode_figure(self.plot_trainingdata())
        encoded2 = self.encode_figure(self.plot_y_stats())
        encoded3 = self.encode_figure(self.plot_P())
        encoded4 = self.encode_figure(self.plot_ts_comparison())
        encoded5 = self.encode_figure(self.plot_RelError())

        table = self.table_summary()

        htmlpage = open(filename, 'w')
        htmlpage.write(page.safe_substitute(TABLE=table,IMAGE1=encoded1,IMAGE2=encoded2,IMAGE3=encoded3,IMAGE4=encoded4,IMAGE5=encoded5))
        htmlpage.close()
        return filename

    def encode_figure(self, fig):

        with tempfile.TemporaryFile(suffix=".png") as tmpfile:
            fig.savefig(tmpfile, format="png")
            tmpfile.seek(0)
            encoded = base64.b64encode(tmpfile.read())
            tmpfile.close()
        return encoded

    class InsufficientData(Exception):
        pass


class SeasonalEvaluator(object):
    def __init__(self, featurenames,selectedfeatures,modelEvaluators, score):
        self.featurenames = featurenames
        self.selectedfeatures = selectedfeatures
        self.modelEvaluators = modelEvaluators
        self.score = score

    def prepare_figure(self, width=12, height=3):
        fig, ax = plt.subplots(1, 1)
        fig.set_figwidth(width)
        fig.set_figheight(height)
        return fig, ax

    def table_summary(self):
        index_best = self.score.index(min(self.score))
        index_worst = self.score.index(max(self.score))
        data =dict({
            'Number of training data': int(mean([CV.trainingdata_count()[0] for CV in self.modelEvaluators])),
            'Minimum':self.modelEvaluators[0]._y.min(),
            'Norm':self.modelEvaluators[0]._y.norm(),
            'Maximum':self.modelEvaluators[0]._y.max(),
            '+/- d': self.modelEvaluators[0]._y.stdev_s(),
            'STDEV/ERROR': mean([mean(CV.computeRelError()) for CV in self.modelEvaluators]),
            'P%': mean([CV.computeP() for CV in self.modelEvaluators])
        })
        df=pandas.DataFrame(data)
        return df.to_html()

    def plot_timeseries(self):
        fig, ax = self.prepare_figure()
        [ax.plot(CV.forecast.timeseries, color='red', label="individual forecasts", alpha=.2) for CV in self.modelEvaluators]
        df_concat = pandas.concat(([CV.forecast.timeseries for CV in self.modelEvaluators]))
        by_row_index = df_concat.groupby(df_concat.index)
        df_means = by_row_index.mean()
        ax.plot(df_means, color='black', label='mean forecast')
        ax.plot(self.modelEvaluators[0].y_clean.timeseries, color='green', label='observed')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-3:], labels[-3:])
        return fig

    def write_html(self, filename):
        """ writes an evaluation report to the specified filepath as an html

            Args:
                filename: path to the html file to be created

            Returns:
                None

            Raises:
                None
            """
        templatefilepath = path.join(path.dirname(__file__),'template')
        with open(templatefilepath, 'r') as htmltemplate:
            page=Template(htmltemplate.read())

        encoded1=self.encode_figure(self.plot_timeseries())

        table = self.table_summary()

        htmlpage = open(filename, 'w')
        htmlpage.write(page.safe_substitute(TABLE=table, IMAGE1=encoded1))
        htmlpage.close()
        return filename

    def encode_figure(self, fig):

        with tempfile.TemporaryFile(suffix=".png") as tmpfile:
            fig.savefig(tmpfile, format="png")
            tmpfile.seek(0)
            encoded = base64.b64encode(tmpfile.read())
            tmpfile.close()
        return encoded