# -*- encoding: UTF-8 -*-
from numpy import nan, isnan, arange, corrcoef, mean
from matplotlib import pyplot as plt
import pandas
from hydromet_forecasting.timeseries import FixedIndexTimeseries
from string import Template
import base64
import tempfile
from os import path, environ
import gettext
from babel.dates import format_date, get_month_names

from utils import to_str
from plot_utils import PlotUtils

class Evaluator(object):
    """Evaluator class for a predicted timeseries that is given as an FixedIndexTimeseries instance and has annual seasonality.

    This class enables to evaluate the performance of an forecast by means of comparing the observed and the forecasted
    timeseries. Those series need to be instances of the FixedIndexTimeseries class. The Evaluator class implements
    various methods to analyse the series statistically. The method write_html writes a report to a specified path.

    Attributes:
        forecast: the forecasted timeseries
        y_adj: the observed timeseries, reduced to the index of the forecasted timeseries
    """

    _rel_error = None
    _p = None

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
        self.y = y
        self.forecast = forecast

        self.y_adj=FixedIndexTimeseries(self.y.timeseries[self.forecast.timeseries.index], mode=self.y.mode)

        datagroups = [len(self.forecast.data_by_index(i)) for i in range(1,self.forecast.maxindex+1)]
        if min(datagroups)<2:
            raise self.__InsufficientData("The length of the forecasted timeseries is not sufficient.")


    def computeP(self, annualindex=None):
        """ Returns the P value (Percentage of forecasts with error/stdev > 0.674)

            Args:
                annualindex (int): default None

            Returns:
                A list of values for each period of the year of the forecasted timeseries if annualindex is None, else one value.
                    NaN is retuned of the value could not be determined (e.g. not enough data)

            Raises:
                None
            """
        P = []
        indexrange = range(1, self.y_adj.maxindex+1) if annualindex is None else [annualindex]

        for index in indexrange:
            allowed_error =  0.674 * self.y.stdev_s(index)
            try:
                error = abs(self.forecast.data_by_index(index) - self.y_adj.data_by_index(index))
                error = error.dropna()
                good = sum(error <= allowed_error)
                P.append(float(good) / len(error))
            except:
                P.append(nan)
        return P

    def computeRelError(self, annualindex=None):
        """ Returns the relative error value of the forecast (error / stdev.sample)

                    Args:
                        None

                    Returns:
                        A list of values for each period of the year of the forecasted timeseries if annualindex is None, else one value.
                    NaN is retuned of the value could not be determined (e.g. not enough data)

                    Raises:
                        None
                    """
        relerror = []
        indexrange = range(1, self.y_adj.maxindex + 1) if annualindex is None else [annualindex]

        for index in indexrange:
            stdev = self.y.stdev_s(index)
            try:
                error = abs(self.forecast.data_by_index(index) - self.y_adj.data_by_index(index))
                error = error.dropna()
                relerror.append(error.values/stdev)
            except:
                relerror.append(nan)
        return relerror

    def trainingdata_count(self, annualindex = None):
        """ Returns the number of training data for each period of the year

                    Args:
                        annualindex (int): the annualindex for which trainingdata shall be counted. default=None

                    Returns:
                        A list of values for each period of the year of the forecasted timeseries if annualindex is None, else a single integer
                    Raises:
                        None
                    """

        if annualindex is None:
            count = list()
            indexrange = range(1, self.y_adj.maxindex + 1)
            for index in indexrange:
                count.append(len(self.forecast.data_by_index(index)))
        else:
            count = len(self.forecast.data_by_index(annualindex))
        return count

    def plot_y_stats(self):
        norm = self.y.norm()
        stdev = self.y.stdev_s()
        upper = [norm[i]+stdev[i] for i in range(0,len(stdev))]
        lower = [norm[i]-stdev[i] for i in range(0,len(stdev))]
        fig, ax = PlotUtils.prepare_figure(len(stdev))
        [ax.plot(self.y.data_by_year(year).values, label='individual years', color='blue', alpha=.2) for year in
         range(self.y.timeseries.index[0].year, self.y.timeseries.index[-1].year + 1)]
        ax.plot(upper, color='black')
        ax.plot(lower, color='black',label="+/- STDEV")
        ax.plot(norm,label="NORM", color='red')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-3:], labels[-3:])
        plt.ylabel(self.y.label)
        return fig

    def plot_trainingdata(self):
        count = self.trainingdata_count()
        fig, ax = PlotUtils.prepare_figure(len(count))
        ax.bar(range(0, len(count)), count, width=0.7)
        ax.bar(range(0, len(count)), count, width=0.7)
        plt.ylabel(_("Number of training data"))

        return fig

    def summary_table(self):
        data = dict({
            _('Number of training data'): self.trainingdata_count(),
            _('Minimum'): self.y.min(),
            _('Norm'): self.y.norm(),
            _('Maximum'): self.y.max(),
            _('+/- d'): self.y.stdev_s(),
            _('P%'): self.p,
            _('ScaledError'): self.rel_error,
        })
        df = pandas.DataFrame(data)
        return df.to_html()

    def p_plot_table(self):
        data = dict({
            _('P%'): self.p,
        })
        df = pandas.DataFrame(data)
        return df.to_html()

    def rel_error_table(self):
        data = dict({
            _('ScaledError'): self.rel_error,
        })
        df = pandas.DataFrame(data)
        return df.to_html()

    @staticmethod
    def load_template_file(filename='template.html'):
        template_path = path.join(path.dirname(__file__), filename)
        with open(template_path, 'r') as template_path:
            page = Template(template_path.read())

        return page

    def write_html(
            self,
            username,
            organization,
            site_code,
            site_name,
            filename=None,
            htmlpage=None,
            language='en'
    ):

        locales = environ.get('LOCALES_PATH', 'locales')
        t = gettext.translation('messages', locales, languages=[language])
        t.install()

        if self.y.mode == 'p':
            frequency = 'fiveday'
        elif self.y.mode == 'd':
            frequency = 'decade'
        elif self.y.mode == 'm':
            frequency = 'monthly'

        page = self.load_template_file()
        scatter_plot = PlotUtils.plot_ts_comparison(
            self.y_adj.timeseries,
            self.forecast.timeseries,
            frequency,
            language=language,
        )

        scaled_error_title = _('Scaled Error [RMSE/STDEV]')
        scaled_error_plot = PlotUtils.plot_rel_error(self.rel_error, frequency, title=scaled_error_title)
        scaled_error_table = self.rel_error_table()

        p_plot_title = _('P% Plot')
        p_plot_plot = PlotUtils.plot_p(self.p, frequency, title=p_plot_title)
        p_plot_table = self.p_plot_table()

        quality_assessment_table = self.summary_table()

        report_data = {
            'SITE_INFO': _('Station: {code} - {name}').format(code=site_code, name=site_name),
            'USERNAME': username,
            'ORGANIZATION': organization,
            'TITLE': _('Forecast Model Training Report'),
            'REPORT_DATE': format_date(format='long', locale=language),
            'PLOTS_HEADER': _('{frequency} Forecast Model Quality Assessment').format(
                frequency=frequency.capitalize()),
            'SCATTER_PLOT_LABEL': _('Scatter Plot: Observed versus Predicted values'),
            'SCALED_ERROR_LABEL': scaled_error_title,
            'P_PLOT_LABEL': p_plot_title,
            'QUALITY_ASSESSMENT_LABEL': _('Quality Assessment'),
            'SCATTER_PLOT_IMAGE': scatter_plot,
            'SCALED_ERROR_PLOT_IMAGE': scaled_error_plot,
            'SCALED_ERROR_TABLE': scaled_error_table,
            'P_PLOT_IMAGE': p_plot_plot,
            'P_PLOT_TABLE': p_plot_table,
            'QUALITY_ASSESSMENT_TABLE': quality_assessment_table,
        }

        self.encode_utf8(report_data)

        if filename:
            htmlpage = open(filename, 'w')
            htmlpage.write(page.safe_substitute(**report_data))
            htmlpage.close()
            return filename
        elif htmlpage:
            htmlpage.write(page.safe_substitute(**report_data))
            return htmlpage

    @classmethod
    def encode_utf8(cls, template_vars):
        for key, value in template_vars.iteritems():
            template_vars[key] = to_str(value)

    @property
    def rel_error(self):
        if self._rel_error is None:
            self._rel_error = self.computeRelError()

        return self._rel_error

    @property
    def p(self):
        if self._p is None:
            self._p = self.computeP()

        return self._p

    class __InsufficientData(Exception):
        pass


class SeasonalEvaluator(object):
    def __init__(self, featurenames,selectedfeatures,modelEvaluators):
        self.featurenames = featurenames
        self.selectedfeatures = selectedfeatures
        self.modelEvaluators = modelEvaluators
        self.score = [mean(CV.computeRelError()) for CV in self.modelEvaluators]

    def __prepare_figure(self, width=12, height=3):
        fig, ax = plt.subplots(1, 1)
        fig.set_figwidth(width)
        fig.set_figheight(height)
        fig.subplots_adjust(left=0.06, bottom=0.16, right=0.94, top=0.92)
        return fig, ax

    def __table_summary(self):
        data = dict({
            _('Minimum'): mean(self.modelEvaluators[0].y.min()),
            _('Norm'): self.modelEvaluators[0].y.norm(),
            _('Maximum'): self.modelEvaluators[0].y.max(),
            _('+/- d'): self.modelEvaluators[0].y.stdev_s()
        })
        df=pandas.DataFrame(data)
        return df.to_html()

    def model_table(self):
        featureselection = dict()
        for i, name in enumerate(self.featurenames):
            name = _(name)
            featureselection.update({name: [selectedfeature[i] for selectedfeature in self.selectedfeatures]})

        data = dict({
            _('Number of training data'): [CV.trainingdata_count()[0] for CV in self.modelEvaluators],
            _('Error/STDEV'): [round(CV.computeRelError()[0],2) for CV in self.modelEvaluators],
            _('P%'): [round(CV.computeP()[0],2) for CV in self.modelEvaluators]
        })
        data.update(featureselection)
        df = pandas.DataFrame(data)
        return df.sort_values(by=[_('Error/STDEV')])

    def __model_htmltable(self):
        # pandas.set_option('display.max_colwidth', 50)
        return self.model_table().to_html(justify='justify-all')

    def plot_timeseries(self):
        fig, ax = self.__prepare_figure()
        [ax.plot(
            CV.forecast.timeseries,
            color='red',
            label=_("individual forecasts").decode('utf-8'),
            alpha=.2
        ) for CV in self.modelEvaluators]
        df_concat = pandas.concat(([CV.forecast.timeseries for CV in self.modelEvaluators]))
        by_row_index = df_concat.groupby(df_concat.index)
        df_means = by_row_index.mean()
        ax.plot(df_means, color='black', label=_('mean forecast').decode('utf-8'))
        ax.plot(
            self.modelEvaluators[0].y_adj.timeseries,
            color='green',
            label=_('observed').decode('utf-8')
        )
        ax.set_xlabel(_('Year').decode('utf-8'))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-3:], labels[-3:])
        return fig

    @staticmethod
    def load_template_file(filename='template_seasonal.html'):
        template_path = path.join(path.dirname(__file__), filename)
        with open(template_path, 'r') as template_path:
            page = Template(template_path.read())

        return page

    @classmethod
    def encode_utf8(cls, template_vars):
        for key, value in template_vars.iteritems():
            template_vars[key] = to_str(value)

    def write_html(
            self,
            username,
            organization,
            site_code,
            site_name,
            filename=None,
            htmlpage=None,
            language='en'
    ):

        locales = environ.get('LOCALES_PATH', 'locales')
        t = gettext.translation('messages', locales, languages=[language])
        t.install()

        page = self.load_template_file()

        timeseries_plot = self.__encode_figure(self.plot_timeseries())
        quality_assessment_table = self.__table_summary()
        model_table = self.__model_htmltable()

        report_data = {
            'SITE_INFO': _('Station: {code} - {name}').format(code=site_code, name=site_name),
            'USERNAME': username,
            'ORGANIZATION': organization,
            'TITLE': _('Forecast Model Training Report'),
            'REPORT_DATE': format_date(format='long', locale=language),
            'PLOTS_HEADER': _('{frequency} Forecast Model Quality Assessment').format(
                frequency=_('seasonal').capitalize()),
            'MODEL_TABLE_LABEL': _('Model table'),
            'MODEL_TABLE': model_table,
            'QUALITY_ASSESSMENT_LABEL': _('Quality Assessment'),
            'QUALITY_ASSESSMENT_TABLE': quality_assessment_table,
            'TIMESERIES_LABEL': _('Timeseries plot'),
            'TIMESERIES_PLOT': timeseries_plot,
        }

        self.encode_utf8(report_data)

        if filename:
            htmlpage = open(filename, 'w')
            htmlpage.write(page.safe_substitute(**report_data))
            htmlpage.close()
            return filename
        elif htmlpage:
            htmlpage.write(page.safe_substitute(**report_data))
            return htmlpage

    def __encode_figure(self, fig):

        with tempfile.TemporaryFile(suffix=".png") as tmpfile:
            fig.savefig(tmpfile, format="png")
            tmpfile.seek(0)
            encoded = base64.b64encode(tmpfile.read())
            tmpfile.close()
        return encoded

