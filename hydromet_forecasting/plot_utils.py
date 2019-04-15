# -*- encoding: UTF-8 -*-
import tempfile
import base64
import csv
import os
from dateutil import parser

import numpy as np
from matplotlib import pyplot as plt
from babel.dates import format_date, get_month_names


class PlotUtils(object):

    @staticmethod
    def get_month_day(frequency, period):
        if frequency == 'fiveday':
            month = ((period - 1) // 6) + 1
            day_of = period % 6 + 1
            day_mapper = {
                1: 1,
                2: 6,
                3: 11,
                4: 16,
                5: 21,
                6: 26,
            }
            return month, day_mapper[day_of]
        if frequency == 'decade':
            month = ((period - 1) // 3) + 1
            day_of = period % 3 + 1
            day_mapper = {
                1: 1,
                2: 11,
                3: 21,
            }
            return month, day_mapper[day_of]
        elif frequency == 'monthly':
            return period, 1

    @classmethod
    def get_line_points(cls, x, y):
        m, c = cls.linear_regression_using_least_squares(x, y)

        max_x = max(x) * 2
        min_x = -max_x
        min_y = m * min_x + c
        max_y = m * max_x + c
        return (min_x, max_x), (min_y, max_y)

    @staticmethod
    def linear_regression_using_least_squares(x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        num = 0
        den = 0
        for i in range(len(x)):
            num += (x[i] - x_mean) * (y[i] - y_mean)
            den += (x[i] - x_mean) ** 2

        if den:
            m = num / den
            c = y_mean - m * x_mean
            return m, c
        else:
            return 1, 0

    @classmethod
    def plot_ts_comparison(cls, x_data, y_data, frequency, language='en', encoded=True):

        if frequency == 'fiveday':
            rows, columns = 12, 6
            left = 0.06
            bottom = 0.08
            right = 0.94
            top = 0.92
            wspace = 0.4
            hspace = 0.8
            figsize = (11, 14)
        if frequency == 'decade':
            rows, columns = 6, 6
            left = 0.06
            bottom = 0.08
            right = 0.94
            top = 0.92
            wspace = 0.4
            hspace = 0.8
            figsize = (11, 7)
        elif frequency == 'monthly':
            rows, columns = 4, 3
            left = 0.06
            bottom = 0.08
            right = 0.94
            top = 0.92
            wspace = 0.2
            hspace = 0.4
            figsize = (11, 7)

        fig, axes = plt.subplots(rows, columns, figsize=figsize)
        fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        fig.text(
            0.5,
            0.01,
            r'${}\ [m^3/s]$'.format(_('Observed')).decode('utf-8'),
            horizontalalignment='center',
            fontsize=12
        )
        fig.text(
            0.01,
            0.5,
            r'${}\ [m^3/s]$'.format(_('Predicted')).decode('utf-8'),
            verticalalignment='center',
            fontsize=12,
            rotation=90
        )

        for row_num, row in enumerate(axes):
            for ax_num, ax in enumerate(row):
                # get period
                period = row_num * len(row) + ax_num + 1
                month, day = cls.get_month_day(frequency, period)
                like_str = "-{:02d}-{:02d}".format(month, day)

                x_sub_data = x_data.filter(like=like_str)
                y_sub_data = y_data.filter(like=like_str)

                # draw points
                ax.scatter(x_sub_data, y_sub_data, s=20, facecolors='none', edgecolors='black')
                # set period label
                ax.xaxis.set_label_position('top')

                if frequency == 'fiveday':
                    x_label = '{pentade} {period}'.format(
                        pentade=_('pentade').capitalize(),
                        period=period
                    )

                elif frequency == 'decade':
                    x_label = '{decade} {period}'.format(
                        decade=_('decade').capitalize(),
                        period=period
                    )

                elif frequency == 'monthly':
                    x_label = get_month_names(
                        width='wide',
                        locale=language
                    )[period]

                ax.set_xlabel(x_label.decode('utf-8'), fontweight='bold')

                # rotate y tick labels
                ax.tick_params(axis='y', labelrotation=90)

                # set only min and max values for ticks
                ax.set_xticks([round(x, 1) for x in ax.get_xlim()])
                ax.set_yticks([round(y, 1) for y in ax.get_ylim()])

                # "zoom out"
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                x_out = x_range * 0.1
                y_out = y_range * 0.1
                ax.set_xlim([ax.get_xlim()[0] - x_out, ax.get_xlim()[1] + x_out])
                ax.set_ylim([ax.get_ylim()[0] - y_out, ax.get_ylim()[1] + y_out])

                if not x_sub_data.empty:
                    a, b = cls.get_line_points(x_sub_data, y_sub_data)
                    ax.plot(a, b, color='red')  # predicted

        return cls.encode_figure(fig) if encoded else fig

    @staticmethod
    def encode_figure(fig):

        with tempfile.TemporaryFile(suffix=".png") as tmpfile:
            fig.savefig(tmpfile, format="png")
            tmpfile.seek(0)
            encoded = base64.b64encode(tmpfile.read())
            tmpfile.close()
        return encoded

    @staticmethod
    def load_csv(path):
        reader = csv.reader(open(path, 'r'))
        value_list = []
        date_list = []
        for row in reader:
            date_list.append(parser.parse(row[0]).date())
            value_list.append(float(row[1]))

        return pandas.Series(data=value_list, index=date_list, name=os.path.basename(path))

    @classmethod
    def prepare_figure(
            cls,
            max_index,
            x_label='',
            y_label='',
            width=12,
            height=3,
            month_names=False,
            language='en',
            title='',
    ):

        fig, ax = plt.subplots(1, 1)
        fig.set_figwidth(width)
        fig.set_figheight(height)
        fig.subplots_adjust(left=0.06, bottom=0.1, right=0.94, top=0.92)

        fig.text(
            0.5,
            0.01,
            x_label.decode('utf-8'),
            horizontalalignment='center',
            fontsize=12
        )

        fig.text(
            0.5,
            0.95,
            title.decode('utf-8'),
            horizontalalignment='center',
            fontsize=16,
            style='oblique',
        )

        fig.text(
            0.01,
            0.5,
            y_label.decode('utf-8'),
            verticalalignment='center',
            fontsize=12,
            rotation=90
        )

        # nr_bars = self.y.maxindex
        nr_bars = max_index

        if month_names:
            monthly_labels_pos = [
                p * nr_bars / 12.0 + (0.0416667 * nr_bars - 0.5) for p in range(0, 12)
            ]
            ax.set_xticks(monthly_labels_pos)
            ax.set_xticklabels(
                get_month_names(width='abbreviated', locale=language).itervalues())

            monthly_dividers = [
                p * nr_bars / 12.0 + (0.0416667 * nr_bars - 0.5) - 1.5 for p in range(0, 13)
            ]
            ax.set_xticks(monthly_dividers, minor=True)
        else:
            monthly_labels_pos = [
                p * nr_bars / nr_bars for p in range(0, nr_bars)
            ]
            ax.set_xticks(monthly_labels_pos)
            labels = [str(x + 1) for x in range(max_index)]
            max_labels_shown = 15
            skip = len(labels) // max_labels_shown
            if skip:
                for i in range(len(labels)):
                    if i % skip:
                        labels[i] = ''

            ax.set_xticklabels(labels)

        ax.grid(True, which="minor", axis="x", color="black", linestyle='--')
        ax.tick_params(axis="x", which="major", length=0)
        return fig, ax

    @classmethod
    def plot_rel_error(cls, rel_error, frequency, encoded=True, title=''):
        if frequency == 'fiveday':
            x_label = _('pentade').capitalize()
        elif frequency == 'decade':
            x_label = _('decade').capitalize()
        elif frequency == 'monthly':
            x_label = _('Month')

        fig, ax = cls.prepare_figure(
            max_index=len(rel_error),
            x_label=x_label,
            y_label=_("Error/STDEV"),
            title=title,
            height=5,
        )
        ax.boxplot(rel_error)
        ax.plot([0, ax.get_xlim()[1]], [0.674, 0.674], color='red', linestyle='dashed')
        return cls.encode_figure(fig) if encoded else fig

    @classmethod
    def plot_p(cls, p, frequency, encoded=True, title=''):
        if frequency == 'fiveday':
            x_label = _('pentade').capitalize()
        elif frequency == 'decade':
            x_label = _('decade').capitalize()
        elif frequency == 'monthly':
            x_label = _('Month')

        fig, ax = cls.prepare_figure(
            max_index=len(p),
            x_label=x_label,
            y_label=_("P[%]"),
            height=6,
            title=title,
        )
        ax.bar(range(0, len(p)), [x * 100 for x in p], width=0.7, color="gray", edgecolor='black')
        ax.set_ylim([0, 100])
        return cls.encode_figure(fig) if encoded else fig


if __name__ == '__main__':
    import argparse
    import gettext
    import pandas

    base_dir = os.path.join('example_data', 'plot_dev')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '-p',
        '--plot-type',
        help="Plot type",
        choices=(
            'plot_ts_comparison',
            'plot_rel_error',
            'plot_p',
        ),
        default='plot_ts_comparison',
    )

    arg_parser.add_argument(
        '-f',
        '--frequency',
        help="Frequency",
        choices=(
            'fiveday',
            'decade',
            'monthly',
            'seasonal',
        ),
        default='decade'
    )

    arg_parser.add_argument('-l', '--language', help='Language', choices=('en', 'ru'), default='en')
    args = arg_parser.parse_args()

    locales = os.environ.get('LOCALES_PATH', 'locales')
    t = gettext.translation('messages', locales, languages=[args.language])
    t.install()

    y_adj_path = os.path.join(base_dir, args.frequency, 'y_adj.csv')
    forecast_path = os.path.join(base_dir, args.frequency, 'forecast.csv')

    if args.plot_type == 'plot_ts_comparison':
        y_adj = PlotUtils.load_csv(y_adj_path)
        forecast = PlotUtils.load_csv(forecast_path)

        PlotUtils.plot_ts_comparison(
            y_adj, forecast, args.frequency, language=args.language, encoded=False)

    if args.plot_type == 'plot_rel_error':
        if args.frequency == 'fiveday':
            rel_error = [
                0.5652615753136934,
                0.5513277256998517,
                0.5262791757663606,
                0.5001540501000519,
                0.5186413123758862,
                0.4787133419008739,
                0.5592434546648722,
                0.5761377081448885,
                0.5200736062286804,
                0.539286495299186,
                0.5208580280883862,
                0.5438895216497983,
                0.6285850547216901,
                0.558386311831245,
                0.5869128276002127,
                0.5578586481005571,
                0.6314274283535166,
                0.46955922846999826,
                0.54156922359098,
                0.5231387572987912,
                0.5381331418255512,
                0.4475198701055407,
                0.7650007435368521,
                0.4422700622433673,
                0.6485639654676758,
                0.2881379712245952,
                0.7130794181007709,
                0.3460712945488712,
                0.7314694407980836,
                0.411408592030331,
                0.812947948991642,
                0.3739286888187171,
                0.7203514513251417,
                0.4056162909126814,
                0.6039754845100551,
                0.35769641423298626,
                0.7567092337712007,
                0.5088398726114542,
                0.8027798369976175,
                0.44437655209443105,
                0.6726068551115392,
                0.446954345299052,
                0.7852952739779304,
                0.5053295822292654,
                0.6305906429402922,
                0.44926127625976464,
                0.7171472607348955,
                0.4828645548042113,
                0.6587618842652487,
                0.4156892811401851,
                0.5339452287420132,
                0.5442623796518833,
                0.5512687605508411,
                0.43400783825122213,
                0.6095336886952397,
                0.48067671490061115,
                0.5509040468406358,
                0.4813677685931509,
                0.5460633534247712,
                0.5005956161592727,
                0.6125431802247321,
                0.556551647011301,
                0.60170126081544,
                0.5689988835643742,
                0.6464874277565515,
                0.4756554376329388,
                0.5544883999352107,
                0.5110528819465447,
                0.414209500231543,
                0.4455472211971826,
                0.5757582710807257,
                0.4669807193552419,
            ]
        elif args.frequency == 'decade':
            rel_error = [
            0.3755913647451166,
            0.3008586080078136,
            0.3389673495917839,
            0.2630958235606327,
            0.28072176684393074,
            0.24358040443782097,
            0.29218901557711935,
            0.3691242211084926,
            0.33154760391565813,
            0.2669608082897862,
            0.4411112799342047,
            0.6145082364622891,
            0.6593701879911972,
            0.6224796316591689,
            0.7907946579399917,
            0.749657146686847,
            0.638591209354681,
            0.6624231359410173,
            0.7242203409887683,
            0.7498364235861119,
            0.6425004072193109,
            0.7253347323187553,
            0.6173496405478044,
            0.629862140680635,
            0.4902780931032714,
            0.5800343891294351,
            0.46838621631855354,
            0.3405035946924497,
            0.3434653118211566,
            0.3308212878716098,
            0.27963154690246067,
            0.30837366413452255,
            0.2543454134345856,
            0.25102936845551493,
            0.24575001587122342,
            0.22191620604395182,
        ]
        elif args.frequency == 'monthly':
            rel_error = [
                0.7920407938500859,
                0.2523890472639323,
                0.3519820219032708,
                0.2630958235606327,
                0.28072176684393074,
                0.24358040443782097,
                0.29218901557711935,
                0.3691242211084926,
                0.33154760391565813,
                0.2669608082897862,
                0.4411112799342047,
                0.6145082364622891,
            ]
        PlotUtils.plot_rel_error(
            rel_error,
            args.frequency,
            encoded=False,
            title=_('Scaled Error [RMSE/STDEV]')
        )

    if args.plot_type == 'plot_p':
        if args.frequency == 'fiveday':
            p = [
                0.7125,
                0.725,
                0.7375,
                0.7439024390243902,
                0.7195121951219512,
                0.7073170731707317,
                0.6951219512195121,
                0.6951219512195121,
                0.7195121951219512,
                0.6829268292682927,
                0.6951219512195121,
                0.6829268292682927,
                0.7073170731707317,
                0.6707317073170732,
                0.7073170731707317,
                0.6829268292682927,
                0.5853658536585366,
                0.7926829268292683,
                0.7317073170731707,
                0.7073170731707317,
                0.6829268292682927,
                0.8048780487804879,
                0.524390243902439,
                0.8048780487804879,
                0.6951219512195121,
                0.9512195121951219,
                0.573170731707317,
                0.8780487804878049,
                0.5609756097560976,
                0.8658536585365854,
                0.45121951219512196,
                0.8658536585365854,
                0.5375,
                0.825,
                0.65,
                0.875,
                0.4875,
                0.7625,
                0.5375,
                0.7875,
                0.575,
                0.75,
                0.5,
                0.7,
                0.6375,
                0.8375,
                0.5875,
                0.7468354430379747,
                0.5443037974683544,
                0.810126582278481,
                0.6835443037974683,
                0.65,
                0.6875,
                0.8,
                0.6625,
                0.775,
                0.75,
                0.725,
                0.675,
                0.75,
                0.6375,
                0.6125,
                0.7125,
                0.6625,
                0.5875,
                0.775,
                0.65,
                0.75,
                0.8,
                0.7875,
                0.7,
                0.7625
            ]
        elif args.frequency == 'decade':
            p = [
            0.825,
            0.9,
            0.9,
            0.9390243902439024,
            0.9390243902439024,
            0.9634146341463414,
            0.9390243902439024,
            0.8902439024390244,
            0.8902439024390244,
            0.9512195121951219,
            0.7682926829268293,
            0.6585365853658537,
            0.5975609756097561,
            0.6341463414634146,
            0.4634146341463415,
            0.573170731707317,
            0.5875,
            0.575,
            0.5625,
            0.5625,
            0.6375,
            0.55,
            0.65,
            0.6375,
            0.7215189873417721,
            0.6708860759493671,
            0.7468354430379747,
            0.9125,
            0.875,
            0.9125,
            0.95,
            0.875,
            0.9375,
            0.95,
            0.9625,
            0.9875,
        ]
        elif args.frequency == 'monthly':
            p = [
                0.49382716049382713,
                0.9629629629629629,
                0.9135802469135802,
                0.9390243902439024,
                0.9390243902439024,
                0.9634146341463414,
                0.9390243902439024,
                0.8902439024390244,
                0.8902439024390244,
                0.9512195121951219,
                0.7682926829268293,
                0.6585365853658537,
            ]
        PlotUtils.plot_p(
            p,
            args.frequency,
            encoded=False,
            title=_('P% Plot')
        )

    plt.show()
