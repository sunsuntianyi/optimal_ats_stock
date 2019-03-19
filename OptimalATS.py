import PySimpleGUI as sg
import pandas as pd
from functools import reduce
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import pi


def user_input_GUI():
    global stock_share_hash, index_hash, chart_num
    layout = [
        [sg.Text('Please enter Portfolio and its individual stock share', font=("Helvetica bold", 20))],
        [sg.Text('Portfolio', size=(7, 1), font=("Helvetica", 16)),
         sg.InputText('', key='stock', do_not_clear=True, font=("Helvetica", 16))],
        [sg.Text('Share', size=(7, 1), font=("Helvetica", 16)),
         sg.InputText('', key='share', do_not_clear=True, font=("Helvetica", 16))],
        [sg.Text('Data Timeline:', font=("Helvetica bold", 16))],
        [sg.InputCombo(('Most Recent Week', 'Most Recent Month', 'All Data'), key='time', font=("Helvetica", 16), size=(16, 1))],
        [sg.Text('Number of Radar Chart (max 8):', font=("Helvetica bold", 16))],
        [sg.InputText('3', key='chart', do_not_clear=True, size=(3, 1), font=("Helvetica", 16))],
        [sg.Text('Indices Weight (0 - 1)', font=("Helvetica bold", 16))],
        [sg.Text('SPI:', size=(4, 1), font=("Helvetica", 16)),
         sg.InputText('1', key='spi_weight', do_not_clear=True, size=(3, 1), font=("Helvetica", 16)),
         sg.Text('TPI:', size=(4, 1), font=("Helvetica", 16)),
         sg.InputText('1', key='tpi_weight', do_not_clear=True, size=(3, 1), font=("Helvetica", 16)),
         sg.Text('SLI:', size=(4, 1), font=("Helvetica", 16)),
         sg.InputText('1', key='sli_weight', do_not_clear=True, size=(3, 1), font=("Helvetica", 16)),
         sg.Text('PRI:', size=(4, 1), font=("Helvetica", 16)),
         sg.InputText('1', key='pri_weight', do_not_clear=True, size=(3, 1), font=("Helvetica", 16)),
         sg.Text('ATSI:', size=(4, 1), font=("Helvetica", 16)),
         sg.InputText('1', key='atsi_weight', do_not_clear=True, size=(3, 1), font=("Helvetica", 16))],
        [sg.Submit('Analyze', font=("Helvetica", 16)), sg.Exit(font=("Helvetica", 16))]
    ]

    window = sg.Window('Client Tool for Finding Optimal ATS').Layout(layout)

    while True:
        event, stock_share_hash_old = window.Read()

        if event is None or event == 'Exit':
            break

        else:

            for key, value in stock_share_hash_old.items():
                stock_share_hash_old.update({key: value.split(',')})

            newlist = []
            for value in stock_share_hash_old['share']:
                newlist.append(int(value))
            stock_share_hash_old.update({'share': newlist})

            stock_share_hash = {}
            for index in range(len(stock_share_hash_old['stock'])):
                stock_share_hash[stock_share_hash_old['stock'][index].upper()] = stock_share_hash_old['share'][index]

            chart_num = int(stock_share_hash_old['chart'][0])

            time = stock_share_hash_old['time'][0]

            index_hash = {}
            index_hash.update({'spi_weight': stock_share_hash_old['spi_weight']})
            # stock_share_hash.pop('spi_weight')
            index_hash.update({'tpi_weight': stock_share_hash_old['tpi_weight']})
            # stock_share_hash.pop('tpi_weight')
            index_hash.update({'sli_weight': stock_share_hash_old['sli_weight']})
            # stock_share_hash.pop('sli_weight')
            index_hash.update({'pri_weight': stock_share_hash_old['pri_weight']})
            # stock_share_hash.pop('pri_weight')
            index_hash.update({'atsi_weight': stock_share_hash_old['atsi_weight']})
            # stock_share_hash.pop('atsi_weight')

            # Remove spaces in key
            stock_share_hash = {k.replace(' ', ''): v for k, v in stock_share_hash.items()}

            finra = subset_data(choice=time, finra_data=finra_data)
            overall_score(input=stock_share_hash, finra_data=finra, sector_data=sector_data)
            market_liquidity_ratio(stock_share_hash=stock_share_hash, finra_data=finra, ratio_data=ratio_data)
            sg.Popup('Most Optimal ATS for Routing this Portfolio:',
                     stock_share_hash, score_sorted, '\n'.join(list_mlr), font=("Helvetica", 16))

    window.Close()

    return


def subset_data(choice, finra_data):
    global week

    finra_data['Week'] = pd.to_datetime(finra_data['Week'])

    if choice == 'Most Recent Week':
        week = 1
        data = finra_data[finra_data['Week'] == min(finra_data.Week.unique())]

    elif choice == 'Most Recent Month':
        week = 4
        data = finra_data[finra_data['Week'].isin(sorted(finra_data.Week.unique())[0:4])]

    else:
        data = finra_data
        week = len(data.Week.unique())

    return data


def portfolio_share_prop_index(portfolio, data):

    portfolio_data = data[data['Symbol'].isin(portfolio)]

    ats_list = data.ATS_MPID.unique()

    hash_portfolio = {stock: [] for stock in portfolio}

    for stock in portfolio:
        each_stock = portfolio_data[portfolio_data['Symbol'] == stock]
        stock_sum_by_ats = each_stock.groupby(['ATS_MPID'])['Shares'].sum()
        model = stock_sum_by_ats / sum(stock_sum_by_ats)
        # model_normalized = (model - min(model)) / (max(model) - min(model))

        for ats in ats_list:
            if ats not in model.index:
                new_ats = pd.Series([0], index=[ats])
                model = model.append(new_ats)

        hash_portfolio.update({stock: model.sort_values(ascending=False)})

    worthfullness_index = pd.Series()
    for ats in ats_list:
        if ats not in worthfullness_index.index:
            new_ats = pd.Series([0], index=[ats])
            worthfullness_index = worthfullness_index.append(new_ats)

    for stock in portfolio:
        worthfullness_index += hash_portfolio[stock]

    worthfullness_index_normalized = \
        (worthfullness_index - min(worthfullness_index)) / (max(worthfullness_index) - min(worthfullness_index))
    # worthfullness_index /= len(portfolio)

    return worthfullness_index_normalized


def portfolio_trade_prop_index(portfolio, data):

    portfolio_data = data[data['Symbol'].isin(portfolio)]

    ats_list = data.ATS_MPID.unique()

    hash_portfolio = {stock: [] for stock in portfolio}

    for stock in portfolio:
        each_stock = portfolio_data[portfolio_data['Symbol'] == stock]
        stock_sum_by_ats = each_stock.groupby(['ATS_MPID'])['Trades'].sum()
        model = stock_sum_by_ats / sum(stock_sum_by_ats)
        # model_normalized = (model - min(model)) / (max(model) - min(model))

        for ats in ats_list:
            if ats not in model.index:
                new_ats = pd.Series([0], index=[ats])
                model = model.append(new_ats)

        hash_portfolio.update({stock: model.sort_values(ascending=False)})

    worthfullness_index = pd.Series()
    for ats in ats_list:
        if ats not in worthfullness_index.index:
            new_ats = pd.Series([0], index=[ats])
            worthfullness_index = worthfullness_index.append(new_ats)

    for stock in portfolio:
        worthfullness_index += hash_portfolio[stock]

    worthfullness_index_normalized = \
        (worthfullness_index - min(worthfullness_index)) / (max(worthfullness_index) - min(worthfullness_index))
    # worthfullness_index /= len(portfolio)

    return worthfullness_index_normalized


# test_portfolio = ['A', 'AA']
# data = pd.read_csv("/Users/TonY/Desktop/capstone/finra.csv")
# portfolio_share_prop_index(test_portfolio, data)
# a = portfolio_share_prop_index(test_portfolio, data) + portfolio_trade_prop_index(portfolio, data)


def sector_liquidity_index(portfolio, data, sector_data):
    sector_list = []
    sector_stock_hash = {}
    hash_index = {}
    ats_list = data.ATS_MPID.unique()

    for stock in portfolio:
        sector_list.append(sector_data.loc[sector_data['Symbol'] == stock, 'sector'].iloc[0])
    sector_list = set(sector_list)

    for sector in sector_list:
        sector_stock_hash.update(
            {sector: sector_data.loc[sector_data['sector'] == sector, 'Symbol'].values[:].tolist()})

    for sector in sector_stock_hash:
        portfolio_data = data[data['Symbol'].isin(sector_stock_hash[sector])]
        sector_sum_by_ats = portfolio_data.groupby(['ATS_MPID'])['Shares'].sum()
        model = sector_sum_by_ats / sum(sector_sum_by_ats)
        # model_normalized = (model - min(model)) / (max(model) - min(model))
        for ats in ats_list:
            if ats not in model.index:
                new_ats = pd.Series([0], index=[ats])
                model = model.append(new_ats)

        hash_index.update({sector: model})

    sl_index = pd.Series()
    for ats in ats_list:
        if ats not in sl_index.index:
            new_ats = pd.Series([0], index=[ats])
            sl_index = sl_index.append(new_ats)

    for sector in sector_list:
        sl_index += hash_index[sector]

    sl_index_normalized = (sl_index - min(sl_index)) / (max(sl_index) - min(sl_index))
    # sl_index /= len(sector_list)

    return sl_index_normalized


# data = pd.read_csv("/Users/TonY/Desktop/capstone/finra.csv")
# sector_data = pd.read_csv('/Users/TonY/Desktop/capstone/market_cap_sector_mktcapcategory_by_symbol.csv', encoding='utf-8')
# test_portfolio = ['A', 'AA', 'ADRO', 'AABA']
# b = sector_liquidity_index(test_portfolio, data, sector_data)
# len(b)


def participation_rate_index(hash_portfolio_share, data):
    hash_par_rate_index = {}
    ats_list = data.ATS_MPID.unique()

    for stock in hash_portfolio_share:
        data_selected = data.loc[data['Symbol'] == stock]
        result = data_selected.groupby('ATS_MPID')['Shares'].sum() / week
        model = hash_portfolio_share[stock] / result
        # model_normalized = (model - min(model)) / (max(model) - min(model))

        for ats in ats_list:
            if ats not in model.index:
                new_ats = pd.Series([0], index=[ats])
                model = model.append(new_ats)

        hash_par_rate_index.update({stock: model})

    pr_index = pd.Series()
    for ats in ats_list:
        if ats not in pr_index.index:
            new_ats = pd.Series([0], index=[ats])
            pr_index = pr_index.append(new_ats)

    for stock in hash_portfolio_share:
        pr_index += hash_par_rate_index[stock]

    pr_index_normalized = (pr_index - min(pr_index)) / (max(pr_index) - min(pr_index))
    # pr_index /= len(hash_portfolio_share)

    for i in range(len(pr_index_normalized)):
        if pr_index_normalized[i] != 0:
            pr_index_normalized[i] = 1 - pr_index_normalized[i]

    return pr_index_normalized


# data = pd.read_csv("/Users/TonY/Desktop/capstone/finra.csv")
#
# hash_portfolio_share = {'A': 100, "AA": 200}
# participation_rate_index(hash_portfolio_share, data)


def avg_trade_size_index(hash_portfolio_share, data):
    hash_par_rate_index = {}
    ats_list = data.ATS_MPID.unique()

    for stock in hash_portfolio_share:
        data_selected = data.loc[data['Symbol'] == stock]
        share_sum = data_selected.groupby('ATS_MPID')['Shares'].sum()
        trade_sum = data_selected.groupby('ATS_MPID')['Trades'].sum()
        model = hash_portfolio_share[stock] / (share_sum / trade_sum)
        # model_normalized = (model - min(model)) / (max(model) - min(model))

        for ats in ats_list:
            if ats not in model.index:
                new_ats = pd.Series([0], index=[ats])
                model = model.append(new_ats)

        hash_par_rate_index.update({stock: model})

    pr_index = pd.Series()
    for ats in ats_list:
        if ats not in pr_index.index:
            new_ats = pd.Series([0], index=[ats])
            pr_index = pr_index.append(new_ats)

    for stock in hash_portfolio_share:
        pr_index += hash_par_rate_index[stock]

    pr_index_normalized = (pr_index - min(pr_index)) / (max(pr_index) - min(pr_index))
    # pr_index /= len(hash_portfolio_share)

    return pr_index_normalized


def overall_score(input, finra_data, sector_data):
    # input = user_input_GUI()
    global spi, tpi, sli, pri, atsi, score_sorted

    spi = portfolio_share_prop_index(portfolio=input.keys(), data=finra_data)
    tpi = portfolio_trade_prop_index(portfolio=input.keys(), data=finra_data)
    sli = sector_liquidity_index(portfolio=input.keys(), data=finra_data, sector_data=sector_data)
    pri = participation_rate_index(hash_portfolio_share=input, data=finra_data)
    atsi = avg_trade_size_index(hash_portfolio_share=input, data=finra_data)

    score = float(index_hash['spi_weight'][0]) * spi + float(index_hash['tpi_weight'][0]) * tpi + \
            float(index_hash['sli_weight'][0]) * sli + float(index_hash['pri_weight'][0]) * pri + \
            float(index_hash['atsi_weight'][0]) * atsi

    weight_list = [float(index_hash['spi_weight'][0]), float(index_hash['tpi_weight'][0]),
                   float(index_hash['sli_weight'][0]), float(index_hash['pri_weight'][0]),
                   float(index_hash['atsi_weight'][0])]

    count_non_zero = 0
    for weight in weight_list:
        if weight != 0:
            count_non_zero += 1

    score /= count_non_zero
    score_sorted = round(score.sort_values(ascending=False), 3)[0:chart_num+3]
    # print(stock_share_hash, '\n', score_sorted[0:5])

    return radar_chart()


def index_to_dataframe():
    data_frame_spi = pd.DataFrame(spi, columns=['SPI'])
    data_frame_spi.index.name = 'ATS'
    data_frame_tpi = pd.DataFrame(tpi, columns=['TPI'])
    data_frame_tpi.index.name = 'ATS'
    data_frame_sli = pd.DataFrame(sli, columns=['SLI'])
    data_frame_sli.index.name = 'ATS'
    data_frame_pri = pd.DataFrame(pri, columns=['PRI'])
    data_frame_pri.index.name = 'ATS'
    data_frame_atsi = pd.DataFrame(atsi, columns=['ATSI'])
    data_frame_atsi.index.name = 'ATS'

    data_frames = [data_frame_spi, data_frame_tpi, data_frame_sli, data_frame_pri, data_frame_atsi]

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['ATS'], how='outer'), data_frames)

    return df_merged


def radar_chart():
    plt.close('all')
    df = index_to_dataframe()

    # number of variable
    categories = list(df)[0:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", '0.8'], color="grey", size=7)
    plt.ylim(0, 1)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 groups makes the chart unreadable

    top_ats = score_sorted
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    for chart in range(chart_num):
        values = df.loc[df.index == top_ats.index[chart]].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=top_ats.index[chart])
        ax.fill(angles, values, color[chart], alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(stock_share_hash, y=1.08)
    plt.show(block=False)


def market_liquidity_ratio(stock_share_hash, finra_data, ratio_data):
    global list_mlr
    # finra_data['Week'] = pd.to_datetime(finra_data['Week'])
    last_week = finra_data['Week'].max()
    last_week_data = finra_data[finra_data['Week'] == last_week]

    lastweek_shares = last_week_data.groupby(['Symbol'])['Shares'].sum()
    lastweek_shares = pd.DataFrame(lastweek_shares)

    ratio_data_merged = pd.merge(left=lastweek_shares, right=ratio_data, left_on="Symbol", right_on="symbol", how="left")

    ratio_data_merged['Total_Volume'] = ratio_data_merged['volume'] / ratio_data_merged['marketPercent']
    ratio_data_merged['ADV'] = ratio_data_merged['Shares'] / 5
    ratio_data_merged['Ratio'] = ratio_data_merged['ADV'] / ratio_data_merged['Total_Volume']

    portfolio_ratio = ratio_data_merged[ratio_data_merged['symbol'].isin(stock_share_hash.keys())]
    list_mlr = []
    # bold_start = '\033[1m'
    # bold_end = '\033[0m'
    for i in range(len(portfolio_ratio)):
        list_mlr.append(str(round(portfolio_ratio.iloc[i]['Ratio'] * 100, 1)) + '%' +
                        ' of Stock ' + portfolio_ratio.iloc[i]['symbol'] +
                        ' was traded on all its ATSs')

    return


if __name__ == "__main__":

    # import pandas as pd

    finra_data = pd.read_csv("/Users/TonY/Desktop/capstone/finra.csv")
    sector_data = pd.read_csv('/Users/TonY/Desktop/capstone/market_cap_sector_mktcapcategory_by_symbol.csv')
    ratio_data = pd.read_csv("/Users/TonY/Desktop/capstone/Ratio.csv")

    user_input_GUI()