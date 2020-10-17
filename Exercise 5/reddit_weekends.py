import sys
import numpy as np
import pandas as pd
import gzip
import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt

OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)


def main():
    reddit_counts = sys.argv[1]

    # Opening and reading gzip file
    counts_gz = gzip.open(reddit_counts, 'rt', encoding = 'utf-8')
    counts = pd.read_json(counts_gz, lines = True)

    # Filtering out the dates before 2012 and after 2013 and in subreddit Canada
    #counts = counts[counts['date'] >= datetime.date(2012,1,1)]
    #counts = counts[counts['date'] <= datetime.date(2013,12,31)]
    #counts = counts[counts['subreddit'] == 'canada']
    #counts = counts.reset_index(drop=True)

    counts = counts[((counts['date'].dt.year == 2012) | (counts['date'].dt.year == 2013))]
    counts= counts[(counts['subreddit'] =='canada')]

    # Separating weekdays and weekends
    counts_wday = counts[(counts['date'].map(lambda x: datetime.date.weekday(x)) != 5) & (counts['date'].map(lambda x: datetime.date.weekday(x)) != 6)]
    counts_wday = counts_wday.reset_index(drop=True)
    
    counts_wend = counts[(counts['date'].map(lambda x: datetime.date.weekday(x)) == 5) | (counts['date'].map(lambda x: datetime.date.weekday(x)) == 6)]
    counts_wend = counts_wend.reset_index(drop=True)

    # Computing mean for weekdays and weekends
    mean_wday = counts_wday['comment_count'].mean()
    mean_wend = counts_wend['comment_count'].mean()

    # Making variables only with 'comment_count'
    wday_cc = counts_wday['comment_count']
    wend_cc = counts_wend['comment_count']


    # Student's T-Test
    ttest = stats.ttest_ind(wday_cc, wend_cc).pvalue
    lev_pval = stats.levene(wday_cc, wend_cc).pvalue
    wday_norm_p = stats.normaltest(wday_cc).pvalue
    wend_norm_p = stats.normaltest(wend_cc).pvalue

    # Plotting Histogram
    plt.hist(wday_cc)
    plt.figure()
    plt.hist(wend_cc)


    # Fix 1: Transforming data to various scales

    # Log
    wday_log = wday_cc.apply(np.log)
    wend_log = wend_cc.apply(np.log)
    wday_log_p = stats.normaltest(wday_log).pvalue
    wend_log_p = stats.normaltest(wend_log).pvalue
    wend_log_lev_p = stats.levene(wend_log, wday_log).pvalue

    # Exp
    wday_exp = wday_cc.apply(np.exp)
    wend_exp = wend_cc.apply(np.exp)
    wday_exp_p = stats.normaltest(wday_exp).pvalue
    wend_exp_p = stats.normaltest(wend_exp).pvalue
    wend_exp_lev_p = stats.levene(wend_exp, wday_exp).pvalue

    # Square Root
    wday_sqrt = wday_cc.apply(np.sqrt)
    wend_sqrt = wend_cc.apply(np.sqrt)
    wday_sqrt_p = stats.normaltest(wday_sqrt).pvalue
    wend_sqrt_p = stats.normaltest(wend_sqrt).pvalue
    wend_sqrt_lev_p = stats.levene(wend_sqrt, wday_sqrt).pvalue

    # Square
    wday_sqr = wday_cc**2
    wend_sqr = wend_cc**2
    wday_sqr_p = stats.normaltest(wday_sqr).pvalue
    wend_sqr_p = stats.normaltest(wend_sqr).pvalue
    wend_sqr_lev_p = stats.levene(wend_sqr, wday_sqr).pvalue



    # Fix 2: The Central Limit Theorem

    # Weekdays
    date_wday = counts_wday['date'].apply(datetime.date.isocalendar)
    date_wday = date_wday.apply(pd.Series)
    counts_wday['Year'] = date_wday[0]
    counts_wday['Week'] = date_wday[1]
    wday_grp = counts_wday.groupby(['Year', 'Week']).aggregate('mean').reset_index()

    # Weekends
    date_wend = counts_wend['date'].apply(datetime.date.isocalendar)
    date_wend = date_wend.apply(pd.Series)
    counts_wend['Year'] = date_wend[0]
    counts_wend['Week'] = date_wend[1]
    wend_grp = counts_wend.groupby(['Year', 'Week']).aggregate('mean').reset_index()

    # Computing p values
    clt_t_stat= stats.ttest_ind(wend_grp['comment_count'],wday_grp['comment_count']).pvalue
    clt_wday_norm=stats.normaltest(wday_grp['comment_count']).pvalue
    clt_wend_norm=stats.normaltest(wend_grp['comment_count']).pvalue
    clt_lev_p=stats.levene(wday_grp['comment_count'], wend_grp['comment_count']).pvalue


    # U-test
    utest = 2*stats.mannwhitneyu(counts_wday['comment_count'], counts_wend['comment_count']).pvalue

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=ttest,
        initial_weekday_normality_p=wday_norm_p,
        initial_weekend_normality_p=wend_norm_p,
        initial_levene_p=lev_pval,
        transformed_weekday_normality_p=wday_sqrt_p,
        transformed_weekend_normality_p=wend_sqrt_p,
        transformed_levene_p=wend_sqrt_lev_p,
        weekly_weekday_normality_p=clt_wday_norm,
        weekly_weekend_normality_p=clt_wend_norm,
        weekly_levene_p=clt_lev_p,
        weekly_ttest_p=clt_t_stat,
        utest_p=utest,
    ))

    plt.show()


if __name__ == '__main__':
    main()
