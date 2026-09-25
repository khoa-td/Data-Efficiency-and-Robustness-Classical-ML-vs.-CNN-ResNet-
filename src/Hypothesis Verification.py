import pandas as pd
from scipy import stats
import numpy as np



if __name__ == '__main__':

    df = pd.read_excel("Schema Log.xlsx")

    model = df.groupby(['Pipeline', 'Noise Sigma', 'Seed', 'Percentage Data'])['Accuracy'].agg(Accuracy= 'mean').reset_index()
    ML_model = model[model['Pipeline'] == 'Machine Learning']
    DL_model = model[model['Pipeline'] == 'Deep Learning']

    
    paired_df = pd.merge(ML_model, DL_model, on= ['Noise Sigma', 'Seed', 'Percentage Data'], suffixes= (' ML', ' DL'), )
    print(paired_df.head())

    H1_df= paired_df[paired_df['Noise Sigma'] == 0.0]
    H1_res = []

    for percent, group in H1_df.groupby('Percentage Data'):
        ML_acc = group['Accuracy ML'].values
        DL_acc = group['Accuracy DL'].values

        t_stat, p_val = stats.ttest_rel(ML_acc, DL_acc)

        H1_res.append({
            'Percentage Data': percent,
            'Mean Accuracy ML': ML_acc.mean(),
            'Mean Accuracy DL': DL_acc.mean(),
            'Mean diff (ML - DL)': (ML_acc - DL_acc).mean(),
            't-statistic': t_stat,
            'p-value': p_val,
            'Significance (p < 0.05)' : 'Yes' if p_val < 0.05 else 'No',
        })

    report_H1 = pd.DataFrame(H1_res)

    seed_summary = (
        model[model['Noise Sigma'].isin([0.0, 76.5]) & model['Percentage Data'].isin([1.0])]
        .groupby(['Pipeline', 'Noise Sigma', 'Seed'])['Accuracy']
        .mean()
        .unstack('Noise Sigma')
        .reset_index()
    )

    seed_summary['Slope'] = (seed_summary[0.0] - seed_summary[76.5]) / 76.5
    ML_slopes = (seed_summary[seed_summary['Pipeline'] == 'Machine Learning']).sort_values('Seed')['Slope'].values
    DL_slopes = (seed_summary[seed_summary['Pipeline'] == 'Deep Learning']).sort_values('Seed')['Slope'].values

    t_stat_h2, p_val_h2 = stats.ttest_rel(ML_slopes, DL_slopes)
    report_H2 = pd.DataFrame([
        {
            'Mean Slope ML': ML_slopes.mean(),
            'Mean Slope DL': DL_slopes.mean(),
            'Mean diff (ML - DL)': (ML_slopes - DL_slopes).mean(),
            't-statistic': t_stat_h2,
            'p-value': p_val_h2,
            'Significance (p < 0.05)': 'Yes' if p_val_h2 < 0.05 else 'No',
        }
    ])


    def check_h1_rule(row):
        pct = row['Percentage Data']
        diff = abs(row['Mean diff (ML - DL)'])
        if pct <= 0.30:
            return (
                'Satisfied (<= 5%)'
                if diff <= 5.0
                else f'VIOLATED ({diff:.2f}% > 5%)'
            )
        else:
            is_dl_higher = row['Mean diff (ML - DL)'] < 0
            return (
                'Satisfied' if (is_dl_higher and diff <= 20.0) else 'VIOLATED'
            )

    report_H1['H1 Verification Status'] = report_H1.apply(check_h1_rule, axis=1)

    curve_100 = (
        model[model['Percentage Data'] == 1.0]
        .groupby(['Pipeline', 'Noise Sigma'])['Accuracy']
        .mean()
        .unstack('Pipeline')
    )

    intervals = [(0.0, 12.75), (12.75, 38.25), (38.25, 76.5)]
    h2_steps = []

    for s_start, s_end in intervals:
        ml_start = curve_100.loc[s_start, 'Machine Learning']
        ml_end = curve_100.loc[s_end, 'Machine Learning']
        dl_start = curve_100.loc[s_start, 'Deep Learning']
        dl_end = curve_100.loc[s_end, 'Deep Learning']

        ml_rel_drop = ((ml_start - ml_end) / ml_start) * 100
        dl_rel_drop = ((dl_start - dl_end) / dl_start) * 100

        h2_steps.append({
            'Noise Interval': f'{s_start} -> {s_end}',
            'ML Initial Acc (%)': round(ml_start, 2),
            'ML Final Acc (%)': round(ml_end, 2),
            'ML Relative Drop (%)': f'{ml_rel_drop:.2f}%',
            'DL Initial Acc (%)': round(dl_start, 2),
            'DL Final Acc (%)': round(dl_end, 2),
            'DL Relative Drop (%)': f'{dl_rel_drop:.2f}%',
            'Faster Degradation': (
                'ML drops faster'
                if ml_rel_drop > dl_rel_drop
                else 'DL drops faster'
            ),
        })

    report_H2_intervals = pd.DataFrame(h2_steps)


    # print(report_H1.to_string(index=False))


    # print(report_H2.to_string(index=False))

    # print(report_H2_intervals.to_string(index=False))

    output_filename = "Statistical Test Results.xlsx"
    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        report_H1.to_excel(writer, sheet_name="H1 Analysis", index=False)
        report_H2.to_excel(writer, sheet_name="H2 Overall Slope", index=False)
        report_H2_intervals.to_excel(
            writer, sheet_name="H2 Interval Drops", index=False
        )

