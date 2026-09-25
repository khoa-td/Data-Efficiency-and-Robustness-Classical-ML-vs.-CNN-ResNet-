import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns



if __name__ == '__main__':
    df = pd.read_excel("Schema Log.xlsx")

    for val in df.isna().sum().values:
        if val != 0:
            print("Error: Data has Nan value")

    for i in range(240):
        if df.loc[i, "Accuracy"] < 5.0 or df.loc[i, "Accuracy"] >= 85.0:
            print("Data has error accuracy at line:", i)

    SigmaGroup = df.groupby(["Pipeline", "Model Name", "Percentage Data", "Seed"]).size().reset_index(name= "Group Sigma Noise")

    Count_Sigma = len(SigmaGroup["Group Sigma Noise"].unique())

    if Count_Sigma != 1:
        print("Error: Data has not enough sigma noise")

    df_ML = df[df['Pipeline'] == 'Machine Learning']
    df_DL = df[df['Pipeline'] == 'Deep Learning'].reset_index()
    ML_Group = df_ML.groupby(['Pipeline', 'Noise Sigma', 'Seed', 'Percentage Data'])["Accuracy"].agg(Max_Accuracy = 'max', Mean_Accuracy = 'mean').reset_index()

    ML_Clean = ML_Group[ML_Group['Noise Sigma'] == 0.0]
    DL_Clean = df_DL[df_DL['Noise Sigma'] == 0.0]

    fig, ax = plt.subplots(figsize= (8, 6))
    sns.lineplot(data= ML_Clean, x='Percentage Data', y= 'Mean_Accuracy', label= 'Classical ML(Mean)', markers= 'o', errorbar=('ci', 95), ax= ax)
    sns.lineplot(data= DL_Clean, x = 'Percentage Data', y= 'Accuracy', label= 'CNN (SmallResNet)', markers= 'o', errorbar= ('ci', 95), ax= ax)


    ax.set_xscale('log')
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(['5%', '10%', '25%', '50%', '100%'])

    ax.set_title("Learning Curve (Mean)")
    ax.set_xlabel("Percentage Data")
    ax.set_ylabel("Mean Accuracy (%)")
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.legend(loc= 'upper left')
    plt.tight_layout()


    plt.savefig("Learning Curve (Mean).png")

    fig2, ax2 = plt.subplots(figsize= (8, 6))
    sns.lineplot(data= ML_Clean, x= 'Percentage Data', y= 'Max_Accuracy', label= 'Classical ML(Max)', markers= 'o', errorbar= ('ci', 95), ax= ax2)
    sns.lineplot(data= DL_Clean,  x = 'Percentage Data', y= 'Accuracy', label= 'CNN (SmallResNet)', markers= 'o', errorbar= ('ci', 95), ax= ax2)


    ax2.set_xscale('log')
    ax2.set_xticks([0.05, 0.1, 0.25, 0.5, 1.0])
    ax2.set_xticklabels(['5%', '10%', '25%', '50%', '100%'])

    ax2.set_title("Learning Curve (Max)")
    ax2.set_xlabel("Percentage Data")
    ax2.set_ylabel("Max Accuracy (%)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax2.legend(loc= 'upper left')
    plt.tight_layout()


    plt.savefig("Leaning Curve (Max).png")

    ML_FullData = ML_Group[ML_Group['Percentage Data'] == 1.0]
    DL_FullData = df_DL[df_DL['Percentage Data'] == 1.0]

    fig3, ax3 = plt.subplots(figsize= (8, 6))
    sns.lineplot(data= ML_FullData, x= 'Noise Sigma', y= 'Mean_Accuracy', marker= 'o', markersize= 7, label = 'Classical ML (Mean)', ax= ax3)
    sns.lineplot(data= DL_FullData, x= 'Noise Sigma', y= 'Accuracy', marker= 'o', markersize= 7, label= "CNN (SmallResNet)", ax= ax3)

    for line in ax3.lines:
        for x, y in zip(*line.get_data()):
            ax3.text(x, y + 1.2, f"{y:.1f}%", ha="center")

    ax3.set_xticks([0.0, 12.75, 38.25, 76.5])

    ax3.set_title("Robustness Curve")
    ax3.set_xlabel("Noise Sigma")
    ax3.set_ylabel("Accuracy (%)")
    ax3.grid(True, linestyle='--', alpha=0.5)

    ax3.legend(loc= 'upper right')
    plt.tight_layout()

    plt.savefig("Robustness Curve.png")