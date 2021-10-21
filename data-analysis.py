import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("\n -- Load & Preprocess Data -- \n")
    # load data 
    data_path = "corpus_IOB.csv"
    data_raw = pd.read_csv(data_path, encoding="latin1" )
    data_raw = data_raw.dropna() # drop NAs
    data_raw = data_raw[~data_raw["Word"].str.contains('\n| ', na=False)] # remove '\n' rows after each abstract

    # split raw dataframe into multiple dataframes each representing an abstract
    id_idx_list = [idx for idx, val in enumerate(data_raw['Word'].tolist()) if '#' in val] # indices where to split dataframe (based on #ID rows)
    idx_mod = id_idx_list + [len(data_raw)] #[max(id_idx_list)+1]
    list_of_dfs = [data_raw.iloc[idx_mod[n]:idx_mod[n+1]] for n in range(len(idx_mod)-1)]

    # extract final test set
    list_dfs_train , list_dfs_test = train_test_split(list_of_dfs, test_size=0.1, random_state=42)

    full_train = pd.concat(list_dfs_train)
    full_train = full_train[~full_train["Tag"].str.contains('#')] # drop ID rows
    final_test = pd.concat(list_dfs_test)
    final_test = final_test[~final_test["Tag"].str.contains('#')] # drop ID rows

    #print(full_train[full_train["Tag"].str.contains('O|B-Gene|B-SNP|I-Gene|I-SNP|B-RS')==False])
    #print(full_train, final_test, sep='\n')

    fig, ax = plt.subplots(1,2)
    ax1 = sns.countplot(x=full_train['Tag'], ax=ax[0])
    ax2 = sns.countplot(x=final_test['Tag'], ax=ax[1])
    ax1.set_title('Training', fontsize=20)
    ax2.set_title('Test', fontsize=20)
    for p in ax1.patches:
        ax1.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()+5000), ha='center', va='top', color='black', size=12)
    for p in ax2.patches:
        ax2.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()+700), ha='center', va='top', color='black', size=12)
    #plt.savefig('output.png')
    plt.show()

    print("\n -- Done -- \n")

    return 0

if __name__ == "__main__":
    main()