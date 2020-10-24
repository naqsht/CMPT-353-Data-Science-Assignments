import sys
import numpy as np
import pandas as pd
import scipy.stats as st


OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g} \n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)


def main():
    searchdata_file = sys.argv[1]
    searches = pd.read_json(searchdata_file, orient='records', lines=True)

    # Separating people given Old and New Interfaces
    new_uid = searches[searches['uid']%2 == 1] # Odd uid, New design
    old_uid = searches[searches['uid']%2 == 0] # Even uid, Old design

    # Questions of interest
    # 1. Did a different fraction of users have search count > 0 ?
    # 2. Is the number of searches per user different ?

    #print(new_uid)
    #print(old_uid)
    
    # For Users
    # Computing users with search counts > 0  and search counts = 0 for both New and Old Interfaces
    new_search_user = len(new_uid[new_uid['search_count']>0].index)
    old_search_user = len(old_uid[old_uid['search_count']>0].index)
    new_nosearch_user = len(new_uid[new_uid['search_count']==0].index)
    old_nosearch_user = len(old_uid[old_uid['search_count']==0].index)

    # Computing chi-squared values by inputting contingency table for Users
    contingency_user = [[old_search_user, old_nosearch_user], [new_search_user, new_nosearch_user]]
    chi, p, dof, ex = st.chi2_contingency(contingency_user)
    mannwhit_user_p = st.mannwhitneyu(new_uid['search_count'],old_uid['search_count']).pvalue

    # For Instructors
    # Filtering out the users which are Instructors
    new_inst = new_uid[new_uid['is_instructor'] == True]
    old_inst = old_uid[old_uid['is_instructor'] == True]

    # Computing instructors with (search counts > 0)  and (search counts = 0) for both New and Old Interfaces
    new_search_inst = len(new_inst[new_inst['search_count']>0].index)
    old_search_inst = len(old_inst[old_inst['search_count']>0].index)
    new_nosearch_inst = len(new_inst[new_inst['search_count']==0].index)
    old_nosearch_inst = len(old_inst[old_inst['search_count']==0].index)

    # Computing chi-squared values by inputting contingency table for Instructors
    contingency_inst = [[old_search_inst, old_nosearch_inst], [new_search_inst, new_nosearch_inst]]
    chi1, p1, dof1, ex1 = st.chi2_contingency(contingency_inst)
    mannwhit_inst_p = st.mannwhitneyu(new_inst['search_count'],old_inst['search_count']).pvalue 
    

    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=p,
        more_searches_p=mannwhit_user_p,
        more_instr_p=p1,
        more_instr_searches_p=mannwhit_inst_p,
    ))


if __name__ == '__main__':
    main()
