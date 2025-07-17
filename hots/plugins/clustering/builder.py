import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

def build_matrix_indiv_attr(df, tick_field, indiv_field, metrics, id_map):
    lines=[]
    for cid, group in df.groupby(indiv_field):
        row={tick: val for tick,val in zip(group[tick_field], group[metrics[0]])}
        row[indiv_field]=cid
        lines.append(row)
    mat=pd.DataFrame(lines).fillna(0).set_index(indiv_field)
    return mat.loc[sorted(mat.index, key=lambda x:id_map[x])]

def build_similarity_matrix(mat):
    return squareform(pdist(mat.values,'euclidean'))
