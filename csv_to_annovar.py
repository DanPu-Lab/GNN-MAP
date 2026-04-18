import pandas as pd
import os
from datetime import datetime
chr_list = [str(chromosome) for chromosome in range(1, 23)] + ['X', 'Y']
def prepare_input(input_file, annovar_dir):
    """
    Prepare input file for ANNOVAR and SpliceAI.
    :param input_file: Original input filename of MAGPIE. Input file should be tab or comma separated files containing five headers: Chr, Start, End, Ref, Alt
    :param annovar_dir: Dictionary of ANNOVAR input file. (e.g. /data/output/annovar/)
    :param spliceai_dir: Dictionary of SpliceAI input file. (e.g. /data/output/spliceai/)
    :return: None
    """
    filename = os.path.splitext(os.path.basename(input_file))[0]
    data = pd.read_csv(input_file, low_memory=False)
    data.Chr = [str(i).replace('chr', '') if 'chr' in str(i) else str(i) for i in data.Chr.tolist()]
    #data.rename(columns={'hg19_chr': 'Chr','hg19_pos(1-based)':'POS','ref':'Ref','alt':'Alt','category':'CLASS'}, inplace=True)
    data = data[data.Chr.isin(chr_list)]
    print(data)
    data2 = data[['Chr', 'Start','End', 'Ref', 'Alt']]
    data2 = data.fillna('-')
    data2.insert(data.shape[-1], 'info', '.')
    data2.to_csv(f'{os.path.join(annovar_dir, filename)}.avinput', sep='\t', index=False, header=False)
prepare_input("/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/varibench.csv", "/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/")
