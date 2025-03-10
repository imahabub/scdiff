import wandb
import numpy as np
from cellot.losses.mmd import mmd_distance
from cellot.data.cell import read_single_anndata

def get_ckpt_path_from_artifact_id(artifact_id):
    run = wandb.init()
    artifact = run.use_artifact(artifact_id, type='model')
    artifact_dir = artifact.download()
    ckpt_path = f'{artifact_dir}/model.ckpt'
    return ckpt_path

def compute_mmd_loss(lhs, rhs, gammas):
    return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])

def load_markers(config, n_genes=50, gene_pool=None):
    data = read_single_anndata(config, path=None)
    key = f'marker_genes-{config.data.condition}-rank'

    if key not in data.varm:
        key = 'marker_genes-condition-rank'
        print('WARNING: using generic condition marker genes')
        
    if gene_pool is not None:
        gene_pool = set(gene_pool)
        potential_mgs = set(data.varm[key].index)
        valid_genes = potential_mgs.intersection(gene_pool)
    else:
        valid_genes = data.varm[key].index

    # Map from gene names to their original row indices
    gene_to_row_index = {gene: idx for idx, gene in enumerate(data.varm[key].index)}

    sel_mg = (
        data.varm[key].loc[valid_genes][config.data.target]
        .sort_values()
        .index
    )[:n_genes]

    # Convert sorted gene names to their original row indices
    marker_gene_indices = [gene_to_row_index[gene] for gene in sel_mg]

    return sel_mg, marker_gene_indices

def get_target_cond_idx(target):
    hardcoded_condition_label_list = ['2_methoxyestradiol',
                                    '_jq1',
                                    'a_366',
                                    'abexinostat',
                                    'abt_737',
                                    'ac480',
                                    'ag_14361',
                                    'ag_490',
                                    'aicar',
                                    'alendronate_sodium_trihydrate',
                                    'alisertib',
                                    'altretamine',
                                    'alvespimycin_hcl',
                                    'amg_900',
                                    'aminoglutethimide',
                                    'amisulpride',
                                    'anacardic_acid',
                                    'andarine',
                                    'ar_42',
                                    'at9283',
                                    'aurora_a_inhibitor_i',
                                    'avagacestat',
                                    'az_960',
                                    'azacitidine',
                                    'azd1480',
                                    'barasertib',
                                    'baricitinib',
                                    'belinostat',
                                    'bisindolylmaleimide_ix',
                                    'bms_265246',
                                    'bms_536924',
                                    'bms_754807',
                                    'bms_911543',
                                    'bosutinib',
                                    'brd4770',
                                    'busulfan',
                                    'capecitabine',
                                    'carmofur',
                                    'cediranib',
                                    'celecoxib',
                                    'cep_33779',
                                    'cerdulatinib',
                                    'cimetidine',
                                    'clevudine',
                                    'control',
                                    'costunolide',
                                    'crizotinib',
                                    'cudc_101',
                                    'cudc_907',
                                    'curcumin',
                                    'cyc116',
                                    'cyclocytidine_hcl',
                                    'dacinostat',
                                    'danusertib',
                                    'daphnetin',
                                    'dasatinib',
                                    'decitabine',
                                    'disulfiram',
                                    'divalproex_sodium',
                                    'droxinostat',
                                    'eed226',
                                    'ellagic_acid',
                                    'enmd_2076',
                                    'enmd_2076_l__tartaric_acid',
                                    'entacapone',
                                    'entinostat',
                                    'enzastaurin',
                                    'epothilone_a',
                                    'fasudil_hcl',
                                    'fedratinib',
                                    'filgotinib',
                                    'flavopiridol_hcl',
                                    'flll32',
                                    'fluorouracil',
                                    'fulvestrant',
                                    'g007_lk',
                                    'gandotinib',
                                    'givinostat',
                                    'glesatinib?',
                                    'gsk1070916',
                                    'gsk_j1',
                                    'gsk_lsd1_2hcl',
                                    'hesperadin',
                                    'iniparib',
                                    'ino_1001',
                                    'iox2',
                                    'itsa_1',
                                    'ivosidenib',
                                    'jnj_26854165',
                                    'jnj_7706621',
                                    'ki16425',
                                    'ki8751',
                                    'kw_2449',
                                    'lapatinib_ditosylate',
                                    'lenalidomide',
                                    'linifanib',
                                    'lomustine',
                                    'luminespib',
                                    'm344',
                                    'maraviroc',
                                    'mc1568',
                                    'meprednisone',
                                    'mercaptopurine',
                                    'mesna',
                                    'mk_0752',
                                    'mk_5108',
                                    'mln8054',
                                    'mocetinostat',
                                    'momelotinib',
                                    'motesanib_diphosphate',
                                    'navitoclax',
                                    'nilotinib',
                                    'nintedanib',
                                    'nvp_bsk805_2hcl',
                                    'obatoclax_mesylate',
                                    'ofloxacin',
                                    'panobinostat',
                                    'patupilone',
                                    'pci_34051',
                                    'pd173074',
                                    'pd98059',
                                    'pelitinib',
                                    'pf_3845',
                                    'pf_573228',
                                    'pfi_1',
                                    'pha_680632',
                                    'pirarubicin',
                                    'pj34',
                                    'pracinostat',
                                    'prednisone',
                                    'quercetin',
                                    'quisinostat_2hcl',
                                    'raltitrexed',
                                    'ramelteon',
                                    'regorafenib',
                                    'resminostat',
                                    'resveratrol',
                                    'rg108',
                                    'rigosertib',
                                    'roscovitine',
                                    'roxadustat',
                                    'rucaparib_phosphate',
                                    'ruxolitinib',
                                    's3i_201',
                                    's_ruxolitinib',
                                    'sb431542',
                                    'selisistat',
                                    'sgi_1776_free_base',
                                    'sirtinol',
                                    'sl_327',
                                    'sns_314',
                                    'sodium_phenylbutyrate',
                                    'sorafenib_tosylate',
                                    'srt1720_hcl',
                                    'srt2104',
                                    'srt3025_hcl',
                                    'streptozotocin',
                                    'tacedinaline',
                                    'tak_901',
                                    'tanespimycin',
                                    'tazemetostat',
                                    'temsirolimus',
                                    'tg101209',
                                    'tgx_221',
                                    'thalidomide',
                                    'thiotepa',
                                    'tie2_kinase_inhibitor',
                                    'tmp195',
                                    'tofacitinib_citrate',
                                    'toremifene_citrate',
                                    'tozasertib',
                                    'trametinib',
                                    'tranylcypromine_hcl',
                                    'triamcinolone_acetonide',
                                    'trichostatin_a',
                                    'tubastatin_a_hcl',
                                    'tucidinostat',
                                    'unc0379',
                                    'unc0631',
                                    'unc1999',
                                    'valproic_acid_sodium_salt',
                                    'vandetanib',
                                    'veliparib',
                                    'whi_p154',
                                    'wp1066',
                                    'xav_939',
                                    'ym155',
                                    'zileuton',
                                    'zm_447439']
    
    return hardcoded_condition_label_list.index(target)
    