#!/usr/bin/env python
# encoding: utf-8

# Leverages the output of Bismark to link SNP alleles
# with the methylation state of CpG sites

import pandas as pd
import pysam

from collections import defaultdict as ddict
import sys, os, glob

def _find_file(pattern, description, required=True):
    matches = sorted(glob.glob(pattern))
    if not matches:
        if required:
            sys.exit(f'ERROR: No {description} found matching {pattern}. Exiting')
        return None
    if len(matches) > 1:
        print(f'WARNING: {len(matches)} {description} files match {pattern}; '
              f'using {matches[0]}')
    return matches[0]

def read_meth(fn, is_ob=False):
    if fn is None:
        return pd.DataFrame(
                columns=['read_id', 'meth_pos', 'meth_state'])
    aux = pd.read_csv(fn, skiprows=1, sep='\t',
        names=['read_id', 'meth_state', 'ref',
            'meth_pos','meth_char'])[[
            'read_id', 'meth_pos', 'meth_state'
            ]]
    if is_ob:
        aux['meth_pos']=aux['meth_pos']-1
    return aux

def print_log(msg, verbose=True):
    if verbose:
        print(msg)

def filter_snps(snps, freq='ref', expected=True, verbose=True):
    pos = []
    for x in snps.fetch():
        alt = x.alts[0]
        if not (len(x.ref) < 2) or not (len(alt) < 2):
            print_log(
                f'> Position {x.pos} discarded - INDEL variant',
                verbose)
            continue
        if not expected:
            pos_dict = {'pos' : x.pos, 'ref' : x.ref, 'alt' : alt}
        else:
            dp4 = x.info['DP4']
            dp = x.info['DP']
            if freq == 'ref':
                ref_prop = (dp4[0] + dp4[1])/dp
                pos_dict = {'pos' : x.pos, 'ref' : x.ref,
                        'alt' : alt, 'ref_prop_exp' : ref_prop}
            else:
                try:
                    alt_prop = x.info['AF'][0]
                except:
                    try:
                        alt_prop = (dp4[2] + dp4[3])/dp
                    except:
                        alt_prop = 0
                pos_dict = {'pos' : x.pos, 'ref' : x.ref,
                        'alt' : alt, 'alt_prop_exp' : alt_prop}
        pos.append(pos_dict)
    return pos

def parse_vars_csv(snps_fn, freq='ref', expected=True, verbose=True):
    print_log(
            '>Reading variants input table', verbose)
    variants = pd.read_csv(snps_fn)
    if not {'pos', 'ref', 'alt'}.issubset(variants.columns):
        sys.exit((
            'ERROR: At least one required column (pos, ref, alt) '
            'not present in variants input table. Exiting'))
    if expected:
        if not {'af'}.issubset(variants.columns):
            sys.exit((
                'ERROR: No column "af" present in variants input table. '
                'Use -x to avoid requiring expected allele frequencies '
                'or include the allele frequency information in the table. '
                'Exiting.'))
    print_log(
            '>Variants input table read correctly',
            verbose)
    variants = _add_freq_table(_filter_vars_table(
            variants, verbose=verbose),
            freq=freq, expected=expected, verbose=verbose)
    return variants

def _filter_vars_table(variants, verbose=True):
    print_log(
            '>Filtering input variants', verbose)
    variants = variants.drop_duplicates(['pos', 'ref', 'alt'])
    print_log(
            f'>{variants.shape[0]} distinct variants in input table',
            verbose)

    print_log('>Ignoring INDELs', verbose)
    keep = (variants['ref'].str.len() == 1) & (variants['alt'].str.len() == 1)
    filtered_variants = variants[keep].copy()

    if variants.shape[0] == filtered_variants.shape[0]:
        print_log('>All variants retained after filters', verbose)
    else:
        print_log(
            f'>{filtered_variants.shape[0]} variants retained after filters',
            verbose)
    return filtered_variants

def _add_freq_table(variants, freq='ref', expected=True, verbose=True):
    if not expected:
        return variants[['pos', 'ref', 'alt']]

    variants = variants.copy()
    variants['af'] = pd.to_numeric(variants['af'], errors='coerce')
    variants = variants.dropna(subset=['af'])

    col = freq + '_prop_exp'
    if freq == 'alt':
        variants = variants.rename(columns={'af' : col})
    else:
        total_alt = variants.groupby('pos')['af'].transform('sum')
        if (total_alt > 1.0001).any():
            print_log(
                    'Warning: alt frequencies sum to >1 at some positions',
                    verbose = verbose)
        variants[col] = (1 - total_alt).clip(lower=0)
        variants = variants.sort_values('af', ascending=False, kind='stable')

    return variants[['pos', 'ref', 'alt', col]]

def extract_bases(alignments, positions, ref='BK000964.3_looped_3008'):
    bases_reads = []
    for pos in positions:
        for pcol in alignments.pileup(ref, pos-1, pos):
            if pcol.pos != (pos-1):
                continue
            for pread in pcol.pileups:
                if not pread.is_del and not pread.is_refskip:
                    bases_reads.append({
                        'read_id' : pread.alignment.query_name,
                        'snp_pos' : pos,
                        'snp_allele' : pread.alignment.query_sequence[
                            pread.query_position]})
    return bases_reads

def select_reads(snp_calls, meth_calls):
    return snp_calls[snp_calls.read_id.isin(
        meth_calls.read_id.values)].drop_duplicates()

def _calculate_coverage(counts):
    return counts.sum(axis=1)

def _find_covered_positions(counts, min_cov=10):
    cov = _calculate_coverage(counts)
    return counts.loc[cov >= min_cov].index.values

def keep_covered_positions(ot_counts, ob_counts, min_cov=10, informed=False):
    ot_cov = _find_covered_positions(ot_counts, min_cov=min_cov)
    ob_cov = _find_covered_positions(ob_counts, min_cov=min_cov)
    if informed:
        return (ot_counts[ot_counts.index.isin(ot_cov)],
                ob_counts[ob_counts.index.isin(ob_cov)])
    covered = list(set(ot_cov) & set(ob_cov))
    return (ot_counts[ot_counts.index.isin(covered)],
            ob_counts[ob_counts.index.isin(covered)])

def _get_alleles_pos(base_counts, maf_th=0.1):
    return [(allele, base_counts[allele])
            for allele in base_counts.index
            if base_counts[allele] >= maf_th*sum(base_counts)]

def _get_alleles(counts, maf_th=0.1):
    alleles = { pos : _get_alleles_pos(base_counts, maf_th=maf_th)
            for pos, base_counts in counts.iterrows()}
    return alleles

def _ot_alleles(counts, maf_th=0.1):
    joint_counts = counts.assign(
        C_T = lambda df: df['C'] + df['T'])[['A', 'G', 'C_T']]
    return _get_alleles(joint_counts, maf_th=maf_th)

def _ob_alleles(counts, maf_th=0.1):
    joint_counts = counts.assign(
        A_G = lambda df: df['A'] + df['G'])[['A_G', 'C', 'T']]
    return _get_alleles(joint_counts, maf_th=maf_th)

def _join_pos(ot_alleles, ob_alleles, multiallelic=False):
    if not multiallelic:
        ot_alleles_d=dict(ot_alleles)
        ob_alleles_d=dict(ob_alleles)
        als = {k : ot_alleles_d.get(k, 0) + ob_alleles_d.get(k, 0)
                for k in set(ot_alleles_d) | set(ob_alleles_d)
                if k != 'C_T' and k != 'A_G'}
        return sorted(als, key=als.get, reverse=True)[:2]
    else:
        return list(set(
            [al[0] for al in ot_alleles
                if al[0] != 'C_T'] +
            [al[0] for al in ob_alleles
                if al[0] != 'A_G']))

def _join_alleles(ot_alleles, ob_alleles, multiallelic=False):
    return { pos :_join_pos(ot_alleles[pos], ob_alleles[pos], multiallelic=multiallelic)
                for pos in ot_alleles
                if (pos in ob_alleles.keys())}

def infer_alleles(ot_counts, ob_counts, maf_th=0.1, multiallelic=False):
    ot_alleles = _ot_alleles(ot_counts, maf_th=maf_th)
    ob_alleles = _ob_alleles(ob_counts, maf_th=maf_th)
    return _join_alleles(ot_alleles, ob_alleles, multiallelic=multiallelic)

def _correct_reads(reads, alleles, strand='ot'):
    aux = reads.copy()
    for i, r in reads.iterrows():
        if (r['snp_allele'] in alleles[r['snp_pos']]):
            continue
        elif ((strand == 'ot') and
                (r['snp_allele'] == 'T') and
                ('C' in alleles[r['snp_pos']])):
            aux.at[i, 'snp_allele'] = 'C'
        elif ((strand == 'ob') and
                (r['snp_allele'] == 'A') and
                ('G' in alleles[r['snp_pos']])):
            aux.at[i, 'snp_allele'] = 'G'
        else:
            aux.drop(i, inplace=True)
    return aux

def correct_reads(reads, alleles, strand='ot'):
    positions = list(alleles.keys())
    conf_bases = ['C', 'T'] if strand == 'ot' else ['A', 'G']
    conf_pos = [pos
            for pos, bases in alleles.items()
            if ((conf_bases[0] in bases) and
                (conf_bases[1] in bases))]
    positions = sorted(list(
        set(list(alleles.keys())) - set(conf_pos)))
    reads_filt = reads[reads['snp_pos'].isin(positions)]
    return _correct_reads(reads_filt, alleles, strand)

def select_ref(x):
    return x[x['ref']]

def select_alt(x):
    return x[x['alt']]

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='Variant-Specific Methylation.')
    parser.add_argument('sample', metavar='S', type=str,
                               help='root name of sample to process')
    parser.add_argument('-r', '--region', type=str, default='BK000964.3_looped_3008',
                               help='contig name')
    parser.add_argument('-o', '--output', type=str,
                               help='path to output folder')
    parser.add_argument('-b', '--bme', type=str,
                               help='path to Bismark Methylation Extractor folder')
    parser.add_argument('-a', '--align', type=str,
                               help='path to Bismark alignments folder')
    parser.add_argument('-s', '--snps', type=str,
                               help='path to SNPs folder (assuming it contains a VCF file) or specific file (in VCF or CSV format)')
    parser.add_argument('-c', '--coverage', type=int,
                               help='minimum coverage')
    parser.add_argument('-t', '--threshold', type=float,
                               help='minimum allele frequency threshold')
    parser.add_argument('-f', '--freq', type=str, default='ref', choices=['alt', 'ref'],
                               help='allele on which to report frequency')
    parser.add_argument('-x', '--ignore_expected', action='store_true',
                               help='bypass retrieval/calculation of allele frequencies from input VCF/CSV')
    parser.add_argument('-i', '--informed', action='store_true',
                               help='allow Blink to obtain alleles to seek in methylation data from input VCF/CSV')
    parser.add_argument('-p', '--pcr', action='store_true',
                               help='enable Blink to work with amplicons of only one directionality')
    parser.add_argument('-m', '--multiallelic', action='store_true',
                               help='permit more than two alleles per position')
    parser.add_argument('-n', '--name_reads', action='store_true',
                               help='change read names to reduce output size')
    parser.add_argument('-v', '--verbose', action='store_true',
                               help='increase output verbosity')

    return parser.parse_args()

def main():

# 00.- Configuration

    args = _parse_args()
    root_fn = args.sample
    region = args.region

    print(f"Starting blink execution for sample {root_fn}")

## Options

    min_cov = args.coverage
    maf_th = args.threshold
    verbose = args.verbose
    informed = args.informed
    pcr = args.pcr
    multiallelic = args.multiallelic
    freq = args.freq
    name_reads = args.name_reads
    expected = not args.ignore_expected

## Paths

    ofd = f"{args.output}/{root_fn}/"

    if not os.path.exists(ofd):
            os.makedirs(ofd)

    bme_fd = f"{args.bme}/{root_fn}/"

    ot_fn = _find_file(f"{bme_fd}CpG_OT_*.txt", 'OT methylation file', required=False)
    ob_fn = _find_file(f"{bme_fd}CpG_OB_*.txt", 'OB methylation file', required=False)

    if ot_fn is None and ob_fn is None:
        sys.exit('ERROR: No OT or OB file present. Exiting')
    elif (ot_fn is None or ob_fn is None) and not pcr:
        sys.exit('ERROR: Both OT and OB files must be present. Exiting')

    als_fd = f"{args.align}/{root_fn}/"
    als_fn = _find_file(f"{als_fd}*.bam", 'BAM file')

    if os.path.isdir(args.snps):
        snps_fd = f"{args.snps}/{root_fn}/"
        snps_fn = _find_file(f"{snps_fd}*.vcf", 'VCF file')
    else:
        snps_fn = args.snps

    if snps_fn.endswith('.vcf'):
        snps_ft = 'vcf'
    elif snps_fn.endswith('.csv'):
        snps_ft = 'csv'
    else:
        sys.exit('ERROR: SNPs file must be in VCF or CSV format. Exiting')

# 01.- Read Bismark reports

    ot = read_meth(ot_fn)
    ob = read_meth(ob_fn, is_ob=True)

# 02.- Read SNPs

    if snps_ft == 'vcf':
        snps = pd.DataFrame(filter_snps(pysam.VariantFile(snps_fn, 'r'),
            freq=freq, expected=expected, verbose=verbose))
    else:
        snps = parse_vars_csv(snps_fn, freq=freq, expected=expected, verbose=verbose)

# 03.- Read alignment files

    als = pysam.AlignmentFile(als_fn, 'rb')

# 04.- Extract bases from reads

    calls = pd.DataFrame(
            extract_bases(als, snps['pos'], ref=region)).sort_values(
            by=['read_id', 'snp_pos'])
    print_log(calls, verbose=verbose)

# 05.- Separate base calls per original strand

    ot_reads = select_reads(calls, ot).reset_index(drop=True)
    print_log("", verbose=verbose)
    print_log("OT Reads", verbose=verbose)
    print_log(ot_reads, verbose=verbose)

    ob_reads = select_reads(calls, ob).reset_index(drop=True)
    print_log("", verbose=verbose)
    print_log("OB Reads", verbose=verbose)
    print_log(ob_reads, verbose=verbose)


# 06.- Obtain base counts from each strand

    ot_counts = pd.crosstab(
        ot_reads.snp_pos,
        ot_reads.snp_allele)

    print_log("", verbose=verbose)
    print_log("OT counts before filtering", verbose=verbose)
    print_log(ot_counts, verbose=verbose)

    ob_counts = pd.crosstab(
        ob_reads.snp_pos,
        ob_reads.snp_allele)

    print_log("", verbose=verbose)
    print_log("OB counts before filtering", verbose=verbose)
    print_log(ob_counts, verbose=verbose)

# 07.- Remove insufficiently-covered positions

    ot_counts, ob_counts = keep_covered_positions(
            ot_counts, ob_counts, min_cov=min_cov, informed=informed)

    if informed:
        positions = list(set(ot_counts.index.values) | set(ob_counts.index.values))
        if not multiallelic:
            if not expected:
                snps = snps.drop_duplicates('pos', keep='first')
            else:
                col = freq + '_prop_exp'
                pick = 'idxmin' if freq == 'ref' else 'idxmax'
                snps = snps.loc[snps.groupby('pos')[col].agg(pick)]
            alleles = { row['pos'] : [row['ref'], row['alt']]
                    for index, row in snps.iterrows()
                    if row['pos'] in positions }
        else:
            alleles = {
                pos : list(set(list(snps[snps['pos']==pos].ref) + list(snps[snps['pos']==pos].alt)))
                for pos in positions }
    else:
        print_log("", verbose=verbose)
        print_log("OT counts after filtering", verbose=verbose)
        print_log(ot_counts, verbose=verbose)

        print_log("", verbose=verbose)
        print_log("OB counts after filtering", verbose=verbose)
        print_log(ob_counts, verbose=verbose)

# 08.- Infer alleles for each position

        print_log("", verbose=verbose)
        print_log("OT alleles", verbose=verbose)
        print_log(_ot_alleles(ot_counts, maf_th=maf_th), verbose=verbose)

        print_log("", verbose=verbose)
        print_log("OB alleles", verbose=verbose)
        print_log(_ob_alleles(ob_counts, maf_th=maf_th), verbose=verbose)
        alleles = infer_alleles(ot_counts, ob_counts, maf_th=maf_th, multiallelic=multiallelic)

    print_log("", verbose=verbose)
    print_log("Joined", verbose=verbose)

    for k,v in alleles.items():
        print_log(f'Position {k}: {v}', verbose=True)

# 09.- Correct reads

    reads_corr = pd.concat([
            correct_reads(ot_reads, alleles, strand='ot'),
            correct_reads(ob_reads, alleles, strand='ob')])

    print_log(reads_corr, verbose=verbose)

    bc = pd.crosstab(
        reads_corr.snp_pos,
        reads_corr.snp_allele)

    bc = bc.join(snps.set_index('pos'))
    if 'A' not in bc:
            bc['A'] = 0
    if 'C' not in bc:
            bc['C'] = 0
    if 'G' not in bc:
            bc['G'] = 0
    if 'T' not in bc:
            bc['T'] = 0
    bc.index = bc.index.set_names(['snp_pos'])

    if freq == 'ref':
        bc['ref_obs'] = bc.apply(select_ref, axis=1)
        bc['ref_prop_obs'] = bc['ref_obs'] / (bc['A'] + bc['C'] + bc['G'] + bc['T'])
    else:
        bc['alt_obs'] = bc.apply(select_alt, axis=1)
        bc['alt_prop_obs'] = bc['alt_obs'] / (bc['A'] + bc['C'] + bc['G'] + bc['T'])

    bc = bc.reset_index()
    print_log(bc, verbose=verbose)

    bc.to_csv(f"{ofd}{root_fn}_snps.csv",
            index=False)

# 10.- Link Methylation Data

    meth = pd.concat([ot, ob])
    link = reads_corr.set_index('read_id').join(
        meth.set_index('read_id')).reset_index()

    if name_reads:
        names = pd.DataFrame(
                data={ 'read_id' : link['read_id'].unique()}
                ).assign(read_name = lambda x : x.index.map(
                    lambda ix : f"{root_fn}.{ix}"))
        link = link.merge(names).drop('read_id', axis=1)
        col = link.pop('read_name')
        link.insert(0, col.name, col)
        names.to_csv(f"{ofd}{root_fn}_reads.csv", index=False)

    print_log(link, verbose=verbose)
    link.to_csv(f"{ofd}{root_fn}_link.csv", index=False)

if __name__ == "__main__":
    main()
