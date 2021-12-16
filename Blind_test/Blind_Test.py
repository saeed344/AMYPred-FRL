#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Created on Mon Dec 16 14: 23:14 2019
@author
: logistics
"""
import os
for dirname, _, filenames in os.walk('/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import re, os, sys
from collections import Counter
import math
import numpy as np
import re
import pandas as pd
from sklearn.metrics import roc_auc_score


def read_protein_sequences(file):
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        #label = 'None' #header_array[1] if len(header_array) >= 1 else '0'
        #label_train = 'None' #header_array[2] if len(header_array) >= 2 else 'training'
        fasta_sequences.append([name, sequence])
    return fasta_sequences


def AAC(fastas, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = []
    for i in AA:
        header.append(i)
    #encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        code = []
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return np.array(encodings, dtype=float), header


def AAINDEX(fastas, props=None, **kw):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    fileAAindex = 'input/data/AAindex.txt'
    with open(fileAAindex) as f:
        records = f.readlines()[1:]

    AAindex = []
    AAindexName = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
        AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    #  use the user inputed properties
    if props:
        tmpIndexNames = []
        tmpIndex = []
        for p in props:
            if AAindexName.index(p) != -1:
                tmpIndexNames.append(p)
                tmpIndex.append(AAindex[AAindexName.index(p)])
        if len(tmpIndexNames) != 0:
            AAindexName = tmpIndexNames
            AAindex = tmpIndex

    header = []
    for idName in AAindexName:
        header.append(idName)

    encodings = []
    for i in fastas:
        name, sequence = i[0], i[1]
        code = []

        for j in AAindex:
            tmp = 0
            for aa in sequence:
                if aa == '-':
                    tmp = tmp + 0
                else:
                    tmp = tmp + float(j[index[aa]])
            code.append(tmp/len(sequence))
        encodings.append(code)
    return np.array(encodings, dtype=float), header


def APAAC(fastas, lambdaValue=10, w=0.05, **kw):
    dataFile = 'data/PAAC.txt'

    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records) - 1):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []
    header = []
    for i in AA:
        header.append('Pc1.' + i)
    for j in range(1, lambdaValue + 1):
        for i in AAPropertyNames:
            header.append('Pc2.' + i + '.' + str(j))

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        theta = []

        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)):
                theta.append(sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                                  range(len(sequence) - n)]) / (len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)

        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [w * value / (1 + w * sum(theta)) for value in theta]

        encodings.append(code)
    return np.array(encodings, dtype=float), header

# ASA required SPINEX external program
# BINARY encoding required same protein sequence length

def BLOSUM62(fastas, **kw):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    blosum62 = {
        'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        '-': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # -
    }
    encodings = []
    header = []
    for i in range(0,20):
        header.append('blosum62.F'+str(AA[i]))

    for i in fastas:
        name, sequence = i[0], i[1]
        code = np.asarray([0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])
        for aa in sequence:
            code = code + np.asarray(blosum62[aa])
        encodings.append(list(code/len(sequence)))
    return np.array(encodings, dtype=float), header

def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1 + '.' + key2] = 0
    return gPair


def CKSAAGP(fastas, gap=3, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1 + '.' + key2)

    encodings = []
    header = []
    for g in range(gap + 1):
        for p in gPairIndex:
            header.append(p + '.gap' + str(g))

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[index[sequence[p1]] + '.' + index[
                        sequence[p2]]] + 1
                    sum = sum + 1

            if sum == 0:
                for gp in gPairIndex:
                    code.append(0)
            else:
                for gp in gPairIndex:
                    code.append(gPair[gp] / sum)
        encodings.append(code)
    return np.array(encodings, dtype=float), header
def GDPC(fastas, **kw):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = group.keys()
    baseNum = len(groupKey)
    dipeptide = [g1 + '.' + g2 for g1 in groupKey for g2 in groupKey]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    encodings = []
    header = []#+dipeptide
    #encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])

        code = [name]
        myDict = {}
        for t in dipeptide:
            myDict[t] = 0

        sum = 0
        for j in range(len(sequence) - 2 + 1):
            myDict[index[sequence[j]] + '.' + index[sequence[j + 1]]] = myDict[index[sequence[j]] + '.' + index[
                sequence[j + 1]]] + 1
            sum = sum + 1

        if sum == 0:
            for t in dipeptide:
                code.append(0)
        else:
            for t in dipeptide:
                code.append(myDict[t] / sum)
        encodings.append(code)

    return np.array(encodings, dtype=float), header
def GAAC(fastas, **kw):
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }

    groupKey = group.keys()

    encodings = []
    header = []
    for key in groupKey:
        header.append(key)


    for i in fastas:
        name, sequence= i[0], re.sub('-', '', i[1])
        code = [name]
        count = Counter(sequence)
        myDict = {}
        for key in groupKey:
            for aa in group[key]:
                myDict[key] = myDict.get(key, 0) + count[aa]

        for key in groupKey:
            code.append(myDict[key]/len(sequence))
        encodings.append(code)

    return np.array(encodings, dtype=float), header

def CKSAAP(fastas, gap=3, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    header = []
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))

    for i in fastas:
        name, sequence = i[0], i[1]
        code = []
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def Count(seq1, seq2):
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum

def AACPCP(fastas, **kw):
    groups = {
        'charged':'DEKHR',
        'aliphatic':'ILV',
        'aromatic':'FHWY',
        'polar':'DERKQN',
        'neutral':'AGHPSTY',
        'hydrophobic':'CVLIMFW',
        'positively-charged':'HKR',
        'negatively-charged':'DE',
        'tiny':'ACDGST',
        'small':'EHILKMNPQV',
        'large':'FRWY'
    }


    property = (
    'charged', 'aliphatic', 'aromatic', 'polar',
    'neutral', 'hydrophobic', 'positively-charged', 'negatively-charged',
    'tiny', 'small', 'large')

    encodings = []
    header = property

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        for p in property:
            c = Count(groups[p], sequence) / len(sequence)
            code = code + [c]
        encodings.append(code)
    return np.array(encodings, dtype=float), list(header)

def CTDC(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = []
    for p in property:
        for g in range(1, len(groups) + 1):
            header.append(p + '.G' + str(g))

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        for p in property:
            c1 = Count(group1[p], sequence) / len(sequence)
            c2 = Count(group2[p], sequence) / len(sequence)
            c3 = 1 - c1 - c2
            code = code + [c1, c2, c3]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def Count2(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >=1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code


def CTDD(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')


    encodings = []
    header = []
    for p in property:
        for g in ('1', '2', '3'):
            for d in ['0', '25', '50', '75', '100']:
                header.append(p + '.' + g + '.residue' + d)

    for i in fastas:
        name, sequence  = i[0], re.sub('-', '', i[1])
        code = []
        for p in property:
            code = code + Count2(group1[p], sequence) + Count2(group2[p], sequence) + Count2(group3[p], sequence)
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def CTDT(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = []
    for p in property:
        for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
            header.append(p + '.' + tr)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
        for p in property:
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                    c1221 = c1221 + 1
                    continue
                if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                    c1331 = c1331 + 1
                    continue
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                    c2332 = c2332 + 1
            code = code + [c1221/len(aaPair), c1331/len(aaPair), c2332/len(aaPair)]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                fea = AADict[sequence[i]] + '.' + AADict[sequence[i + g + 1]] + '.' + AADict[
                    sequence[i + 2 * g + 2]]
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res
def KSCTriad(fastas, gap=0, **kw):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    encodings = []
    header = ['#']
    for g in range(gap + 1):
        for f in features:
            header.append(f + '.gap' + str(g))


    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        if len(sequence) < 2 * gap + 3:
            print('Error: for "KSCTriad" encoding, the input fasta sequences should be greater than (2*gap+3). \n\n')
            return 0
        code = code + CalculateKSCTriad(sequence, gap, features, AADict)
        encodings.append(code)

    return np.array(encodings, dtype=float), header

def CTriad(fastas, gap = 0, **kw):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.'+ f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    encodings = []
    header = []
    for f in features:
        header.append(f)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        if len(sequence) < 3:
            print('Error: for "CTriad" encoding, the input fasta sequences should be greater than 3. \n\n')
            return 0
        code = code + CalculateKSCTriad(sequence, 0, features, AADict)
        encodings.append(code)

    return np.array(encodings, dtype=float), header

def DDE(fastas, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'

    myCodons = {
        'A': 4,
        'C': 2,
        'D': 2,
        'E': 2,
        'F': 2,
        'G': 4,
        'H': 2,
        'I': 3,
        'K': 2,
        'L': 6,
        'M': 1,
        'N': 2,
        'P': 4,
        'Q': 2,
        'R': 6,
        'S': 6,
        'T': 4,
        'V': 4,
        'W': 1,
        'Y': 2
    }

    encodings = []
    diPeptides = ['DDE_'+aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = [] + diPeptides


    myTM = []
    for pair in diPeptides:
        myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]

        myTV = []
        for j in range(len(myTM)):
            myTV.append(myTM[j] * (1-myTM[j]) / (len(sequence) - 1))

        for j in range(len(tmpCode)):
            tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

        code = code + tmpCode
        encodings.append(code)
    return np.array(encodings, dtype=float), header

# Disorder (Disorder) Protein disorder information was first predicted using external VSL2
# DisorderB also required external program VSL2
# DisorderC also required external program VSL2

def DPC(fastas, gap, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = [] + diPeptides

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1 - gap):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+gap+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+gap+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def EAAC(fastas, window=5, **kw):
    AA ='ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = []
    for aa in AA:
        header.append('EACC.'+aa)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = []
        for aa in AA:
            tmp = 0
            for j in range(len(sequence)):
                if j < len(sequence) and j + window <= len(sequence):
                    count = Counter(sequence[j:j+window])
                    for key in count:
                        count[key] = count[key] / len(sequence[j:j+window])
                    tmp = tmp + count[aa]
            code.append(tmp/len(sequence))
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def EGAAC(fastas, window=5, **kw):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }
    groupKey = group.keys()
    encodings = []
    header = []
    for w in range(1, len(fastas[0][1]) - window + 2):
        for g in groupKey:
            header.append('SW.' + str(w) + '.' + g)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = []
        for key in groupKey:
            tmp=0
            for j in range(len(sequence)):
                if j + window <= len(sequence):
                    count = Counter(sequence[j:j + window])
                    myDict = {}
                    #for key in groupKey:
                    for aa in group[key]:
                        myDict[key] = myDict.get(key, 0) + count[aa]


                    #for key in groupKey:
                    tmp = tmp + (myDict[key] / window)
            code.append(tmp/len(sequence))
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)


def PAAC(fastas, lambdaValue=5, w=0.05, **kw):
    dataFile = 'data/PAAC.txt'

    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []
    header = []
    for aa in AA:
        header.append('Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))

    for i in fastas:
        name, sequence= i[0], re.sub('-', '', i[1])
        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
                    len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
    return np.array(encodings, dtype=float), header



# AAC, DPC, APAAC,PAAC,DDE,GAAC,KSCtraid, Ctraid, GDPC, CTDC, CTDD, CTDT,
header = []
fasta = read_protein_sequences('TR_P_132.fasta')
feat0, h = AAC(fasta)
header.append(h)
allfeat_pos = feat0
allfeat_head = header[0]
feat1, h = DPC(fasta,0)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat1),axis=1)
allfeat_head = allfeat_head + header[1]
feat2, h = APAAC(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat2),axis=1)
allfeat_head = allfeat_head + header[2]
feat3, h = CTDC(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat3),axis=1)
allfeat_head = allfeat_head + header[3]
feat4, h = CTDD(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat4),axis=1)
allfeat_head = allfeat_head + header[4]
feat5, h = CTDT(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat5),axis=1)
allfeat_head = allfeat_head + header[5]
#feat, h = GDPC(fasta)
#header.append(h)
#allfeat_pos = np.concatenate((allfeat_pos,feat),axis=1)
#allfeat_head = allfeat_head + header[6]
feat6, h = GAAC(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat6[:,1:]),axis=1)
allfeat_head = allfeat_head + header[6]
feat7, h = KSCTriad(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat7[:,1:]),axis=1)
allfeat_head = allfeat_head + header[7]
feat8, h = CTriad(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat8),axis=1)
allfeat_head = allfeat_head + header[8]
feat9, h = DDE(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat9),axis=1)
allfeat_head = allfeat_head + header[9]
feat10, h = PAAC(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat10),axis=1)
allfeat_head = allfeat_head + header[10]
CTD_featurP=np.concatenate((feat3,feat4,feat5),axis=1)

fasta = read_protein_sequences('TR_N_305.fasta')
feat0, headerx = AAC(fasta)
allfeat_neg = feat0
feat1, headerx = DPC(fasta,0)
allfeat_neg = np.concatenate((allfeat_neg,feat1),axis=1)
feat2, headerx = APAAC(fasta)
allfeat_neg = np.concatenate((allfeat_neg,feat2),axis=1)
feat3, headerx = CTDC(fasta)
allfeat_neg = np.concatenate((allfeat_neg,feat3),axis=1)
feat4, headerx = CTDD(fasta)
allfeat_neg = np.concatenate((allfeat_neg,feat4),axis=1)
feat5, headerx = CTDT(fasta)
allfeat_neg = np.concatenate((allfeat_neg,feat5),axis=1)
#feat, headerx =GDPC(fasta)
#allfeat_neg = np.concatenate((allfeat_neg,feat),axis=1)
feat6, headerx = GAAC(fasta)
allfeat_neg = np.concatenate((allfeat_neg,feat6[:,1:]),axis=1)
feat7, headerx = KSCTriad(fasta)
allfeat_neg = np.concatenate((allfeat_neg,feat7[:,1:]),axis=1)
feat8, headerx = CTriad(fasta)
allfeat_neg = np.concatenate((allfeat_neg,feat8),axis=1)
feat9, headerx = DDE(fasta)
allfeat_neg = np.concatenate((allfeat_neg,feat9),axis=1)
feat10, headerx = PAAC(fasta)
allfeat_neg = np.concatenate((allfeat_neg,feat10),axis=1)
f = []
before = 0
for i in range(0,10):
    after = before + len(header[i])
    f.append(list(range(before, after)))
    before = after
allfeat = np.concatenate((allfeat_pos, allfeat_neg), axis=0)
allclassT = np.concatenate((np.zeros(len(allfeat_pos)), np.ones(len(allfeat_neg))))
X = allfeat
y = allclassT
ix = []
for i in range(0, len(y)):
    ix.append(i)
ix = np.array(ix)
# Generate Test features
header = []
fasta = read_protein_sequences('blind_test_dataset.fasta')
feat0, h = AAC(fasta)
header.append(h)
allfeat_pos = feat0
allfeat_head = header[0]
feat1, h = DPC(fasta,0)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat1),axis=1)
allfeat_head = allfeat_head + header[1]
feat2, h = APAAC(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat2),axis=1)
allfeat_head = allfeat_head + header[2]
feat3, h = CTDC(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat3),axis=1)
allfeat_head = allfeat_head + header[3]
feat4, h = CTDD(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat4),axis=1)
allfeat_head = allfeat_head + header[4]
feat5, h = CTDT(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat5),axis=1)
allfeat_head = allfeat_head + header[5]
#feat, h = GDPC(fasta)
#header.append(h)
#allfeat_pos = np.concatenate((allfeat_pos,feat),axis=1)
#allfeat_head = allfeat_head + header[6]
feat6, h = GAAC(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat6[:,1:]),axis=1)
allfeat_head = allfeat_head + header[6]
feat7, h = KSCTriad(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat7[:,1:]),axis=1)
allfeat_head = allfeat_head + header[7]
feat8, h = CTriad(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat8),axis=1)
allfeat_head = allfeat_head + header[8]
feat9, h = DDE(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat9),axis=1)
allfeat_head = allfeat_head + header[9]
feat10, h = PAAC(fasta)
header.append(h)
allfeat_pos = np.concatenate((allfeat_pos,feat10),axis=1)
allfeat_head = allfeat_head + header[10]


allfeat = allfeat_pos


Xt = allfeat


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

featx = []

for i in range(0, 10):
    Xs = X[:, f[i]]
    Xts = Xt[:, f[i]]
    clf = RandomForestClassifier(n_estimators=500, random_state=0)
    clf.fit(Xs, y)
    pr = clf.predict_proba(Xts)[:, 0]
    pr_c = clf.predict(Xts)
    feat = pr
    feat = np.reshape(feat, (len(Xts), 1))
    if (i == 0):
        featx = feat
    else:
          featx = np.concatenate((featx, feat), axis=1)

    clf = ExtraTreesClassifier(n_estimators=500, random_state=0)
    clf.fit(Xs, y)
    pr = clf.predict_proba(Xts)[:, 0]
    pr_c = clf.predict(Xts)
    feat = pr
    feat = np.reshape(feat, (len(Xts), 1))
    featx = np.concatenate((featx, feat), axis=1)

    clf = SVC(probability=True, random_state=0)
    clf.fit(Xs, y)
    ppr = clf.predict_proba(Xts)[:, 0]
    pr_c = clf.predict(Xts)
    feat = pr
    feat = np.reshape(feat, (len(Xts), 1))
    featx = np.concatenate((featx, feat), axis=1)

    clf = LogisticRegression(random_state=0, max_iter=5000)
    clf.fit(Xs, y)
    pr = clf.predict_proba(Xts)[:, 0]
    pr_c = clf.predict(Xts)
    feat = pr
    feat = np.reshape(feat, (len(Xts), 1))
    featx = np.concatenate((featx, feat), axis=1)

    clf = XGBClassifier()
    clf.fit(Xs, y)
    pr = clf.predict_proba(Xts)[:, 0]
    pr_c = clf.predict(Xts)
    feat = pr
    feat = np.reshape(feat, (len(Xts), 1))
    featx = np.concatenate((featx, feat), axis=1)

    clf = KNeighborsClassifier(weights="distance", algorithm="auto")
    clf.fit(Xs, y)
    pr = clf.predict_proba(Xts)[:, 0]
    pr_c = clf.predict(Xts)
    feat= pr
    feat = np.reshape(feat, (len(Xts), 1))
    featx = np.concatenate((featx, feat), axis=1)

Test_PF=featx
import pandas as pd
import numpy as np
import pickle
mask=[2,3,4,5,9,10,12,14,15,18,24,25,28,34,46,47,48,53,56,57]
Selected_feat = featx[:,mask]

Xt=Selected_feat


xtest = np.vstack(Xt)

ldmodel = pickle.load(open("model/pima.pickle_model_svm_PF.dat", "rb"))
print("Loaded model from disk")
y_score = ldmodel.predict_proba(xtest)
prediction_results=pd.DataFrame(data=y_score) 
prediction_results.to_csv('prediction_results.csv')
