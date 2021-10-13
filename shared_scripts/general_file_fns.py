# March 22nd 2019
# A few general functions for loading and saving files

import pickle
import glob
import os

def load_file_from_pattern(file_pattern):
    file_matches = glob.glob(file_pattern)
    if len(file_matches)>1:
        print('Multiple matches. Using the first one')
    if len(file_matches)==0:
        print('No file found')
        return
    fname = file_matches[0]
    data = load_pickle_file(fname)
    return data, fname

def load_pickle_file(filename):
    fr = open(filename, 'r')
    data = pickle.load(fr)
    fr.close()
    return data

def save_pickle_file(data, filename):
    fw = open(filename, 'w')
    pickle.dump(data, fw)
    fw.close()
    return 1

def return_dir(input_dir):
    '''Makes the directory input_dir if it doesn't exist.
    Return input_dir.'''
    if not os.path.exists(input_dir):
        print('Making %s'%input_dir)
        os.makedirs(input_dir)
    return input_dir

def read_numerical_file(path, data_type, list_type):
    '''
    Reads in a file consisting of UTF-8 encoded lists of numbers with single or 
    multiple observations per line.

    Parameters
    ----------
    path: str or Path object
        file to be read
    data_type: int or float
        data type of the observations in the file
    list_type: str
        'single'
            single observations per line
        'multiple'
            multiple observations per line
    
    Returns
    -------
    data_list: list
        Simple list of single values, or if 'multiple' data type then nested lists for each
        line in input file
    '''
    if data_type not in ('float', 'int'):
        raise ValueError('Must specify either \'float\' or \'int\' as data_type')
    if list_type not in ('single', 'multiple'):
        raise ValueError('list type must be \'single\' or \'multiple\'')
    fr = open(path, 'r')
    if data_type == 'int':
        d_type = int
    elif data_type == 'float':
        d_type = float
    
    if list_type == 'single':
        data_list = [d_type(line.rstrip()) for line in fr]
    elif list_type == 'multiple':
        data_list = [[d_type(y) for y in line.split()] for line in fr]
    fr.close()
    return data_list

# def old_read_float_file(file_name, list_type='multiple'):
#     ''' Read in a file of float numbers. If list_type is single then we just read everything into the same 
#     list (e.g. if there's only 1 number a line). Otherwise separate sub-lists for each line.'''

#     with open(file_name, 'r') as f:
#         if list_type == 'single':
#             ret_list = [float(x) for line in f for x in line.split()]
#         elif list_type == 'multiple':
#             ret_list = [[float(x) for x in line.split()] for line in f]
#         else:
#             print 'Unknown list type. Should be single or multiple'
#     # print f.closed
#     return ret_list


# def old_read_int_file(file_name, list_type='multiple'):
#     ''' Read in a file of int numbers. If list_type is single then we just read everything into the same 
#     list (e.g. if there's only 1 number a line). Otherwise separate sub-lists for each line.
#     Maybe merge with the above eventually.
#     Also should test a bit'''

#     with open(file_name, 'r') as f:
#         if list_type == 'single':
#             ret_list = [int(x) for line in f for x in line.split()]
#         elif list_type == 'multiple':
#             ret_list = [[int(x) for x in line.split()] for line in f]
#         else:
#             print 'Unknown list type. Should be single or multiple'
#     # print f.closed
#     return ret_list
