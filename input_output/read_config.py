from misc import read_input_csv as ricsv
from copy import deepcopy
from input_output.organize import Organize_input

def read_txt(init_file):
    """
    Read a PIPT input file (.pipt), parse and output dictionaries for data assimilation and simulator classes.

    Parameters
    ----------
    init_file: str
        PIPT init. file containing info. to run the inversion algorithm

    Returns
    -------
    keys_da : dict
        Parsed keywords from DATAASSIM
    keys_fwd : dict
        Parsed keywords from FWDSSIM
    """
    # Check for .pipt suffix
    if not init_file.endswith('.pipt'):
        raise FileNotFoundError(f'No PIPT input file (.pipt) found! If {init_file} is  a PIPT input file, change '
                                'suffix to .pipt')

    # Read the init file and output lines without comments (lines starting with '#')
    lines = read_clean_file(init_file)

    # Find where the separate parts are located in the file. FWDSIM will always be a part, but the
    # inversion/optimiztation part may be DATAASSIM or OPTIM
    for i in range(len(lines)):
        if lines[i].strip().lower() == 'dataassim' or lines[i].strip().lower() == 'optim':
            ipind = i
            ip_part = lines[i].strip().lower()
        elif lines[i].strip().lower() == 'fwdsim':
            fwdsimind = i

    # Split the file into the two separate parts. Each part will (only) contain the keywords of each part:
    if ipind < fwdsimind:  # Data assim. part is the first part of file
        lines_ip = lines[2:fwdsimind]
        lines_fwd = lines[fwdsimind + 2:]
    else:  # Fwd sim. part is the first part of file
        lines_fwd = lines[2:ipind]
        lines_ip = lines[ipind + 2:]

    # Get rid of empty lines in lines_ip and lines_fwd
    clean_lines_ip = remove_empty_lines(lines_ip)
    clean_lines_fwd = remove_empty_lines(lines_fwd)

    # Assign the keys and values to different dictionaries depending on whether we have data assimilation (DATAASSIM)
    # or optimization (OPTIM). FWDSIM info is always assigned to keys_fwd
    if ip_part == 'dataassim' or ip_part == 'optim':
        keys_da = parse_keywords(clean_lines_ip)
    keys_fwd = parse_keywords(clean_lines_fwd)

    check_mand_keywords_da_fwdsim(keys_da, keys_fwd)
    org = Organize_input(keys_da,keys_fwd)
    org.organize()

    return org.get_keys_da(), org.get_keys_fwd()

def read_clean_file(init_file):
    """
    Read PIPT init. file and lines that are not comments (marked with octothorpe)

    Parameters
    ----------
    init_file: str
        Name of file to remove all comments. WHOLE filename needed (with suffix!)

    Returns
    -------
    lines: list
        Lines from init. file converted to list entries
    """
    # Read file except lines starting with an octothorpe (#) and return the python variable
    with open(init_file, 'r') as f:
        lines = [line for line in f.readlines() if not line.startswith('#')]

    # Return clean lines
    return lines


def remove_empty_lines(lines):
    """
    Small method for finding empty lines in a read file.

    Parameters
    ----------
    lines: list
        List of lines from a file

    Returns
    -------
    lines_clean: list
        List of clean lines (without empty entries)
    """
    # Loop over lines to find '\n'
    sep = []
    for i in range(len(lines)):
        if lines[i] == '\n':
            sep.append(i)

    # Make clean output
    lines_clean = []
    for i in range(len(sep)):
        if i == 0:
            lines_clean.append(lines[0:sep[i]])
        else:
            lines_clean.append(lines[sep[i-1] + 1:sep[i]])

    # Return
    return lines_clean


def parse_keywords(lines):
    """
    Here we parse the lines in the init. file to a Python dictionary. The keys of the dictionary is the keywords
    entered in the PIPT init. file, and the information entered in each keyword is stored in each key of the
    dictionary. To know how the keyword-information is organized in the keys of the dictionary, confront the
    manual located in the doc folder.

    Parameters
    ----------
    lines: list
        List of (clean) lines from the PIPT init. file.

    Returns
    -------
    keys: dict
        Dictionary with all info. from the init. file.
    """
    # Init. the dictionary
    keys = {}

    # Loop over all input keywords and store in the dictionary.
    for i in range(len(lines)):
        if lines[i] != []:  # Check for empty list (corresponds to empty line in file)
            try:  # Try first to store the info. in keyword as float in a 1D list
                # A scalar, which we store as scalar...
                if len(lines[i][1:]) == 1 and len(lines[i][1:][0].split()) == 1:
                    keys[lines[i][0].strip().lower()] = float(lines[i][1:][0])
                else:
                    keys[lines[i][0].strip().lower()] = [float(x) for x in lines[i][1:]]
            except:
                try:  # Store as float in 2D list
                    if len(lines[i][1:]) == 1:  # Check if it is actually a 1D array disguised as 2D
                        keys[lines[i][0].strip().lower()] = \
                            [float(x) for x in lines[i][1:][0].split()]
                    else:  # if not store as 2D list
                        keys[lines[i][0].strip().lower()] = \
                            [[float(x) for x in col.split()] for col in lines[i][1:]]
                except:  # Keyword contains string(s), not floats
                    if len(lines[i][1:]) == 1:  # If 1D list
                        if len(lines[i][1:][0].split('\t')) == 1:  # If it is a scalar store as single input
                            keys[lines[i][0].strip().lower()] = lines[i][1:][0].strip().lower()
                        else:  # Store as 1D list
                            keys[lines[i][0].strip().lower()] = \
                                [x.rstrip('\n').lower() for x in lines[i][1:][0].split('\t') if x != '']
                    else:  # It is a 2D list
                        # Check each row in 2D list. If it is single column (i.e., one string per row),
                        # we make it a 1D list of strings; if not, we make it a 2D list of strings.
                        one_col = True
                        for j in range(len(lines[i][1:])):
                            if len(lines[i][1:][j].split('\t')) > 1:
                                one_col = False
                                break
                        if one_col is True:  # Only one column
                            keys[lines[i][0].strip().lower()] = \
                                [x.rstrip('\n').lower() for x in lines[i][1:]]
                        else:  # Store as 2D list
                            keys[lines[i][0].strip().lower()] = \
                                [[x.rstrip('\n').lower() for x in col.split('\t') if x != '']
                                    for col in lines[i][1:]]

    # Need to check if there are any only-string-keywords that actually contains floats, and convert those to
    # floats (the above loop only handles pure float or pure string input, hence we do a quick fix for mixed
    # lists here)
    # Loop over all keys in dict. and check every "pure" string keys for floats
    for i in keys:
        if isinstance(keys[i], list):  # Check if key is a list
            if isinstance(keys[i][0], list):  # Check if it is a 2D list
                for j in range(len(keys[i])):  # Loop over all sublists
                    if all(isinstance(x, str) for x in keys[i][j]):  # Check sublist for strings
                        for k in range(len(keys[i][j])):  # Loop over enteries in sublist
                            try:  # Try to make float
                                keys[i][j][k] = float(keys[i][j][k])  # Scalar
                            except:
                                try:  # 1D array
                                    keys[i][j][k] = [float(x) for x in keys[i][j][k].split()]
                                except:  # If it is actually a string, pass over
                                    pass
            else:  # It is a 1D list
                if all(isinstance(x, str) for x in keys[i]):  # Check if list only contains strings
                    for j in range(len(keys[i])):  # Loop over all entries in list
                        try:  # Try to make float
                            keys[i][j] = float(keys[i][j])
                        except:
                            try:
                                keys[i][j] = [float(x) for x in keys[i][j].split()]
                            except:  # If it is actually a string, pass over
                                pass

    # Return dict.
    return keys

def check_mand_keywords_da_fwdsim(keys_da, keys_fwd):
    """Check for mandatory keywords in `DATAASSIM` and `FWDSIM` part, and output error if they are not present"""
    # Mandatory keywords in DATAASSIM
    assert 'ne' in keys_da, 'NE not in DATAASSIM!'
    assert 'truedataindex' in keys_da, 'TRUEDATAINDEX not in DATAASSIM!'
    assert 'assimindex' in keys_da, 'ASSIMINDEX not in DATAASSIM!'
    assert 'truedata' in keys_da, 'TRUEDATA not in DATAASSIM!'
    assert 'staticvar' in keys_da, 'STATICVAR not in DATAASSIM!'
    assert 'datavar' in keys_da, 'DATAVAR not in DATAASSIM!'
    assert 'obsname' in keys_da, 'OBSNAME not in DATAASSIM!'
    if 'importstaticvar' not in keys_da:
        assert filter(list(keys_da.keys()), 'prior_*') != [], 'No PRIOR_<STATICVAR> in DATAASSIM'
    assert 'energy' in keys_da, 'ENERGY not in DATAASSIM!'

    # Mandatory keywords in FWDSIM
    assert 'simulator' in keys_fwd, 'SIMULATOR not in FWDSIM!'
    assert 'parallel' in keys_fwd, 'PARALLEL not in FWDSIM!'
    assert 'datatype' in keys_fwd, 'DATATYPE not in FWDSIM!'