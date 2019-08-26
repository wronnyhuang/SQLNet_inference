import os
import json
from os.path import join, basename
from nltk.tokenize.stanford import StanfordTokenizer
from time import time

home = os.environ['HOME']
os.environ['CLASSPATH'] = join(home,'datasets/stanford-postagger-2018-10-16/')

def parse_table(lines, stanford):
    '''parses the table data'''
    
    entry = dict(page_title='mock_time_machine', name='mock_time_machine', caption='R', section_title='R', id='mock_time_machine')
    
    def parse_row(row):
        row = row.strip('\ufeff')
        row = row.strip('\n')
        row = row.split(',')
        row = [col.strip() for col in row]
        return row

    # parse column names
    cols = parse_row(lines[0])
    entry['header'] = cols
    entry['header_tok'] = [stanford.tokenize(col) for col in cols]
    
    # parse rows and also get schema type
    entry['rows'] = []
    for i, line in enumerate(lines[1:]):
        row = parse_row(line)
        if i == 0: entry['types'] = ['real' for _ in row]
        entry['rows'].append(row)
        for j, r in enumerate(row):
            try: float(r)
            except: entry['types'][j] = 'text'
    
    return entry

def parse_sql(line, cols):
    '''parses the sql query into wikisql-required json format'''
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['EQL', 'GT', 'LT', 'OP']
    cond_ops_alt = ['=', '>', '<', 'OP']
    syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG', 'AGGOPS', 'CONDOPS']
    sql = dict(conds=[])
    
    ## ==> select clause
    # first word better be the seelct symbol
    assert line.find('SELECT') == 0
    
    # bound the select value
    findid = line.find('FROM')
    sel = line[6:findid]
   
    # parse aggregator
    sql['agg'] = 0
    for aggid, aggop in enumerate(agg_ops[1:]):
        if aggop in sel:
            sql['agg'] = aggid + 1
            sel = sel.replace(aggop, '')
            break
    
    # remove remaining parens
    sel = sel.replace('(', '')
    sel = sel.replace(')', '')
    
    # remove whitespace
    sel = sel.strip()
    
    # now the cleaned up select value better be a column name of the table
    assert sel in cols
    sql['sel'] = cols.index(sel)
    
    ## ==> where clause
    
    def parse_where(where):
        
        for alt, orig in zip(cond_ops_alt, cond_ops):
            where = where.replace(alt, orig)
        
        cond = [None for _ in range(3)]
        
        # see which operation it is and where its located
        for condid, condop in enumerate(cond_ops):
            if condop in where:
                cond[1] = condid
                break
                
        assert cond[1] is not None
    
        # split where clause into left (column) and right (value)
        condidx = where.find(condop)
    
        # parse left
        left = where[:condidx]
        col = left.strip()
        assert col in cols
        cond[0] = cols.index(col)
    
        # parse right
        right = where[condidx+len(condop):]
        val = right.strip()
        cond[2] = val
        
        return cond

    # find start of where clauses
    whereid = line.find('WHERE')
    wheres = line[whereid+5:]

    while wheres.find('AND') != -1:
        # find next AND symbol
        andid = wheres.find('AND')
        # get next where clause
        where = wheres[:andid]
        # parse current where clause
        sql['conds'].append(parse_where(where))
        # striped processed where clause out of wheres
        wheres = wheres[andid+3:]

    # parse last where clause
    sql['conds'].append(parse_where(wheres))
    
    return sql

if __name__ == '__main__':
    
    stanford = StanfordTokenizer()
    
    with open('pairs_ronny.csv', 'r+') as f:
        lines_pairs = f.readlines()[1:]
    with open('table_ronny.csv', 'r') as f:
        lines_table = f.readlines()
        
    # parse table
    entry_table = parse_table(lines_table, stanford)
    with open('dummy_tok.tables.jsonl', 'w') as f:
        json.dump(entry_table, f)
    
    # parse all pairs
    f = open('dummy_tok.jsonl', 'w+')
    for line in lines_pairs:
        entry = dict(phase=2)
        # line.replace('"', '')
        # line = line.encode()
        line = line.split(',')
        sqlquery = line[0]
        question = line[1]
        entry['query'] = sqlquery
        entry['question'] = question
        entry['sql'] = parse_sql(sqlquery, entry_table['header'])
        entry['query_tok'] = stanford.tokenize(sqlquery)
        entry['question_tok'] = stanford.tokenize(question)
        entry['table_id'] = 'mock_time_machine'
        json.dump(entry, f)
        f.write('\n')
    
    f.close()
    os.system('cp dummy_tok.jsonl ../data/')
