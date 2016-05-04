import xlrd


table_data = xlrd.open_workbook('~') 

table = table_data.sheet_by_index(0) 

col_data = table.col_values(3)

raw_data = table.raw_values(2)

row_num = table.nrows



for i in col_data:
 
   print i 