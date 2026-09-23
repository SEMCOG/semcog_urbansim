import os
import win32com.client as w32c
import win32pdhutil, win32api, win32process
import numpy as np, pandas as pd
import csv
import orca

def transcad_interaction(zonal_indicators, taz_table):
    datatable = 'TAZ Data Table'
    joinfield = 'ID'
    input_file = 'D:\\semcog_e6\\tm_input.tab' #Should be full path
    dbname = 'D:\\semcog_e6\\semcog_e6_ui'
    transcad_exec = 'C://Program Files (x86)//TransCAD//tcw.exe'
    tm_output_full_name = 'D:\\semcog_e6\\out\\tm_output.txt'
    zonal_indicators.to_csv(input_file, sep='\t', index = False)
    
    ####Start transcad prior to running simulation!
    def transcad_operation(macroname, dbname, args):
        tc = w32c.Dispatch("TransCAD.AutomationServer")
        return tc.Macro(macroname, dbname, args)
    
    #gets dict of locations of all tm files
    file_locations = dict(transcad_operation('SEMCOGGetFileLocation', dbname, None))
    
    datatable = file_locations[datatable]
    #Import urbansim csv file (need to give correct path to csv and also it should be a tab file)
    macro_args = [["InputFile", input_file],
                  ["DataTable", datatable],
                  ["JoinField", joinfield]
              ]
    print('Importing UrbanSim data into Transcad')
    transcad_operation('SEMCOGImportTabFile', dbname, macro_args)
    
    ####Run TM      
    loops = 1
    print('Running Transcad model')
    transcad_operation('SEMCOG Run Loops', dbname, loops)
    
    #Import skims from travel model
    matrix_attribute_name_map = { 'row_index_name':'ZoneID', 'col_index_name':'ZoneID', 
                                  'AMHwySkims':{'Miles':'highway_distance','Trav_Time':'highway_travel_time'},
                                  'AMTransitSkim':{'Generalized Cost':'generalized_cost'} }
    matrices = []
    row_index_name, col_index_name = "ZoneID", "ZoneID"  #default values
    if 'row_index_name' in matrix_attribute_name_map:
        row_index_name = matrix_attribute_name_map['row_index_name']
    if 'col_index_name' in matrix_attribute_name_map:
        col_index_name = matrix_attribute_name_map['col_index_name']
    for key, val in matrix_attribute_name_map.items():
        if (key != 'row_index_name') and (key != 'col_index_name'):
            if 'row_index_name' in val:
                row_index_name = val['row_index_name']
            if 'col_index_name' in val:
                col_index_name = val['col_index_name']
            matrix_file_name = file_locations[key]  #replace internal matrix name with absolute file name
            matrices.append([matrix_file_name, row_index_name, col_index_name, list(val.items())])
    macro_args =[ ("ExportTo", tm_output_full_name) ]
    macro_args.append(("Matrix", matrices))
    print('Exporting matrices from Transcad to csv')
    transcad_operation('SEMCOGExportMatrices', dbname, macro_args)
    
    #Read/process outputted csv
    def read_macro_output_file(filename, MISSING_VALUE=-9999):
        """this function is tailored to read the output file from semcog export macro"""
        root, ext = os.path.splitext(filename)
        header_filename = root + ".DCC"
        header_fd = open(header_filename, 'r')
        headers = []
        for line in header_fd.readlines():
            line=line.replace('"','')
            item_list = line.split(",")
            if len(item_list) < 2:
                continue
            if item_list[0] == "ZoneID" and "from_zone_id" not in headers:
                headers.append("from_zone_id")  # the first RCIndex is from_zone_id?
            elif item_list[0] == "ZoneID":
                headers.append("to_zone_id")    # the second RCIndex is to_zone_id?
            else:
                headers.append(item_list[0].lower())
        header_fd.close()
        return_dict = {}
        text_file = open(filename, 'r')
        for item in headers:
            return_dict[item] = []
        reader = csv.reader(text_file)
        for items in reader:
            for col_index in range(0, len(items)):
                value = items[col_index]
                if value == '':  #missing value
                    value = MISSING_VALUE
                try: 
                    v = int(value) #integer
                except ValueError:  #not an integer
                    try:
                        v = float(value) #float
                    except ValueError:
                        v = value  #string
                return_dict[headers[col_index]].append(v)
        text_file.close()
        for item, value in return_dict.items():
            try:
                return_dict[item] = np.array(value)
            except:
                pass
        return return_dict
    
    data_dict = read_macro_output_file(tm_output_full_name)
    id_zone_xref = taz_table[['ID','TAZCE10_N']]
    df = pd.DataFrame(data_dict)
    df2 = pd.merge(df, id_zone_xref, left_on='from_zone_id',right_on='ID',how='outer')
    df2.TAZCE10_N[df2.TAZCE10_N.isnull()]= df2.from_zone_id[df2.TAZCE10_N.isnull()]+73407
    df2['from_zone_id'] = df2.TAZCE10_N
    del df2['TAZCE10_N']
    del df2['ID']
    df3 = pd.merge(df2, id_zone_xref, left_on='to_zone_id',right_on='ID',how='outer')
    df3.TAZCE10_N[df3.TAZCE10_N.isnull()]= df3.to_zone_id[df3.TAZCE10_N.isnull()]+73407
    df3['to_zone_id'] = df3.TAZCE10_N
    del df3['TAZCE10_N']
    del df3['ID']
    df3 = df3.set_index(['from_zone_id','to_zone_id'])
    df3 = df3.rename(columns={'highway_travel_time': 'am_single_vehicle_to_work_travel_time'})
    
    #Update simulation travel data
    orca.add_table("travel_data", df3)
