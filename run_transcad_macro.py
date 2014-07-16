import os
import win32com.client as w32c
import win32pdhutil, win32api, win32process
import numpy as np, pandas as pd

def run_get_file_location_macro():    
    macroname, dbname = 'SEMCOGGetFileLocation', 'D:\\semcog_e5_35\\semcog_e5_ui'
    args = None
    return dict(run_transcad_macro(macroname, dbname, args))

def run_transcad_macro(macroname, dbname, args=None):
    tc = w32c.Dispatch("TransCAD.AutomationServer")
    try:
        return_val = tc.Macro(macroname, dbname, args)
        if return_val is not None:
            try:
                print return_val[-1][-1]  #if the return is an error msg
            except:
                return return_val  #else pass the return_val to the caller
    finally:
        del tc

def set_project_ini_file(project_year_dir, year):
    ini_file = 'C://Program Files (x86)//TransCAD//semcog_e5.ini'
    ini_fp = open(ini_file,"r")
    import ConfigParser
    cfg = ConfigParser.ConfigParser()
    cfg.readfp(ini_fp)
    for section in cfg.sections():
        for option in cfg.options(section):
            dir = cfg.get(section, option)  #[('c', '\semcog\semcog_bin.mode')]
            cfg.remove_option(section, option)
            
            file_name = os.path.basename(dir)
            if not os.path.splitext(file_name)[1]:   #special handling for data directory section
                file_name = ''
            new_dir = os.path.join(project_year_dir, file_name)
    #                new_option,new_value = new_dir.split(":")
            new_option = new_dir
            new_value = ''
            v=cfg.set(section, new_option, new_value)
    
       # cfg.write(ini_fp)
    
    #if '[UI File]' section is in ini file and config has key ui_file
    #overwrite [UI File] section with config['ui_file']
    #config['ui_file'] must be ui_file without extension
    ui_section = 'UI File'
    if cfg.has_section(ui_section):
            ui_option = cfg.options(ui_section)[0]  #only 1 option
            ui_ext = os.path.splitext(ui_option)[1] #file extension
            cfg.remove_option(ui_section, ui_option)
            cfg.set(ui_section, 'D:\\semcog_e5_35\\semcog_e5_ui' + ui_ext, '')
            
    ini_fp.close()
    ini_fp = open(ini_file,"w")
    
    for section in cfg.sections():
        ini_fp.write("[%s]\n" % section)
        for option in cfg.options(section):
            ini_fp.write("%s\n" % option)
    
    ini_fp.close()

def prepare_for_run(input_dir, year):
    """before calling travel model macro, check if transcad GUI is running, 
    if not, try to start transcad binary process"""
    ## TODO: TransCAD COM server is very picky about tcw.exe process in memory
    ## so perhaps manually start TransCAD program is needed before running this script
    
    set_project_ini_file(input_dir, year)
    if not check_tcw_process:
        return
    
    cmdline = 'C://Program Files (x86)//TransCAD//tcw.exe'
    head, tail = os.path.split(cmdline)
    procname, ext = os.path.splitext(tail)  #tcw
    
    kill_process = False
    start_program = False
    tc_program_classname = "tcFrame"  #ClassName for TransCAD program
    try:
        hwnd=win32gui.FindWindow(tc_program_classname, None)
    except:
        start_program = True  # No Transcand Window found, we'll need to start TransCAD program
    else:
        try:
            #first check if tcw process is in memory
            win32pdhutil.GetPerformanceAttributes('Process','ID Process',procname)
            pids=win32pdhutil.FindPerformanceAttributesByName(procname)
            for pid in pids:
                win32process.TerminateProcess(pid)
            start_program = True
        except:
            raise RuntimeError, "Unable to kill TransCAD process in memory"
        
    ##transcad not started, try to start it
    if start_program:
        try:
            pass
            cmdline = win32api.GetShortPathName(cmdline)
            cmdline = cmdline + " -q"
            os.system('start /B "start TransCAD" ' + cmdline)  #start TransCAD in background
            time.sleep(9)
            #procHandles = win32process.CreateProcess(None, cmdline, None, None, 0, 0, None, None,
                                 #win32process.STARTUPINFO())
            #self.hProcess, hThread, PId, TId = procHandles
        except:
            print "Unable to start TransCAD in %s; it must be running to invoke travel model macro." % cmdline
            sys.exit(1)

  #otherwise transcad is running, do nothing
  
def run(project_year_dir, year):
    """Runs the travel model 
    """
#        year_dir = tm_config[year]  #'CoreEA0511202006\\urbansim\\2001'
#        dir_part1,dir_part2 = os.path.split(year_dir)
#        while dir_part1:
#            dir_part1, dir_part2 = os.path.split(dir_part1)
#        project_year_dir = os.path.join(tm_data_dir, dir_part2)   #C:/SEMCOG_baseline/CoreEA0511202006
    prepare_for_run(project_year_dir, year)
    print 'Start travel model from directory %s for year %d' % (project_year_dir, year)
    #for macroname, ui_db_file in tm_config['macro']['run_semcog_travel_model'].iteritems():
        #pass 
    macroname, ui_db_file = 'SEMCOG Run Loops', 'D:\\semcog_e5_35\\semcog_e5_ui'

    loops = 1
    logger.log_status('Running travel model ...')
    tcwcmd = win32api.GetShortPathName(tm_config['transcad_binary'])

    os.system('start /B "start TransCAD" %s' % tcwcmd)  #start TransCAD in background
    time.sleep(1)
    #os.system("%s -a %s -ai '%s'" % (tcwcmd, ui_db_file, macroname))
    run_transcad_macro(macroname, ui_db_file, loops)
    
    try:
        pass
        ##win32process.TerminateProcess(self.hProcess, 0)
    except:
        print "The code has problem to terminate the TransCAD it started."
        
        
def get_travel_data_from_travel_model(tm_data_dir, 
                                      year, 
                                      tm_output_file="tm_output.txt",
                                      ):
    """
    Extracts a new travel data set from a given set of transcad matrices 
    by calling a pre-specified transcad macro.  
    """

    tm_output_full_name = os.path.join(tm_data_dir, tm_output_file)
    matrix_attribute_name_map = { 'row_index_name':'ZoneID', 'col_index_name':'ZoneID', 
                                  'AMHwySkims':{'Miles':'highway_distance','Trav_Time':'highway_travel_time'},
                                  'AMTransitSkim':{'Generalized Cost':'generalized_cost'} }
    
    transcad_file_location = run_get_file_location_macro()
    
    matrices = []
    row_index_name, col_index_name = "ZoneID", "ZoneID"  #default values
    if matrix_attribute_name_map.has_key('row_index_name'):
        row_index_name = matrix_attribute_name_map['row_index_name']
    if matrix_attribute_name_map.has_key('col_index_name'):
        col_index_name = matrix_attribute_name_map['col_index_name']
        
    for key, val in matrix_attribute_name_map.iteritems():
        if (key != 'row_index_name') and (key != 'col_index_name'):
            if val.has_key('row_index_name'):
                row_index_name = val['row_index_name']
            if val.has_key('col_index_name'):
                col_index_name = val['col_index_name']
            matrix_file_name = transcad_file_location[key]  #replace internal matrix name with absolute file name
            matrices.append([matrix_file_name, row_index_name, col_index_name, val.items()])
              
    macro_args =[ ("ExportTo", tm_output_full_name) ]
    macro_args.append(("Matrix", matrices))
    #for macroname, ui_db_file in tm_config['macro']['get_transcad_data_into_cache'].iteritems():
        #ui_db_file = os.path.join(tm_config['directory'], ui_db_file)
    macroname, ui_db_file = 'SEMCOGExportMatrices', 'D:\\semcog_e5_35\\semcog_e5_ui'
    run_transcad_macro(macroname, ui_db_file, macro_args)

    table_name = "travel_data"
    data_dict = read_macro_output_file(tm_output_full_name)
    df = pd.DataFrame(data_dict)
    dset.travel_data = df

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
    
    for item, value in return_dict.iteritems():
        try:
            return_dict[item] = np.array(value)
        except:
            ##TODO: add handling for string array
            pass
    
    return return_dict
