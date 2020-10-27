# -*- coding: utf-8 -*-

import configparser
import argparse
import collections
from collections import OrderedDict
import json
import h5py
import numpy as np
import os
import subprocess

def get_args():
        parser = argparse.ArgumentParser()

        parser.add_argument("--config", help="Network Architecture. set same json file of dbbs-scaffold: default(./json/scaffold_configuration.json)", default="./json/scaffold_configuration.json")
        parser.add_argument("--hdf5", help="hdf5 file made by dbbs-scaffold 2.0.2: default(./hdf5/scaffold_network_200x800.hdf5)", default="./hdf5/scaffold_network_200x800.hdf5")
        parser.add_argument("--parameter", help="simulation parameter. if this is empty, the file set config will be used.")
        parser.add_argument("--input", help="input stimulation parameter.")
        parser.add_argument("--outdir", help="output directory.", default="./output")
        parser.add_argument("--target", help="set files you want to make. example) --target sud. s = struct_enum_def.h, u = userdefined.h, d = parse data, i = input stimulation parameter", default="suid")

        args = parser.parse_args()

        return(args)


def struct_enum_def( Cells, Connections, hdf5, outdir ):
    with open( 'base_file/struct_enum_def.h' ) as f:
        data_lines = f.read()

    s = '\n'.join([ '#define\t'+key.replace(" ","")+'_num\t('+str(value['num'])+')' for key, value in Cells.items() ]) + '\n\n\n'
    data_lines = data_lines.replace("{def_neuron_num}", s)

    ################################# cell type enum def
    s = 'enum\tNeuronType {\n'\
            '\tNONE = -1,\n\t'\
            + ',\n\t'.join( Cells.keys() ) + ',\n'\
            + '\tTotalNumOfCellTypes\n'\
            '};\n\n'
    s = s.replace(" ", "")
    data_lines = data_lines.replace("{enum_neuron_type}", s)
    
    ConnectionVolume = {}
    for name, value in Connections.items() :
        ConnectionVolume[name] = hdf5['cells/connections/'+name ].len() / hdf5['cells/placement/'+value['to_cell_types'][0]['type']+'/positions'].len()
    volume_sorted = sorted( ConnectionVolume.items(), key=lambda x:x[1], reverse=True )
    Keys = [ v[0] for v in volume_sorted ]

    ################################# connection enum def
    s = 'enum\tConnectionType {\n'\
            '\tInput= -1,\n\t'\
            + ',\n\t'.join( Keys ) + ',\n'\
            + '\tTotalNumOfConnectivityTypes\n'\
            '};\n\n'
    s = s.replace(" ", "")
    data_lines = data_lines.replace("{enum_connection_type}", s)

    with open( outdir+'/struct_enum_def.h', mode='w') as f:
        f.write(data_lines)

def ParseConnectionMatricies(Cells, Connections, hdf5,base_id_list, outdir):

        id_arr = np.array( hdf5['cells/positions'] )
        id_List = {}
        for key, value in Cells.items():
            tmp = id_arr[  base_id_list[1][value['tmp_id']]:base_id_list[1][value['tmp_id']]+value['num'] ,:]
            id_List[key] = np.argsort( np.argsort(tmp[:,2]) )

        for connection, value in Connections.items():
                print(outdir+'/data/'+connection+'.dat')
                arr = np.array( hdf5['cells/connections/' + connection ] ,dtype=np.int64 )
                from_base = Cells[ value['from_cell_types'][0]['type'] ]['base_id']
                to_base = Cells[ value['to_cell_types'][0]['type'] ]['base_id']
                List = []
                from_ids = id_List[value['from_cell_types'][0]['type']][ arr[:,0] - from_base ]
                to_ids = id_List[value['to_cell_types'][0]['type']][ arr[:,1] - to_base ]

                A = np.stack( (from_ids, to_ids), axis=-1 )
                A = A[np.lexsort( (A[:, 0], A[:,1] ) ) ]
                np.savetxt(outdir+'/data/'+connection+'.dat', A, delimiter='\t', fmt='%d')


def init_params(Cells, Connections, hdf5, outdir):
    with open('base_file/init_params.cu') as f:
        data_lines = f.read()
    #s = 'void init_neurons_params( Neuron *Neurons, int *NeuronTypeID){\n\n'
    s = ""
    for key, value in Cells.items():
            if 'neuron_model' in value.keys()  and value['neuron_model']  == 'parrot_neuron':
                    s += '\tNeuronTypeID['+key.replace(" ","")+'] = set_neuron_params(\n'\
                             '\t\tNeurons,\n'\
                             '\t\t'+key.replace(" ","")+',\n'\
                             '\t\t"'+key.replace(" ","")+'.dat",\n'\
                             '\t\t'+str(value['duplicate'])+',\n'\
                             '\t\t'+key.replace(" ","")+'_num,\n'\
                             '\t\t0,\n'\
                             '\t\t0,\n'\
                             '\t\t0,\n'\
                             '\t\t0,\n'\
                             '\t\t0,\n'\
                             '\t\t0,\n'\
                             '\t\t0,\n'\
                             '\t\t0,\n'\
                             '\t\t0,\n'\
                             '\t\t0,\n'\
                             '\t\t'+value['device'].upper()+'\n'\
                             '\t);\n\n'
            else:
                    iaf_cond_alpha = value['iaf_cond_alpha']
                    param = value['parameters']
                    s += '\tNeuronTypeID['+key.replace(" ","")+'] = set_neuron_params(\n'\
                             '\t\tNeurons,\n'\
                             '\t\t'+key.replace(" ","")+',\n'\
                             '\t\t"'+key.replace(" ","")+'.dat",\n'\
                             '\t\t'+str(value['duplicate'])+',\n'\
                             '\t\t'+key.replace(" ","")+'_num,\n'\
                             '\t\t'+str(param['C_m'])+',\n'\
                             '\t\t'+str(param['C_m']/iaf_cond_alpha['g_L'])+',\n'\
                             '\t\t'+str(param['E_L'])+',\n'\
                             '\t\t'+str(param['t_ref'])+',\n'\
                             '\t\t'+str(iaf_cond_alpha['I_e'])+',\n'\
                             '\t\t'+str(param['V_reset'])+',\n'\
                             '\t\t'+str(param['V_th'])+',\n'\
                             '\t\t'+str(iaf_cond_alpha['tau_syn_ex'])+',\n'\
                             '\t\t'+str(iaf_cond_alpha['tau_syn_in'])+',\n'\
                             '\t\t'+str(iaf_cond_alpha['g_L'])+',\n'\
                             '\t\t'+value['device'].upper()+'\n'\
                             '\t);\n\n'
    data_lines = data_lines.replace("{set_neuron_params}", s)

    ########### init_connectivity_params
    s=""
    ConnectionVolume = {}
    for name, value in Connections.items() :
        ConnectionVolume[name] = hdf5['cells/connections/'+name ].len() / hdf5['cells/placement/'+value['to_cell_types'][0]['type']+'/positions'].len()
    volume_sorted = sorted( ConnectionVolume.items(), key=lambda x:x[1], reverse=True )
    Keys = [ v[0] for v in volume_sorted ]

    for key in Keys:
        value = Connections[key]
        print(key)
        avg_conv = hdf5['cells/connections/'+key].len() / Cells[ value['to_cell_types'][0]['type'] ]['num']
        pr = 0
        delay = 1
        if avg_conv > 10000 :
            pr = 1
        if type( value['connection']['delay'] ) is float :
            delay = value['connection']['delay']
        elif 'mu' in value['connection']['delay'].keys() :
            delay = value['connection']['delay']['mu']

        s += '\tConnectivityTypeID['+key.replace(" ","")+'] = set_connectivity_params(\n'\
                         '\t\tconnectivities,\n'\
                         '\t\tneurons,\n'\
                         '\t\t'+key+',\n'\
                         '\t\t"'+key+'.dat",\n'\
                         '\t\t'+value['from_cell_types'][0]['type']+'_num,\n'\
                         '\t\t'+value['to_cell_types'][0]['type']+'_num,\n'\
                         '\t\tNeuronTypeID['+value['from_cell_types'][0]['type']+'],\n'\
                         '\t\tNeuronTypeID['+value['to_cell_types'][0]['type']+'],\n'\
                         '\t\t'+str(value['connection']['weight'])+',\n'\
                         '\t\t'+str(delay)+',\n'\
                         '\t\t'+str(pr)+'\n'\
                         '\t);\n\n'
    data_lines = data_lines.replace("{set_connectivity_params}", s) 
    with open( outdir + '/init_params.cu', 'w') as f :
        f.write(data_lines)


def input_stim( config, hdf5,  Cells, base_id_list, outdir ):

   with open('base_file/UserInputFunctions.cu') as f:
        data_lines_cu = f.read()

   with open('base_file/UserInputFunctions.h') as f:
        data_lines_h = f.read()

   id_arr = np.array( hdf5['cells/positions'] )
   sorted_id = {}
   for key, value in Cells.items():
       tmp = id_arr[  base_id_list[1][value['tmp_id']]:base_id_list[1][value['tmp_id']]+value['num'] ,:]
       sorted_id[key] = np.argsort( np.argsort(tmp[:,2]) )

   input_devices = { k:v for k, v in config['simulations']['FCN_2019']['devices'].items() if v['io'] == 'input' }
   print(input_devices)
   
   #extract target cells
   for name, stim in input_devices.items() :
       cell_type = stim["cell_types"][0]
       if stim["targetting"]  == "cylinder" :
           stim["center"] =np.array( [config["network_architecture"]["simulation_volume_x"]/2, 0,  config["network_architecture"]["simulation_volume_z"]/2] ).reshape((1,3))
           cells = np.array( hdf5['cells/placement/'+cell_type+'/positions'] )
           stim["IdList"] = sorted_id[cell_type][np.sum( (cells[:,(0,2)] - stim["center"][:,(0,2)]) ** 2, axis=1 ) < stim['radius']**2 ]
       elif stim["targetting"] == "cell_type" :
           stim["IdList"] = sorted_id[cell_type]

       np.savetxt(outdir+'/data/'+name+'.dat', np.sort( stim["IdList"] ), delimiter='\n', fmt='%d')

   s = ''
   for key, stim in input_devices.items():
       s += '__device__ char '+str(key)+'(const float r, const CTYPE time){\n\tchar flag = 0;\n'
       s += '\tflag = ('
       if stim['device'] == 'poisson_generator' :
           s += '('+str(stim['parameters']['start'])+' <= time && time < '+str( stim['parameters']['stop'] )+ ') && '
           s += 'PoissonProcess( r, time, '+str( stim['parameters']['rate'] )+', '+str(stim['parameters']['start'])+' ) );\n'
       elif stim['device'] == 'spike_generator' :
           s += '('+str(stim['stimulus']['variables']['start'])+' <= time && time <= '+str( stim['stimulus']['variables']['start'] + stim['stimulus']['variables']['duration'] )+ ') && '
           s += 'PeriodicFiring( r, time, '+str( stim['stimulus']['variables']['num_spikes']*1000/stim['stimulus']['variables']['duration'] )+', '+str(stim['stimulus']['variables']['start'])+' ) );\n'
       s += '\treturn (flag)?1:0;\n}\n\n'
   data_lines_cu = data_lines_cu.replace('{noise_func}', s)
   

   s = ''
   count = 0
   for key, stim in input_devices.items():
           s+=     '\tif ( (fp = fopen( "'+str(key)+'.dat" , "r")) == NULL){\n'\
                   '\t\tfprintf(stderr, "cannot open file: '+str(key)+'.dat\\n");\n'\
                   '\t\texit(1);\n'\
                   '\t}\n\n'\
                   '\tList['+str(count)+'].type = NeuronTypeID['+str(stim["cell_types"][0])+'];\n'\
                   '\tList['+str(count)+'].base_id = host_neurons[NeuronTypeID['+str(stim["cell_types"][0])+']].base_id;\n'\
                   '\tList['+str(count)+'].num = '+str(len(stim["IdList"]))+';\n'\
                   '\tList['+str(count)+'].func_id = '+str(count)+';\n'\
                   '\ti=0;\n\tList['+str(count)+'].IdList = (unsigned int*)malloc(sizeof(unsigned int)*List['+str(count)+'].num);\n'\
                   '\twhile( fgets(str, 256, fp) != NULL ){\n\t\tsscanf(str, "%u", &List['+str(count)+'].IdList[i]);\n\t\ti++;\n\t}\n'\
                   '\tfclose(fp);\n\n\n'
           count+=1
   data_lines_cu = data_lines_cu.replace('{stim_init}', s)

   data_lines_cu = data_lines_cu.replace('{func_pointer}' , '__device__ pointFunction_t d_pInputFunctions[] = {'+','.join( input_devices.keys() )+'};\n\n')
   
   with open(outdir + '/UserInputFunctions.cu', 'w') as f:
       f.write(data_lines_cu)
   
   with open(outdir + '/UserInputFunctions.h', 'w') as f:
       f.write( data_lines_h.replace('{input_stim_num}', str(len(input_devices)) ) )
   #subprocess.run('cat header.cu > '+outdir+'/UserInputFunctions.cu', shell=True)
   #subprocess.run('cat tmp.cu >> '+outdir+'/UserInputFunctions.cu', shell=True)
   #subprocess.run('cat footer.cu >> '+outdir+'/UserInputFunctions.cu', shell=True)
   #subprocess.call(["rm","tmp.cu"])
   #
   #subprocess.run('cat header_UserInput.h > '+outdir+'/UserInputFunctions.h',shell=True)
   #subprocess.run('echo "#define INPUT_STIM_NUM '+str( len(input_devices) )+' ">> '+outdir+'/UserInputFunctions.h',shell=True)
   #subprocess.run('echo "#endif ">> '+outdir+'/UserInputFunctions.h',shell=True)
        

def main():

        args = get_args()

#        config_ini = configparser.ConfigParser()
#        config_ini.read( args.config ,'utf-8')

        jsonfile = open(args.config, 'r')
        config_data = json.load( jsonfile, object_pairs_hook=OrderedDict )
        hdf5 = h5py.File( args.hdf5 , mode='r+')


#        sim_ini = configparser.ConfigParser()
#        if args.parameter:
#                sim_ini.read( args.parameter )
#        else:
#                sim_ini.read( args.config )


        total_cell_nums = hdf5["cells/positions"].len()
        base_id_list = np.unique(hdf5["cells/positions"][:,1], return_index=True)

        Cells = config_data['cell_types']
        tmp = 0
        cell_i = 0
        for key, value in Cells.items():
            if key in config_data['simulations']['FCN_2019']['cell_models'].keys():
                Cells[key].update( config_data['simulations']['FCN_2019']['cell_models'][key] )
            Cells[key]['type_id'] = cell_i
            Cells[key]['tmp_id'] = tmp
            Cells[key]['num'] = int( hdf5["cells/placement/"+key+"/identifiers"][1] )
            if 'duplicate' not in Cells[key].keys():
                Cells[key]['duplicate'] = 0
            if key in config_data['simulations']['FCN_2019']['GPU_option']['output'] :
                Cells[key]['device'] = 'OUTPUT'
            elif  key in config_data['simulations']['FCN_2019']['GPU_option']['input'] :
                Cells[key]['device'] = 'INPUT'
            else :
                Cells[key]['device'] = 'NORMAL'

            if cell_i in base_id_list[0]:
                Cells[key]['base_id'] = int( hdf5["cells/positions"][base_id_list[1][ tmp ], 0] )
                print( str( base_id_list[1][tmp] ) + ":" + str( Cells[key]['base_id'] ))
            else :
                Cells[key]['base_id'] = -1
                tmp -= 1
            tmp += 1
            cell_i += 1

        for value in config_data["simulations"]['FCN_2019']['GPU_option']["output"]:
            Cells[value]['device'] = 'output'

        Connections = {}
        for key, value in config_data['connection_types'].items():
            if len(  value['from_cell_types'] ) > 1 :
                for fromcell in value['from_cell_types'] :
                    sub_key = fromcell['type'].split('_')[0] + "_to_" + value['to_cell_types'][0]['type'].split('_')[0]
                    Connections[ sub_key ] = {"from_cell_types":[ fromcell ], "to_cell_types": value['to_cell_types']}
                    if sub_key in config_data['simulations']['FCN_2019']['connection_models'].keys() :
                        Connections[sub_key].update( config_data['simulations']['FCN_2019']['connection_models'][sub_key] )
            elif len(value['from_cell_types'][0]['compartments']) > 1:
                for sub_key in [v for k, v in value.items() if 'tag' in k]:
                    Connections[ sub_key ] = value
                    if sub_key in config_data['simulations']['FCN_2019']['connection_models'].keys() :
                       Connections[sub_key].update( config_data['simulations']['FCN_2019']['connection_models'][sub_key] )
            else :
                if key in config_data['simulations']['FCN_2019']['connection_models'].keys() :
                    Connections[key] = value
                    Connections[key].update( config_data['simulations']['FCN_2019']['connection_models'][key] )


#list removable entities
        entities = []
        for key, value in Cells.items():
            if "entity" in value.keys() and value['entity']:
                entities.append( key )
                print("remove "+key)

        uc_connections = [] #unconsidered_connections
        for key, value in Connections.items():
            if value['from_cell_types'][0]['type'] in entities :
                uc_connections.append( key )
                print("remove "+key)

#remove
        for key in entities:
            Cells.pop(key)
        for key in uc_connections :
            Connections.pop(key)

        os.makedirs(args.outdir+'/data', exist_ok=True)


#ad-hoc  from ( mf->glom and mf->dcn) to (glom -> dcn)
        if 'glomerulus_to_dcn' not in hdf5['cells/connections/'].keys():
            glom_to_mf = np.stack( [np.array( hdf5['cells/connections/mossy_to_glomerulus'][:,1] ), np.array( hdf5['cells/connections/mossy_to_glomerulus'][:,0])], 1 )
            tmp = np.array( hdf5['cells/connections/mossy_to_dcn'])
            mf_to_dcn = tmp[ np.argsort( tmp[:,0] ) ]
            glom_to_dcn = []
            for mf_dcn in mf_to_dcn:
                for glom_mf in glom_to_mf:
                    if glom_mf[1] == mf_dcn[0]:
                        glom_to_dcn.append( [glom_mf[0], mf_dcn[1] ] )
            glom_to_dcn = np.array(glom_to_dcn)
            print('glom_to_dcn')
            print( glom_to_dcn.shape )
            print( np.lexsort( (glom_to_dcn[:, 0], glom_to_dcn[:,1] ) ).shape  )
            glom_to_dcn = glom_to_dcn[np.lexsort( (glom_to_dcn[:, 0], glom_to_dcn[:,1] ) ) ]
            hdf5.create_dataset("cells/connections/glomerulus_to_dcn", data=glom_to_dcn)
        tmp = config_data['simulations']['FCN_2019']['connection_models']
        Connections['glomerulus_to_dcn'] = {'from_cell_types':[{'type': 'glomerulus','compartments':''}], 'to_cell_types':[{'type':'dcn_cell','compartments':''}], 'connection': {'weight': tmp['mossy_to_dcn']['connection']['weight']/tmp['mossy_to_glomerulus']['connection']['weight'], 'delay':tmp['mossy_to_dcn']['connection']['delay'] - tmp['mossy_to_glomerulus']['connection']['delay']}, 'synapse':{'static_synapse':{}}}
        print('add glomerulus_to_dcn')



#        base = 0
#        for key, value in cell_nums.items():
#                base_id[key] = base
#                base += value

        total_cell_types = len(Cells)
       
        print(args.target)
        if "d" in args.target:
                ParseConnectionMatricies(Cells, Connections, hdf5, base_id_list, args.outdir)
        if "u" in args.target:
                init_params( Cells, Connections, hdf5,  args.outdir)
        if "i" in args.target:
                input_stim( config_data, hdf5,  Cells, base_id_list, args.outdir )
        if "s" in args.target:
                struct_enum_def(Cells, Connections, hdf5, args.outdir)

        hdf5.close()
        jsonfile.close()


if __name__ == '__main__' :
        main()
