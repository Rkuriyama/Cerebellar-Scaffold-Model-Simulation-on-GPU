import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import NullFormatter
import sys

args = sys.argv

input_file = args[1]
data = np.loadtxt( input_file )
data = data[ np.lexsort( ( data[:,0], data[:,1], data[:,2] ) ) ]


neuron_id, index, count = np.unique( data[:,2], return_index=True, return_counts=True )
print(neuron_id)
neuron_name = open( "NeuronNames.txt", "r" )
names = neuron_name.readlines()

neuron_id = neuron_id.astype(np.int32)
neuron_id_l = neuron_id.tolist()
neuron_type_num = len( neuron_id_l )

bin_width = float( args[2] )

duration = np.max( data, axis=0 )[0]
print(duration)
firing_rate = []
for i, val in enumerate( neuron_id_l) :
    tmp = []
    for start in [ k*bin_width for k in range( 0, int( duration/bin_width ) )] :
        tmp_d = data[ index[i] : index[i]+count[i] - 1, 0 ]
        neuron_num = np.unique( data[ index[i] : index[i]+count[i] - 1, 1 ] ).size
        if neuron_num == 0 :
            tmp.append(0)
            continue
        tmp.append( np.count_nonzero( (start <= tmp_d) & (tmp_d < start + bin_width) ) / neuron_num / bin_width * 1000 )
    firing_rate.append( tmp )


x_axis = [ k*bin_width+bin_width/2 for k in range( 0, int( duration/bin_width ) )]
x_range = [ np.min( data[:,0] ), np.max(data[:,0]) ]
#x_range = [ 1000, 1250 ]

#figure = plt.figure()
#j = 0
#ids = [0,3,4,5,2,6]
#plt.rcParams["font.size"] = 28
#
#for val in ids :
#    i = val
#    ax = figure.add_subplot(3,2,j+1)
#    ax.set_xlim(x_range[0], x_range[1])
#    ax.scatter( data[ index[i] : index[i]+count[i] - 1 , 0], data[ index[i] : index[i]+count[i] - 1, 1], marker=",", s=(72./figure.dpi)**2, color=cm.Set1.colors[1]  )
#    ax.set_title( names[i].replace("\n",""), verticalalignment='top' )
#    j = j + 1
#plt.gca().yaxis.set_minor_formatter(NullFormatter())
#plt.subplots_adjust(hspace=0.35,wspace=0.35)

figures = []
i = 0
for val in neuron_id_l :
    figures.append( plt.figure() )
    ax = figures[-1].add_subplot(1,1,1)
    ax.set_xlim( x_range[0], x_range[1] )
    if val == 0:
        #ax.set_ylim( 10000, 20000 )
        print("GrC")
    ax2 = ax.twinx()

    ax.scatter( data[ index[i] : index[i]+count[i] - 1 , 0], data[ index[i] : index[i]+count[i] - 1, 1], marker=",", s=(72./figures[-1].dpi)**2, color=cm.Set1.colors[1]  )
    ax2.plot( x_axis, firing_rate[i], color=cm.Set1.colors[0] )
    ax.set_title( names[val] )
    i = i + 1

plt.show()

neuron_name.close()
