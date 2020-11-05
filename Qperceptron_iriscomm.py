import DB_iris
from sklearn.utils import shuffle

from math import *
import numpy as np
from qiskit import *
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit import IBMQ
import matplotlib.pyplot as plt
#import operator


AGAINST_CLASS_1 = True # if True then read it as AGAINST CLASS 1 ('IS 'X' ELEMENT FROM CLASS 0 OR FROM CLASS 1?')
# if False then read it as AGAINST CLASS 2 ('IS 'X' ELEMENT FROM CLASS 0 OR FROM CLASS 2?')

DB_iris.HOW_MANY_CLASSES = 2 # BINARY CLASSIFICATION ONLY


classesarray = []

if AGAINST_CLASS_1:
    classesarray = [0,1] 
else:
    classesarray = list(range(3)) # 0, 1, 2... CLASSES APPENDED TO OVERALL[] IN qthresholds

MAX_DATA = len(DB_iris.X)



#===============================================================================
# PARAMETERS AND INIT
#===============================================================================
REAL_DEVICE = False
REVERSE = False


forsolutionsarray = [0] # NUMBER OF CYCLES FOR SOLUTIONS, IN THIS CASE 1 CYCLE ONLY




step = 1        # > 1 FOR BIG DATASETS ONLY    # !!!!!
divisorpower = 0.5


# PREPARE ARRAYS
FLATTEN_MERG_MATR, MERGED_LABELS = DB_iris.main(classnumbers=classesarray, start_index=0, max_data=MAX_DATA)





def input_state(circ, n, theta):
    """special n-qubit input state"""
    for j in range(n):
        circ.h(j)
        circ.u1(theta*float(2**(j)), j)
        #

        
def qft(circ, n, theta):
    """special n-qubit QFT on the qubits in circ."""
    for j in range(n):
        for k in range(j+1,n):
            circ.cu1(theta, k, j)
            #
        circ.barrier()
    circ.cx(1, 2)
    circ.cu3(np.arccos(theta/6), theta, np.arccos(theta/6), 0, 2)
    swap_registers(circ, n)


def qft_rev(circ, n, theta):
    """special alternative n-qubit QFT on the qubits in circ."""
    for j in range(n):
        for k in range(j+1,n):
            circ.cu1(theta, k, j)
            #
        circ.barrier()
    circ.cx(1, 2)
    circ.cu3(exp(sin(theta)), pi*theta/3, exp(-theta), 0, 2)
    swap_registers(circ, n)
    circ.cx(0, 1)
    for jj in range(n):
        circ.h(jj)


def swap_registers(circ, n):
    for j in range(int(np.floor(n/2.))):
        circ.swap(j, n-j-1)
    return circ


def circuit_calcs(nqubits, XX, what_model):


    qft_circuit = QuantumCircuit(nqubits)
    
    # first, prepare the state
    gamma_fun = 0
    
    
    if what_model == 'straight':

        FIRST_COL = 0

        for i in range(FIRST_COL, np.shape(XX)[1] - 2, step):
            for k in range(0, np.shape(XX)[1] - 1, step):
                if i+k >= np.shape(XX)[1]:
                    break
                gamma_fun += (XX[0][i+k]-XX[0][i])*cos(XX[0][k+1-i]-XX[0][i+2])/(np.shape(XX)[1]**divisorpower)
        
        print('gamma_fun start',gamma_fun)
    
        if gamma_fun < -1:
            gamma_fun = -1
        elif gamma_fun > 1:
            gamma_fun = 1
    
        input_state(qft_circuit, nqubits, 1-np.cosh(gamma_fun))
        # qft_circuit.draw(output='mpl')
        # plt.show()
        
        # next, do our modified qft on the prepared state
        qft_circuit.barrier()
        for i in range(FIRST_COL, np.shape(XX)[1] - 2, step):
            for k in range(0, np.shape(XX)[1] - 1, step):
                if i+k >= np.shape(XX)[1]:
                    break
                gamma_fun += tanh(XX[0][i+k]-XX[0][-1])/(np.shape(XX)[1]**divisorpower)

        print('gamma_fun end',gamma_fun)
    
        if gamma_fun < -1:
            gamma_fun = -1
        elif gamma_fun > 1:
            gamma_fun = 1
        
    
        qft(qft_circuit, nqubits, -pi*gamma_fun)



    elif what_model == 'reverse':
    
    # first, prepare the state    
    
        FIRST_COL = 0
        
        for i in range(FIRST_COL, np.shape(XX)[1] - 2, step):
            for k in range(0, np.shape(XX)[1] - 1, step):
                if i+k >= np.shape(XX)[1]:
                    break
                gamma_fun += (XX[0][i+k]-XX[0][i])*exp(sin(XX[0][k+1-i]-XX[0][i+2]))/(np.shape(XX)[1]**divisorpower)

        print('gamma_fun start',gamma_fun)

        if gamma_fun < -1:
            gamma_fun = -1
        elif gamma_fun > 1:
            gamma_fun = 1

        input_state(qft_circuit, nqubits, 1-np.cosh(gamma_fun))
        # qft_circuit.draw(output='mpl')
        # plt.show()
        
        # next, do our modified qft on the prepared state
        qft_circuit.barrier()
        for i in range(FIRST_COL, np.shape(XX)[1] - 2, step):
            for k in range(0, np.shape(XX)[1] - 1, step):
                if i+k >= np.shape(XX)[1]:
                    break
                gamma_fun += tanh(XX[0][i+k]-XX[0][-1])/(np.shape(XX)[1]**divisorpower)

        print('gamma_fun end',gamma_fun)

        if gamma_fun < -1:
            gamma_fun = -1
        elif gamma_fun > 1:
            gamma_fun = 1
           

        qft_rev(qft_circuit, nqubits, -pi*gamma_fun)
    
    
    qft_circuit.measure_all()
    
    
    return qft_circuit


def qcounts(qcircuit, nqubits, REAL_DEVICE):
    
    if REAL_DEVICE == True:
        # run on real Q device
        #=======================================================================
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= nqubits and
                                           not x.configuration().simulator and x.status().operational==True))
        print("least busy backend: ", backend)
        shots = 8192
        job_exp = execute(qcircuit, backend=backend, shots=shots)
        job_monitor(job_exp)
        print(job_exp.error_message())
        fourier_counts = job_exp.result().get_counts()
        #plot_histogram([fourier_counts], legend=['counts'])
        #plt.show()
    else:
        # run on local simulator
        #=======================================================================
        shots = 8192
        qasm_sim = Aer.get_backend("qasm_simulator")
        fourier_counts = execute(qcircuit, backend=qasm_sim, shots=shots).result().get_counts()
        #plot_histogram([fourier_counts], legend=['counts'])
        #plt.show()    

    return fourier_counts




def qthresholds(fourier_counts, reverse, what_model):

    OVERALL = []
    
    
    if what_model == 'straight':

    
        if '110' in fourier_counts and '111' in fourier_counts:
            
            if REVERSE == False:
    
                if fourier_counts['110'] < fourier_counts['111']:
                    OVERALL.append(classesarray[len(classesarray)-1])
                else:
                    OVERALL.append(classesarray[0])
                
            else:
                
                if fourier_counts['110'] < fourier_counts['111']:
                    OVERALL.append(classesarray[0])
                else:
                    OVERALL.append(classesarray[len(classesarray)-1])        

        else:
            OVERALL.append(classesarray[1])
    

    elif what_model == 'reverse':


        if '001' in fourier_counts and '101' in fourier_counts:
            
            if REVERSE == False:
    
                if fourier_counts['001'] < fourier_counts['101']:
                    OVERALL.append(classesarray[len(classesarray)-1])
                else:
                    OVERALL.append(classesarray[0])
            
            else:
                
                if fourier_counts['001'] < fourier_counts['101']:
                    OVERALL.append(classesarray[0])
                else:
                    OVERALL.append(classesarray[len(classesarray)-1])
    
        else:
            OVERALL.append(classesarray[1])



    return OVERALL



print('forsolutionsarray',forsolutionsarray) # IN THIS CASE, FOR THIS PURPOSE, WE ITERATE ONLY 1 TIME

MODEL_TYPE = ['straight','reverse']
ERRORSCORES = -1 * np.ones((len(MODEL_TYPE),len(forsolutionsarray)))
ENTROPY = -1 * np.ones((len(forsolutionsarray)))


for everyclass in forsolutionsarray:

    

    print('CHECK max min',np.max(FLATTEN_MERG_MATR),np.min(FLATTEN_MERG_MATR))
    
    
    X = np.exp(np.array(FLATTEN_MERG_MATR))
    CY = MERGED_LABELS
    
    print('CY',len(CY),'X',len(X))

    
    # SHUFFLE    
    X, CY = shuffle(X,CY)
    
    print(CY,'shape(CY)',np.shape(CY),'\n','shape(X)',np.shape(X))
    
    
    # BINARIZATION OF THE CONTROL
    TOTAL_SCANNED_RESULT  = np.zeros((len(MODEL_TYPE), len(CY)))
    TOTAL_SCANNED_ERRORS_AS_CLASS = np.zeros((len(MODEL_TYPE), len(CY)))
    TOTAL_UNKNOWNS = np.zeros((len(MODEL_TYPE), len(CY)))

        
    
    start_for = 0
    end_for = min(MAX_DATA, len(CY))

    
    for model in range(len(MODEL_TYPE)):   

        counter = 0
        for index in range(start_for,end_for):
            
            
            # SKIP THE NOT-USED CLASS -> check classesarray[] 
            if AGAINST_CLASS_1:
                if CY[index] == 2:
                    continue
            else:
                if CY[index] == 1:
                    continue

            
            
            row = index   
            XX = [X[row]]
        
            #===============================================================================
            # APPLICATION OF OUR MODIFIED QFT
            #===============================================================================
            nqubits = 3
            qft_circuit = circuit_calcs(nqubits=nqubits, XX=XX, what_model=MODEL_TYPE[model])
            
            fourier_counts = qcounts(qcircuit=qft_circuit, nqubits=nqubits, REAL_DEVICE=REAL_DEVICE)
        
        
            #key_list = list(fourier_counts.keys())
            #sorted_C = sorted(fourier_counts.items(), key=operator.itemgetter(1), reverse=True)
            #print(sorted_C[:2])
        
            # THRESHOLDS
            #=======================================================================
            OVERALL = qthresholds(fourier_counts=fourier_counts, reverse=REVERSE, what_model=MODEL_TYPE[model]) 
                    
                    
            
            print('end of cycle:',MODEL_TYPE[model],counter)
            counter += 1
        
            
            
            RESULT = 999
            print('OVERALL',OVERALL)
            binc = np.bincount(OVERALL) 
            print('binc',binc)
            if len(OVERALL) > 0:
                RESULT = np.argmax(binc) # RESULT IS JUST THE CLASS VALUE (WITHIN classesarray VALUES) 
                # AFTER qthresholds COMPARISONS BETWEEN OUR PIVOTAL COUNTS ('110' and '111', '001' and '101')
            
            #
            CRESULT = RESULT
            print('\n',row,'max data:',MAX_DATA,'\n')
        
        
            if CRESULT != 999: # FOR EXCLUDING ANY ERRORS IN ENCODING THE RESULT

                print('RESULT CLASS:',CRESULT,'( WITH MODEL=',MODEL_TYPE[model],')','REAL CLASS TO GUESS:',CY[row],'\n','\n')
                TOTAL_SCANNED_RESULT[model,index] = CRESULT        
        

                if abs(CRESULT - int(CY[row])) > 0:
                    TOTAL_SCANNED_ERRORS_AS_CLASS[model,index] = 1
                else:
                    TOTAL_SCANNED_ERRORS_AS_CLASS[model,index] = 0

        
            else:
        
                print('RESULT CLASS: UNDEFINED')
                TOTAL_UNKNOWNS[model,index] = 1
        


        if len(TOTAL_SCANNED_ERRORS_AS_CLASS[model]) > 0:
            print('MODEL',MODEL_TYPE[model],'TOTAL_SCANNED_ERRORS_AS_CLASS',TOTAL_SCANNED_ERRORS_AS_CLASS[model],'\n',
                  '!!! CLASS ERROR % !!!:',np.sum(TOTAL_SCANNED_ERRORS_AS_CLASS[model])/np.shape(TOTAL_SCANNED_ERRORS_AS_CLASS)[1]*100,'\n',
                  'TOTAL UNKNOWNS %',np.sum(TOTAL_UNKNOWNS[model])/np.shape(TOTAL_UNKNOWNS)[1]*100)
    
        ERRORSCORES[model,everyclass] = np.sum(TOTAL_SCANNED_ERRORS_AS_CLASS[model])/np.shape(TOTAL_SCANNED_ERRORS_AS_CLASS)[1]*100


    print('ERROR SCORES:',ERRORSCORES,'\n')

    for i in forsolutionsarray:
        sumentr = 0
        for m in range(len(MODEL_TYPE)):
            if ERRORSCORES[m,i] < 0:
                continue
            sumentr += ( ERRORSCORES[m,i]/100 * log(1/(ERRORSCORES[m,i]/100)) )
        ENTROPY[i] = min(ERRORSCORES[:,i])/100 * np.exp(np.cosh(-sumentr))
    
    
    
print('OUR MODIFIED ENTROPY ON ERRORSCORES OF THE MODELS:',ENTROPY,'\n') # THE LOWER, THE MORE CONSISTENT