# Códigos para apresentação sobre Análise de Sensibilidade 

# Default Python imports
import os
import pathlib
import csv
import sys
import stat
import subprocess
import datetime
import math
import itertools

# Other imports
import numpy as np
from numpy import random as rng
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import scipy.integrate as integrate

# Local imports
import SensitivityAnalysisModels as SAModels

# Constantes para os modelos
REFERENCE_VALUES_HARRIS_EOQ = [1230, 0.0135, 2.15]                  # int, float, float
INTERESSE_ANUAL_HARRIS_EOQ = 240                                    # 2 * 12 meses/ano * 10% ao mês no modelo de juros simples
VAR_LABELS = ['Saída Mensal','Custo de pedido','Custo de armazenamento']

REFERENCE_RESULTS_HARRIS_EOQ = SAModels.HarrisEOQ(REFERENCE_VALUES_HARRIS_EOQ[0],
                                                  REFERENCE_VALUES_HARRIS_EOQ[1],
                                                  REFERENCE_VALUES_HARRIS_EOQ[2],
                                                  INTERESSE_ANUAL_HARRIS_EOQ
                                                  )

OAT_SIM_SIZE = 10000

HARRIS_VAR = 0.1

H_VMIN = 1-HARRIS_VAR
H_VMAX = 1+HARRIS_VAR

H_GLOBAL_VMIN = [0.9,0.7,0.9]
H_GLOBAL_VMAX = [1.1,1.3,1.1]

MODEL_INPUT_RANGE = np.linspace(start=(H_VMIN-1), stop=(H_VMAX-1), num=OAT_SIM_SIZE+1, endpoint=True)
GLOBAL_SIM_SIZE = 10000

# Functions for OAT Analysis

def modeloHarrisEOQ(inpM, inpS, inpC): return SAModels.HarrisEOQ(inpM, inpS, inpC,INTERESSE_ANUAL_HARRIS_EOQ)


def createTornadoDiagram(*resultModels, resultModelRef, fName):     # inp -> um array que contém arrays do Numpy & o resultado de referência
                                                                    # se espera que (resultmodels[i])[0] seja o valor neg e (resultmodels[i])[-1] seja o pos
                                                                    # out -> gera um gráfico com o Tornado Diagram em fName
    numVars = len(resultModels)
    negDeltas = np.zeros(numVars)
    posDeltas = np.zeros(numVars)
    ordDeltas = np.zeros(numVars)

    for i in range(numVars):
        negDeltas[i] = (resultModels[i])[0] - resultModelRef
        posDeltas[i] = (resultModels[i])[-1] - resultModelRef
        ordDeltas[i] = np.abs(posDeltas[i] - negDeltas[i])

    ordDeltas = np.argsort(ordDeltas)

    maxDelta = np.max(np.abs(np.concatenate([negDeltas,posDeltas])))

    fig, ax = plt.subplots(layout = 'constrained')
    varN = []
    for i in range(numVars):
        k = ordDeltas[numVars - i - 1]
        ax.broken_barh(
            [(0, negDeltas[k]), (0, posDeltas[k])],
            (i + 0.6, 0.8),
            facecolors=['darkgreen', 'limegreen'],  # Try different colors if you like
            edgecolors=['black', 'black'],
            linewidth=0.8,
        )
        varN.append('$x_' + str(k+1) + '$')

    ax.set_xlim(-1.25*maxDelta,1.25*maxDelta)
    ax.set_ylim(0,numVars+1.2)                                
    ax.axvline(0, color = 'black', lw = 1.5)
    ax.set_yticks((1+np.arange(numVars)), labels = varN)
    ax.set_xlabel('$\Delta y$')
    ax.set_title('Tornado Diagram')
    legendHiddenLine = [plt.Line2D([0], [0], color='limegreen', lw=6, label = '$\Delta x_{+}$'),
                        plt.Line2D([0], [0], color='darkgreen', lw=6, label = '$\Delta x_{-}$')]
    ax.legend(handles = legendHiddenLine, loc = 'upper left')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(left = False, width = 1.5)
    fig.savefig(fName)
    plt.close(fig)
    saveToLog(fName,'write')

    return

def createSpiderPlot(*resultModels, modelRange, resultModelRef = 0, fName):                 # modelRange -> se espera que tenha o mesmo tamanho que cada um dos elementos de 
                                                                                            #               resultModel e que resultModel seja entre (-1;+inf](porcentagem)
    numInputs = len(resultModels)

    fig, ax = plt.subplots(layout = 'constrained')
    for i in range(numInputs):
        ax.plot(modelRange*100,((resultModels[i])-resultModelRef), linewidth=2.0, label=('$x_' + str(i+1) + '$'))

    ax.set_xlabel('Input % variation')
    ax.set_ylabel('$\Delta y$')
    ax.legend(loc = 'upper left')
    ax.axvline(0,color = 'black', lw = 1.5, linestyle = ':')
    ax.axhline(0,color = 'black', lw = 1.5, linestyle = ':')
    ax.tick_params(width = 1.5)
    ax.set_title('Spider Plot')
    fig.savefig(fName)
    plt.close(fig)
    saveToLog(fName,'write')

    return

# Functions for Local Analysis

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"         # É importantíssimo que se gere os iteráveis nesta ordem de aumento de len()
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def calculateSensitivityIndexes(funcModel,modelInputReference,modelInputVars):          
    
    numVars = len(modelInputReference)

    valuesSI = dict()
    
    valuesSI[()] = funcModel(*modelInputReference)

    iterableInputs = powerset(list(range(numVars)))                             
    for k in iterableInputs:                                                    # gera todos os subconjutos possíves de [0,1,...,numVars] = U
        if k not in valuesSI:                                                   # retorna um k que é um subconjunto de U(e.g. k = (0,2,numVars))
            tempInput = list(modelInputReference)
            for i in k:                                                         # para todos os valores dentro de k soma a variação(e.g. tempInput = modelInputReference + modelInputVars[0,2,numVars])
                tempInput[i] += modelInputVars[i]            
            valuesSI[k] = funcModel(*tempInput)                                 # roda o código para os valores variados
            for lowerSet in powerset(k):                                        # depois gera todos os subconjuntos possíveis de k(e.g. (),(0,),(2,),(numVars,),(0,2,),(0,numVars),(2,numVars),(0,2,numVars))
                if lowerSet != k:                                               # e caso não seja ele mesmo, subtrai o valor do subconjunto
                    valuesSI[k] -= valuesSI[lowerSet]

    returnSensitivities = [0]*numVars                                           # Retorna um array de tamanho numVars composto por tuplas de 3 elementos, 
    for i in range(numVars):                                                    # onde as tuplas são: 1º) Si; 2º) STi; e 3º) SIi.
        ret0 = valuesSI[(i,)]
        ret1 = 0
        for k in powerset(list(range(numVars))):
            if i in k:
                ret1 += valuesSI[k]
        ret2 = ret1 - ret0
        returnSensitivities[i] = (ret0,
                                  ret1,
                                  ret2
                                  )

    return returnSensitivities

def createGeneralizedTornadoDiagram(posSensitivities,negSensitivities,fName):

    numVars = len(posSensitivities)
    ordDeltas = []
    for i in range(numVars):
        ordDeltas.append(np.abs((posSensitivities[i])[1] - (negSensitivities[i])[1]))
    ordDeltas = np.argsort(ordDeltas)
    maxDelta = np.max(np.abs(np.concatenate([negSensitivities,posSensitivities])))

    fig, ax = plt.subplots(layout = 'constrained')
    varN = []
    for i in range(numVars):
        k = ordDeltas[numVars - i - 1]
        facecol = ['limegreen','darkgreen','royalblue','navy','gold','darkorange']      # Escuro -> pos, Claro -> neg.
        for m in range(3):
            ax.broken_barh(
                [(0, (negSensitivities[k])[m]), (0, (posSensitivities[k])[m])],         # Caso o valorneg < valorpos a barra do neg vai ficar escondida
                (i + 0.6 + 0.3*m, 0.2),
                facecolors=facecol[2*m:2*m+2],  
                edgecolors=['black', 'black'],
                linewidth=0.8,
            )
        varN.append('$x_' + str(k+1) + '$')

    ax.set_xlim(-1.25*maxDelta,1.25*maxDelta)
    ax.set_ylim(0,numVars+0.6)                                
    ax.axvline(0, color = 'black', lw = 1.5)
    ax.set_yticks((1+np.arange(numVars)), labels = varN)
    ax.set_xlabel('$\Delta y$')
    ax.set_title('Generalized Tornado Diagram')
    legendHiddenLine = [plt.Line2D([0], [0], color='darkgreen', lw=4, label = '$S_{+}$'),
                        plt.Line2D([0], [0], color='limegreen', lw=4, label = '$S_{-}$'),
                        plt.Line2D([0], [0], color='navy',      lw=4, label = '$S_{T+}$'),
                        plt.Line2D([0], [0], color='royalblue', lw=4, label = '$S_{T-}$'),
                        plt.Line2D([0], [0], color='darkorange',lw=4, label = '$S_{I+}$'),
                        plt.Line2D([0], [0], color='gold',      lw=4, label = '$S_{I-}$')]
    ax.legend(handles = legendHiddenLine, loc = 'upper left')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(left = False, width = 1.5)
    fig.savefig(fName)
    plt.close(fig)
    saveToLog(fName,'write')

    return

def calculateElasticity(*derivatives, resultModelRef, modelInputReference): # derivatives -> um array que contém arrays do Numpy, com os elementos sendo as derivadas. se espera que (derivatives[i])[len((derivatives[i])//2)] seja aproximadamente dg(x0)
                                                                            # resultModelRef-> o resultado de referência em x0
                                                                            # modelInputReference -> x0
                                                                            # out -> gera um gráfico com o Tornado Diagram em fName

    numVars = len(modelInputReference)
    arrElasticity = [0] * numVars
    for i in range(numVars):
        arrElasticity[i] = (derivatives[i])[len(derivatives[i])//2] * modelInputReference[i] / resultModelRef

    return arrElasticity

def calculateDiffImportanceMeasure(varElasticities):
    if type(varElasticities) != np.ndarray:
        varElasticities = np.array(varElasticities)
    return varElasticities/np.sum(varElasticities)

# Functions for Global Analysis

def createConditionalDensitiesPlot(resultBins,totalResults,fName):

    numBins = len(resultBins)
    fig = sns.displot(data=totalResults,kind='kde',color='black', bw_adjust=5, lw = 3.0)
    sns.set_palette(palette="husl", n_colors=numBins)
    for i in range(numBins):
        sns.kdeplot(data=(np.array(resultBins[i]).T)[0],bw_adjust=5,alpha=0.7, lw=1.5)   
    fig.set_axis_labels('QOP','Densidade')
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_title('Densidade Condicional com {} partições'.format(numBins))
#    fig.set(xlim=(np.mean(totalResults)*H_VMIN*0.7,np.mean(totalResults)*H_VMAX*1.3))
    fig.savefig(fName)
    

    saveToLog(fName,'write')

    return

# Main Analysis Codes

def runOATAnalysis():   
    
    modelInput0 = np.linspace(start=H_VMIN*REFERENCE_VALUES_HARRIS_EOQ[0], stop=H_VMAX*REFERENCE_VALUES_HARRIS_EOQ[0], num=OAT_SIM_SIZE+1, endpoint=True)
    modelInput1 = np.linspace(start=H_VMIN*REFERENCE_VALUES_HARRIS_EOQ[1], stop=H_VMAX*REFERENCE_VALUES_HARRIS_EOQ[1], num=OAT_SIM_SIZE+1, endpoint=True)
    modelInput2 = np.linspace(start=H_VMIN*REFERENCE_VALUES_HARRIS_EOQ[2], stop=H_VMAX*REFERENCE_VALUES_HARRIS_EOQ[2], num=OAT_SIM_SIZE+1, endpoint=True)


    try:
        resultsHarrisEOQ0 = np.loadtxt('./MonteCarloSims/HarrisOAT_outfor0.txt', dtype = np.float64)
        saveToLog('./MonteCarloSims/HarrisOAT_outfor0.txt', 'read')
    except:
        resultsHarrisEOQ0 = modeloHarrisEOQ(modelInput0,REFERENCE_VALUES_HARRIS_EOQ[1],REFERENCE_VALUES_HARRIS_EOQ[2])
        np.savetxt('./MonteCarloSims/HarrisOAT_outfor0.txt',resultsHarrisEOQ0)
        saveToLog('./MonteCarloSims/HarrisOAT_outfor0.txt', 'write')
    try:
        resultsHarrisEOQ1 = np.loadtxt('./MonteCarloSims/HarrisOAT_outfor1.txt', dtype = np.float64)
        saveToLog('./MonteCarloSims/HarrisOAT_outfor1.txt', 'read')
    except:
        resultsHarrisEOQ1 = modeloHarrisEOQ(REFERENCE_VALUES_HARRIS_EOQ[0],modelInput1,REFERENCE_VALUES_HARRIS_EOQ[2])
        np.savetxt('./MonteCarloSims/HarrisOAT_outfor1.txt',resultsHarrisEOQ1)
        saveToLog('./MonteCarloSims/HarrisOAT_outfor1.txt', 'write')
    try:
        resultsHarrisEOQ2 = np.loadtxt('./MonteCarloSims/HarrisOAT_outfor2.txt', dtype = np.float64)
        saveToLog('./MonteCarloSims/HarrisOAT_outfor2.txt', 'read')
    except:
        resultsHarrisEOQ2 = modeloHarrisEOQ(REFERENCE_VALUES_HARRIS_EOQ[0],REFERENCE_VALUES_HARRIS_EOQ[1],modelInput2)
        np.savetxt('./MonteCarloSims/HarrisOAT_outfor2.txt',resultsHarrisEOQ2)
        saveToLog('./MonteCarloSims/HarrisOAT_outfor2.txt', 'write')

    createTornadoDiagram(*[resultsHarrisEOQ0,resultsHarrisEOQ1,resultsHarrisEOQ2],resultModelRef = REFERENCE_RESULTS_HARRIS_EOQ, fName = './Results/OAT/TornadoDiagram.png')
    createSpiderPlot(*[resultsHarrisEOQ0,resultsHarrisEOQ1,resultsHarrisEOQ2], modelRange = MODEL_INPUT_RANGE, resultModelRef = REFERENCE_RESULTS_HARRIS_EOQ, fName = './Results/OAT/SpiderPlot.png')

    fig, axes = plt.subplots(1,3, layout = 'constrained', sharey=True)
    axes[0].plot(modelInput0, resultsHarrisEOQ0, 'black', linewidth=2.0)
    axes[0].set_xlabel('Saída Mensal')
    axes[0].set_ylabel('QOP')
    axes[1].plot(modelInput1, resultsHarrisEOQ1, 'black', linewidth=2.0)
    axes[1].set_xlabel('Custo de pedido')
    axes[2].plot(modelInput2, resultsHarrisEOQ2, 'black', linewidth=2.0)
    axes[2].set_xlabel('Custo de armazenamento')

    for i in range(3):
        ax = axes[i]
        plt.setp(ax, ylim = (REFERENCE_RESULTS_HARRIS_EOQ*H_VMIN,REFERENCE_RESULTS_HARRIS_EOQ*H_VMAX))
        ax.axhline(REFERENCE_RESULTS_HARRIS_EOQ, color = 'red', linestyle = ':', linewidth = 1)
        ax.axvline(REFERENCE_VALUES_HARRIS_EOQ[i], color = 'red', linestyle = ':', linewidth = 1)
    
    fig.suptitle('Gráficos de Harris EOQ')
    plt.savefig('./Results/OAT/graphsVar.png')
    plt.close(fig)
    saveToLog('./Results/OAT/graphsVar.png','write')
    
    return 0

def runLocalAnalysis():

    try:
        resPosSensitivities = np.loadtxt('./MonteCarloSims/HarrisLocal_posSens.txt', dtype = np.float64)
        resNegSensitivities = np.loadtxt('./MonteCarloSims/HarrisLocal_negSens.txt', dtype = np.float64)
        saveToLog('./MonteCarloSims/HarrisLocal_posSens.txt & ./MonteCarloSims/HarrisLocal_negSens.txt', 'read')
    except OSError:
        resPosSensitivities = calculateSensitivityIndexes(modeloHarrisEOQ,REFERENCE_VALUES_HARRIS_EOQ,[HARRIS_VAR*item for item in REFERENCE_VALUES_HARRIS_EOQ])
        resNegSensitivities = calculateSensitivityIndexes(modeloHarrisEOQ,REFERENCE_VALUES_HARRIS_EOQ,[-HARRIS_VAR*item for item in REFERENCE_VALUES_HARRIS_EOQ])#
        np.savetxt('./MonteCarloSims/HarrisLocal_posSens.txt', resPosSensitivities,header='First Order\t\t\tTotal\t\t\tInteraction')
        np.savetxt('./MonteCarloSims/HarrisLocal_negSens.txt', resNegSensitivities,header='First Order\t\t\tTotal\t\t\tInteraction')
        saveToLog('./MonteCarloSims/HarrisLocal_posSens.txt & ./MonteCarloSims/HarrisLocal_negSens.txt', 'write')

    createGeneralizedTornadoDiagram(resPosSensitivities,resNegSensitivities,'./Results/Local/GeneralizedTornadoDiagram.png')

    resultsHarrisEOQ0 = np.loadtxt('./MonteCarloSims/HarrisOAT_outfor0.txt', dtype = np.float64)
    resultsHarrisEOQ1 = np.loadtxt('./MonteCarloSims/HarrisOAT_outfor1.txt', dtype = np.float64)
    resultsHarrisEOQ2 = np.loadtxt('./MonteCarloSims/HarrisOAT_outfor2.txt', dtype = np.float64)
    saveToLog('./MonteCarloSims/HarrisOAT_outfor0.txt, ./MonteCarloSims/HarrisOAT_outfor1.txt & ./MonteCarloSims/HarrisOAT_outfor2.txt', 'read')

    propStep = np.diff(MODEL_INPUT_RANGE)
    derivativesHarrisEOQ0 = np.diff(resultsHarrisEOQ0) / (propStep*REFERENCE_VALUES_HARRIS_EOQ[0])
    derivativesHarrisEOQ1 = np.diff(resultsHarrisEOQ1) / (propStep*REFERENCE_VALUES_HARRIS_EOQ[1])   
    derivativesHarrisEOQ2 = np.diff(resultsHarrisEOQ2) / (propStep*REFERENCE_VALUES_HARRIS_EOQ[2])

    createSpiderPlot(*[derivativesHarrisEOQ0,derivativesHarrisEOQ1,derivativesHarrisEOQ2],modelRange=MODEL_INPUT_RANGE[0:-1],fName='./Results/Local/DerivativesSpiderPlot.png')

    arrElast = np.array(calculateElasticity(*[derivativesHarrisEOQ0,derivativesHarrisEOQ1,derivativesHarrisEOQ2],resultModelRef=REFERENCE_RESULTS_HARRIS_EOQ,modelInputReference=REFERENCE_VALUES_HARRIS_EOQ),dtype=np.float64)
    np.savetxt('./Results/Local/Elasticity.txt', arrElast)
    saveToLog('./Results/Local/Elasticity.txt', 'write')
    arrDIM = calculateDiffImportanceMeasure(arrElast)
    np.savetxt('./Results/Local/DIM.txt', arrDIM)
    saveToLog('./Results/Local/DIM.txt', 'write')

    return 0

def runGlobalAnalysis():

    rng = np.random.default_rng()
    try:
        montecarloInputs = np.loadtxt('./MonteCarloSims/HarrisMonteCarlo_Inputs.txt', dtype=np.float64)
        saveToLog('./MonteCarloSims/HarrisMonteCarlo_Inputs.txt', 'read')
        if montecarloInputs.shape[0] != GLOBAL_SIM_SIZE: raise FileNotFoundError
    except FileNotFoundError:
        montecarloInputs = [REFERENCE_VALUES_HARRIS_EOQ[i]*rng.uniform(H_GLOBAL_VMIN[i],H_GLOBAL_VMAX[i],GLOBAL_SIM_SIZE) 
                            for i in range(len(REFERENCE_VALUES_HARRIS_EOQ))]
        montecarloInputs = np.array(montecarloInputs).T
        np.savetxt('./MonteCarloSims/HarrisMonteCarlo_Inputs.txt',montecarloInputs)
        saveToLog('./MonteCarloSims/HarrisMonteCarlo_Inputs.txt', 'write')

    try:
        montecarloResults = np.loadtxt('./MonteCarloSims/HarrisMonteCarlo_Results.txt', dtype=np.float64)
        saveToLog('./MonteCarloSims/HarrisMonteCarlo_Results.txt', 'read')
        if len(montecarloResults) != GLOBAL_SIM_SIZE: raise FileNotFoundError
    except FileNotFoundError:
        montecarloResults = modeloHarrisEOQ(*montecarloInputs.T)
        np.savetxt('./MonteCarloSims/HarrisMonteCarlo_Results.txt',montecarloResults)
        saveToLog('./MonteCarloSims/HarrisMonteCarlo_Results.txt', 'write')        

    sns.displot(montecarloResults, kind="kde", bw_adjust=10)
    sns.displot(montecarloResults,kind="ecdf")

    # Linear Regression Coeff
    tempA = np.concatenate([montecarloInputs,np.ones((len(montecarloResults),1))],axis=1)

    lrCoefficients = (np.linalg.lstsq(tempA,montecarloResults,rcond=None))[0]

    inpVariances = np.var(montecarloInputs,axis=0)
    inpMeans = np.mean(montecarloInputs,axis=0)
    inpSTDev = np.sqrt(inpVariances)
    outVariance = np.var(montecarloResults)
    outMean = np.mean(montecarloResults)
    outSTDev = np.sqrt(outVariance)

    covInput = np.concatenate([montecarloResults.reshape((len(montecarloResults),1)), montecarloInputs],axis=1)
    covMatrix = np.cov(covInput,rowvar=False)

    if True:                # Calculate SRC & PEAR
        SRC = [0]*len(inpVariances)
        for i in range(len(inpVariances)):
            SRC[i] = lrCoefficients[i]*np.sqrt(inpVariances[i]/outVariance)
        np.savetxt('./Results/Global/SRC.txt',SRC)

        PEAR = [0]*len(inpVariances)
        for i in range(len(inpVariances)):
            PEAR[i] = covMatrix[0][i+1]/np.sqrt(outVariance*inpVariances[i])
        np.savetxt('./Results/Global/PEAR.txt',PEAR)


#    curva ecdf/integral = np.cumsum(outHist*np.diff(outBin_edges))


    outHistBins = np.histogram(montecarloResults, density=True, bins='fd')
    inHistBins = [np.histogram(montecarloInputs[:,i], density=True, bins=8) for i in range(montecarloInputs.shape[1])]
    orderedRelations = [0]*montecarloInputs.shape[1]

    if True:
        numVars = montecarloInputs.shape[1]
        arrlinds = []
        sctLabels = ['Saída Mensal','Custo de pedido','Custo de armazenamento']
        fig, axes = plt.subplots(1,3, layout = 'constrained', sharey=True)
        axes[0].set_ylabel('QOP')
        for i in range(numVars):
            tempInputs = montecarloInputs[:,i]
            arrlinds = np.argsort(tempInputs)
            tempInputs = tempInputs[arrlinds[::]]
            tempOutputs = montecarloResults[arrlinds[::]]
            orderedRelations[i] = np.concatenate([tempOutputs.reshape(len(tempOutputs),1),tempInputs.reshape(len(tempOutputs),1)],axis=1)

            axes[i].scatter(tempInputs, tempOutputs, s = 0.35, alpha = 0.5)
            axes[i].set_xlabel(sctLabels[i])
            plt.setp(axes[i], ylim = (REFERENCE_RESULTS_HARRIS_EOQ*H_VMIN*0.8,REFERENCE_RESULTS_HARRIS_EOQ*H_VMAX*1.2))
        fig.suptitle('Harris EOQ Scatterplots')
        fig.savefig('./Results/Global/Scatterplots.png')
        plt.close(fig)

    subPartitions = []
    for i in range(len(orderedRelations)):
        np.savetxt('./MonteCarloSims/HarrisMonteCarlo_RelationsIO'+str(i)+'.txt',orderedRelations[i])

        tempOrder = orderedRelations[i]

        Bins = (inHistBins[i])[1]
        Bins = Bins[1:]        
        rel_index = 0
        subPartitions.append([])
        for index in range(len(tempOrder)):
            if (tempOrder[rel_index])[1] > Bins[0]:
                tempSubPartition, tempOrder = np.split(tempOrder,[rel_index],axis=0)
                subPartitions[i].append(tempSubPartition)
                Bins = Bins[1:]
                rel_index = 0
            else: rel_index += 1
        

    # densidades condicionais para os bins criados com os histogramas
    for i in range(len(subPartitions)):
        createConditionalDensitiesPlot(subPartitions[i],montecarloResults,'./Results/Global/ConditionalDensity'+str(i)+'.png')

    # densidades condicionais com xi = x_ref
    for i in range(len(REFERENCE_VALUES_HARRIS_EOQ)):

        ValueInputs = [REFERENCE_VALUES_HARRIS_EOQ[i] 
                       if index == i else 
                       REFERENCE_VALUES_HARRIS_EOQ[index]*rng.uniform(H_GLOBAL_VMIN[index],H_GLOBAL_VMAX[index],GLOBAL_SIM_SIZE) 
                       for index in range(len(REFERENCE_VALUES_HARRIS_EOQ))]

        ValueResults = modeloHarrisEOQ(*ValueInputs)

        sns.displot(pd.DataFrame(data={'conditional':ValueResults, 'base':montecarloResults}), 
                    kind = 'kde', color = 'black', bw_adjust = 4, fill=True, multiple = 'layer', legend = False)
        ax = plt.gca()
        tickerFormatter = ticker.ScalarFormatter( useMathText=True )
        tickerFormatter.set_scientific(True)
        tickerFormatter.set_powerlimits((-2,3))
        ax.yaxis.set_major_formatter(tickerFormatter)
        plt.legend(loc='upper right', labels = [r'$f_{Y}(y)$', r'$f_{Y|X=x_{ref}}(y)$'])

        ax.set_title('Comparação de densidades \npara a {}ª variável'.format(str(i)))

        plt.savefig('./Results/Global/FixedPointForVar'+str(i)+'.png')
        saveToLog('./Results/Global/FixedPointForVar'+str(i)+'.png','write')

    # phi_a(x_a)
    S_ind = []
    for i in range(len(REFERENCE_VALUES_HARRIS_EOQ)):
        varSpace = REFERENCE_VALUES_HARRIS_EOQ[i] * np.linspace(H_GLOBAL_VMIN[i],H_GLOBAL_VMAX[i], num = 1000, endpoint = False)
        resultantMeans = []
        for varValue in varSpace:
            resultantMeans.append(np.mean(
                                    modeloHarrisEOQ(*[
                                        varValue 
                                        if index == i else 
                                        REFERENCE_VALUES_HARRIS_EOQ[index]*rng.uniform(H_GLOBAL_VMIN[index],H_GLOBAL_VMAX[index],GLOBAL_SIM_SIZE*10) 
                                        for index in range(len(REFERENCE_VALUES_HARRIS_EOQ))])
                                        )
                              )
        fig, ax = plt.subplots(layout='constrained')
        ax.set_xlabel(VAR_LABELS[i])
        ax.set_ylabel('QOP')
#        ax.set_title(r'$\varphi_{\alpha}(x_{\alpha})$')
        ax.set_title(r'E[$Y|X_{\alpha}=x_{\alpha}$]')
        ax.plot(varSpace,resultantMeans)

        fig.savefig('./Results/Global/Phi_'+str(i)+'.png')
        saveToLog('./Results/Global/Phi_'+str(i)+'.png','write')
        plt.close(fig)

        S_ind.append(np.var(resultantMeans)/outVariance)

    print('{} \t Sum:{}'.format(S_ind,np.sum(S_ind)))

#   Density based method
    reference_KDE = stats.gaussian_kde(montecarloResults)
    deltas = [0]*len(REFERENCE_VALUES_HARRIS_EOQ)
    for i in range(len(REFERENCE_VALUES_HARRIS_EOQ)):
        varSpace = REFERENCE_VALUES_HARRIS_EOQ[i] * rng.uniform(H_GLOBAL_VMIN[i],H_GLOBAL_VMAX[i], size = 100)
        innerStatistics = []
        for varValue in varSpace:
            harrisResults = modeloHarrisEOQ(*[
                                    varValue 
                                    if index == i else 
                                    REFERENCE_VALUES_HARRIS_EOQ[index]*rng.uniform(H_GLOBAL_VMIN[index],H_GLOBAL_VMAX[index],GLOBAL_SIM_SIZE) 
                                    for index in range(len(REFERENCE_VALUES_HARRIS_EOQ))])
            var_KDE = stats.gaussian_kde(harrisResults)
            innerStatistics.append(integrate.quad(lambda y: np.abs(reference_KDE.evaluate(y)-var_KDE.evaluate(y)),3000,outMean,limit=50)[0] +
                                   integrate.quad(lambda y: np.abs(reference_KDE.evaluate(y)-var_KDE.evaluate(y)),outMean,11000,limit=50)[0]
                                   )
        deltas[i] = 0.5*np.mean(innerStatistics)

    print(deltas)

    return 0

# Main

def main():
    startTime = datetime.datetime.now()
    createLocalStorage(pSubFolders = ['./MonteCarloSims/','./Results/OAT/','./Results/Local/','./Results/Global/'])

    runOATAnalysis()

    runLocalAnalysis()

    runGlobalAnalysis()

    saveToLog('Code execution took {}'.format(datetime.datetime.now()-startTime),'lone')

    return 0


# Other Functions

def createLocalStorage(pName: str = './SensibilityAnalysisStorage', pSubFolders: list[str] = None):
    try:
        os.chdir(pName)
    except FileNotFoundError:
        print('Storage path not found.')
        os.makedirs(pName)
        os.chdir(pName)
        print('Created storage path to ' + os.getcwd())
        logBegin.has_run = True          # Don't log if just created folder
    
    if pSubFolders != None:
        if type(pSubFolders) == str:
            pSubFolders = [pSubFolders]
        for fSubFolder in pSubFolders:
            if fSubFolder[0] != '.': fSubFolder = '.' + fSubFolder
            if (os.access(fSubFolder, os.R_OK | os.W_OK)):
                pass
            elif not os.access(fSubFolder, os.F_OK):
                os.makedirs(fSubFolder)
            else:
                os.chmod(fSubFolder, stat.S_IWRITE | stat.S_IEXEC)

        try:
            os.chmod('storageLog.sff', stat.S_IWRITE|stat.S_IREAD)
        except FileNotFoundError: 
            temp = open('storageLog.sff', 'w')                                                         # .sff -> Storage Folder Log File
            temp.close()
        with open('storageLog.sff', 'r+', newline='\n', encoding='utf-8') as storageLog:
            hasWrittenToLog = False
            storageLogLines = storageLog.read().splitlines()
            currentTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            if not storageLogLines:
                storageLog.write('Created folder(s) at {}\n'.format(currentTime))
                subprocess.run(["attrib","+H","storageLog.sff"])
                hasWrittenToLog = True

            for pSingleSubFolders in pSubFolders:
                if pSingleSubFolders[0] != '.': pSingleSubFolders = '.' + pSingleSubFolders
                if pSingleSubFolders not in storageLogLines:
                    if not hasWrittenToLog:
                        storageLog.write('Written to log at {}\n'.format(currentTime))
                        hasWrittenToLog = True
                    storageLog.write(pSingleSubFolders + '\n' )
            
        os.chmod('storageLog.sff', stat.S_IREAD)

    return

def saveToLog(fText: str, log_case: str = None):
    os.chmod('storageLog.sff', stat.S_IWRITE|stat.S_IREAD)
    currentTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('storageLog.sff', 'a', newline='\n', encoding='utf-8') as log:
        logBegin(log,currentTime)
        match log_case:
            case None:
                log.write('{} at {}\n'.format(fText,currentTime))
            case 'read' :
                log.write('Read {} at {}\n'.format(fText,currentTime))
            case 'write':
                log.write('Wrote to {} at {}\n'.format(fText,currentTime))
            case 'lone':
                log.write('{}\n'.format(fText))            
            case _:
                log.write('{} {} at {}\n'.format(log_case,fText,currentTime))
    os.chmod('storageLog.sff', stat.S_IREAD)
    return

def once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

@once
def logBegin(fLog,cTime): fLog.write('Began logging at {}\n'.format(cTime))



if __name__ == '__main__': 
    sys.exit(main())
    


