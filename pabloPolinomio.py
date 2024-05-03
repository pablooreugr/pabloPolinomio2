#Voy a crear un codigo de python para calcular las regresiones
# Por minimos cuadrados para cualquier tipo de resultados

import pandas as pd #pandas sirve para trabajar con los datos del csv
import numpy as np #Sirve para controlar matrices
import scipy.stats as stats #Me sirve para saber los valores de chi^2
import matplotlib.pyplot as plt #Me sirve para crear las graficas como tal
import string #Me sirve para tener el abecedario
import csv # Sirve para crear un archivo csv

#El archivo tiene que estar escrito de manera que todo este separado por comas, sin ningun espacio
#Y que la primera fila sea: x,dx,y,dy

#FUNCION PARA CALCULAR UN PUNTO DE LA FUNCION CUANDO ESTE SE HAYA CALCULADO
def FuncionPolinomica(x, gradoDelPolinomio, listaDelPolinomio, listaDeCoeficientes):
    indiceCoeficiente = 0
    resultado = 0
    for i in range(gradoDelPolinomio+1):
        if(listaDelPolinomio[i] == True):
            resultado += listaDeCoeficientes[indiceCoeficiente][0] * (x**i)
            indiceCoeficiente += 1
    
    return resultado


def FuncionPolinomicaLista(x, gradoDelPolinomio, listaDelPolinomio, listaDeCoeficientes):
    indiceCoeficiente = 0
    resultado = []
    for i in range(gradoDelPolinomio+1):
        if(listaDelPolinomio[i] == True):
            resultado.append(listaDeCoeficientes[indiceCoeficiente][0] * (x[i]**i))
            indiceCoeficiente += 1
    
    return resultado

def hazmeLaRegresion(nombreArchivo, grado):
    datos = pd.read_csv(nombreArchivo, delimiter=',', header=0, names=['x', 'dx', 'y', 'dy'])
    print(f'----------- Se han abiero los datos de {nombreArchivo} -----------')
    #Aqui se puede elegir el tipo de ajuste del que se quiere utilizar de polinomio
    gradoMaxDelPolinomio = grado
    tipoDePolinomio = []
    print("Escribe S si quiere aceptarlo, y N si no.")
    print()
    puedeSalir = False
    for i in range(gradoMaxDelPolinomio+1):
        puedeSalir = False
        while(puedeSalir == False):
            valorQuerido = input(f"Quieres un valor x^({i}): ")
            valorQuerido = valorQuerido.lower()
            if(valorQuerido == 's' or valorQuerido == "n"):
                if(valorQuerido == 's'):
                    tipoDePolinomio.append(True)
                else:
                    tipoDePolinomio.append(False)
                puedeSalir = True

#Cojo vectores de los datos que necesite
    vectorDatosX = datos['x']
    vectorDatosDX = datos['dx']
    vectorDatosY = datos['y']
    vectorDatosDY = datos['dy']


#Aquí creare la matriz de Diseño
#Aqui calculo cuantas columnas tendra la matriz dependiendo del polinomio
    numColumnasMaD = 0
    for i in tipoDePolinomio:
        if(i == True):
            numColumnasMaD += 1

#Creo una matriz de las filas y columnas que necesito lleno de ceros
    matrizDiseño = np.zeros((len(vectorDatosX),numColumnasMaD))

#Relleno la matriz de los distintos datos que necesito
    numColumna = 0

    for i in range(len(tipoDePolinomio)):
        if(tipoDePolinomio[i] == True):
            for j in range(len(vectorDatosX)):
                matrizDiseño[j, numColumna] = vectorDatosX[j]**i
        
            numColumna += 1

    #Creo la matriz de puntos de y (en columnas)
    matrizY = np.array(vectorDatosY).reshape(-1, 1) #Convierto en una matriz fila y luego en columna

    #A PARTIR DE AQUI CREO LA MATRIZ INVERSA DE COVARIANZAS

    numFilas = len(vectorDatosDY)
    #Creo la matriz de varianzas
    matrizCovarianzas = np.zeros((numFilas, numFilas))
    for i in range(numFilas):
        matrizCovarianzas[i, i] = vectorDatosDY[i]**2

    #Hago la inversa
    inversaMatrizCovarianzas = np.linalg.inv(matrizCovarianzas)


    #YA TENEMOS TODOS LOS CALCULOS, A PATIR DE AQUI ES EL ALGORITMO DE LA ECUACION MATRICIAL
    # N = (A^t * W * A)^(-1) * A^t * W * Y
    #Nos interesa los valores de la diagonal de la primera inversa, ya que seran los errores
    #de cada cociente del resultado

    #Calcularemos primero esa inversa
    productoDiseñoCovarianzas = np.dot(np.transpose(matrizDiseño), inversaMatrizCovarianzas)
    matrizInversaPrimerProducto = np.linalg.inv(np.dot(productoDiseñoCovarianzas, matrizDiseño))

    #Sacaremos los errores y los pondremos en una lista
    listaErrores = []
    for i in range(matrizInversaPrimerProducto.shape[0]):
        listaErrores.append(matrizInversaPrimerProducto[i, i])


    #AHORA SACAREMOS LOS COEFICIENTES
    matrizCoeficientes = np.dot(matrizInversaPrimerProducto, np.dot(productoDiseñoCovarianzas, matrizY))

    #Para mostrar los coeficientes voy a crear una lista con el abecedario
    abecedario = list(string.ascii_lowercase)


    #Ahora lo evaluaremos con CHI^2 para saber si el ajuste es lo suficientemente bueno
    #chiExperimental = 0
    #
    #for i in range(len(vectorDatosY)):
    #    chiExperimental += inversaMatrizCovarianzas[i, i] * ((vectorDatosY[i] - FuncionPolinomica(vectorDatosX[i], gradoMaxDelPolinomio, tipoDePolinomio, matrizCoeficientes.tolist()))**2)
    #
    def test_chi_squared():
    
        # Evaluate the polynomial function at the x values in vectorDatosX
        y_predicted = [FuncionPolinomica(x, gradoMaxDelPolinomio, tipoDePolinomio, matrizCoeficientes.tolist()) for x in vectorDatosX]
    
        # Calculate the chi-squared value
        residuals = vectorDatosY - y_predicted
        chi_squared = np.sum(residuals**2 / vectorDatosDY)
    
        # Print the result
        return chi_squared



    chiExperimental = test_chi_squared()

    gradosDeLibertad = matrizDiseño.shape[0] - matrizDiseño.shape[1]

    chiTeorico = stats.chi2.ppf(0.05, gradosDeLibertad)

    chiExperimentalRed = chiExperimental / gradosDeLibertad



    # Evaluate the polynomial function at the x values in vectorDatosX
    y_predicted = [FuncionPolinomica(x, gradoMaxDelPolinomio, tipoDePolinomio, matrizCoeficientes.tolist()) for x in vectorDatosX]

    # Calculate the Pearson correlation coefficient
    corr, _ = stats.pearsonr(vectorDatosY, y_predicted)

    #IMPRIMIMOS TODOS LOS RESULTADOS
    print('\n \n')

    print(datos)
    print('\n')
    print('En regresion lineal es a + bx')
    listaDeCoeficientes = matrizCoeficientes.tolist()
    print('Los coeficientes son: ')
    for i in range(len(listaDeCoeficientes)):
        print(f"{abecedario[i]}:\t{listaDeCoeficientes[i][0]} ± {listaErrores[i]}")
    print()

    # Print chi-squared values
    print(f"ChiTeorico: {chiTeorico}; \t ChiExperimental: {chiExperimental}")

    # Compare chi-squared values
    if chiExperimental > chiTeorico:
        print("The fit is not good enough (chiExperimental > chiTeorico)")
    else:
        print("The fit is good enough (chiExperimental <= chiTeorico)")

    # Print chiTeoricoRed
    print(f"ChiTeoricoRed:\t{chiExperimentalRed}")
    print("")
    print(f"Pearson:\t{corr}")
    print("")

    return [listaDeCoeficientes, listaErrores, chiTeorico, chiExperimental, chiExperimentalRed, corr]


def guardameElCSV(resultados, nombreArchivo):

    resultado = pd.DataFrame({
        'coeficientes': resultados[0],
        'erroresCof': resultados[1],
        'chiTeor': resultados[2],
        'chiExp': resultados[3],
        'chiRed': resultados[4],
        'pearson': resultados[5]
    })

    resultado.to_csv(f'{nombreArchivo}.csv', index=False)

#A PARTIR DE AQUI NOS DEDICAREMOS A DIBUJAR LA GRAFICA CON SEABORN

def representacionPolinomica(x, resultados):
    resultado = 0
    grado = 0
    for i in resultados[0]:
        resultado += i[0] *(x**grado)
        grado += 1
    
    return resultado


#Creamos los datos para los puntos de la grafica

def ponmeLaGrafica(nombreArchivo, resultados, nombreEjeX, nombreEjeY, colorData='b', colorFit='r'):
    datos = pd.read_csv(nombreArchivo, delimiter=',', header=0, names=['x', 'dx', 'y', 'dy'])

    vectorDatosX = datos['x']
    vectorDatosDX = datos['dx']
    vectorDatosY = datos['y']   
    vectorDatosDY = datos['dy']


    Min = min(vectorDatosX)
    Max = max(vectorDatosX)
    n = 100
    p = abs(Max - Min) / n
    puntosDeLineaX = []
    puntosLineaY = []

    for i in range(n):
        x = Min + i * p  # Calcular el valor de x en cada iteración
        puntosDeLineaX.append(x)
        puntosLineaY.append(representacionPolinomica(x, resultados))

    fig=plt.figure(figsize=[18,12]) 
    ax=fig.gca() 
    plt.errorbar(datos.x, datos.y, xerr=datos.dx, yerr=datos.dy, fmt=f'{colorData}.', label='data', linewidth=3) 
    plt.plot(puntosDeLineaX, puntosLineaY, f'{colorFit}-', label='fit',linewidth=4.0) 
    #plt.xlim([0.2, 1.55]) 
    #plt.ylim([0.9, 6.5]) 
    plt.xlabel(r'${}$'.format(nombreEjeX),fontsize=25) 
    plt.ylabel(r'${}$'.format(nombreEjeY),fontsize=25) 
    plt.legend(loc='best',fontsize=25) 
 
    # Este comando permite modificar el grosor de los ejes: 
    for axis in ['top','bottom','left','right']: 
        ax.spines[axis].set_linewidth(4) 
 
    # Con estas líneas podemos dar formato a los "ticks" de los ejes: 
    plt.tick_params(axis="x", labelsize=25, labelrotation=0, labelcolor="black") 
    plt.tick_params(axis="y", labelsize=25, labelrotation=0, labelcolor="black") 
 
    plt.show()

def guardameLaGrafica(nombreArchivo, resultados, nombreEjeX, nombreEjeY, nombreImagen,colorData='b', colorFit='r'):
    datos = pd.read_csv(nombreArchivo, delimiter=',', header=0, names=['x', 'dx', 'y', 'dy'])

    vectorDatosX = datos['x']
    vectorDatosDX = datos['dx']
    vectorDatosY = datos['y']   
    vectorDatosDY = datos['dy']


    Min = min(vectorDatosX)
    Max = max(vectorDatosX)
    n = 100
    p = abs(Max - Min) / n
    puntosDeLineaX = []
    puntosLineaY = []

    for i in range(n):
        x = Min + i * p  # Calcular el valor de x en cada iteración
        puntosDeLineaX.append(x)
        puntosLineaY.append(representacionPolinomica(x, resultados))

    fig=plt.figure(figsize=[18,12]) 
    ax=fig.gca() 
    plt.errorbar(datos.x, datos.y, xerr=datos.dx, yerr=datos.dy, fmt=f'{colorData}.', label='data', linewidth=3) 
    plt.plot(puntosDeLineaX, puntosLineaY, f'{colorFit}-', label='fit',linewidth=4.0) 
    #plt.xlim([0.2, 1.55]) 
    #plt.ylim([0.9, 6.5]) 
    plt.xlabel(r'${}$'.format(nombreEjeX),fontsize=25) 
    plt.ylabel(r'${}$'.format(nombreEjeY),fontsize=25) 
    plt.legend(loc='best',fontsize=25) 
 
    # Este comando permite modificar el grosor de los ejes: 
    for axis in ['top','bottom','left','right']: 
        ax.spines[axis].set_linewidth(4) 
 
    # Con estas líneas podemos dar formato a los "ticks" de los ejes: 
    plt.tick_params(axis="x", labelsize=25, labelrotation=0, labelcolor="black") 
    plt.tick_params(axis="y", labelsize=25, labelrotation=0, labelcolor="black") 
 
    plt.savefig(f'{nombreImagen}.png')


def hazmeGraficas(ejeX, ejeY,*args):
    fig=plt.figure(figsize=[18,12]) 
    ax=fig.gca()
    for i in args:
        listaX = i[0]
        listaY = i[1]
        if i [3] != None:
            colorData = i[3]
        else:
            colorData = 'b'
        
        plt.plot(listaX, listaY, f'{colorData}-', label='fit',linewidth=4.0) 
        #plt.xlim([0.2, 1.55]) 
        #plt.ylim([0.9, 6.5]) 
    plt.xlabel(r'${}$'.format(ejeX),fontsize=25) 
    plt.ylabel(r'${}$.format(ejeY)',fontsize=25) 
    plt.legend(loc='best',fontsize=25) 

    # Este comando permite modificar el grosor de los ejes: 
    for axis in ['top','bottom','left','right']: 
        ax.spines[axis].set_linewidth(4) 

    # Con estas líneas podemos dar formato a los "ticks" de los ejes: 
    plt.tick_params(axis="x", labelsize=25, labelrotation=0, labelcolor="black") 
    plt.tick_params(axis="y", labelsize=25, labelrotation=0, labelcolor="black") 

    plt.show()

def dameDatosCSV(nombreArchivo):
    datos = pd.read_csv(nombreArchivo, delimiter=',', header=0, names=['x', 'dx', 'y', 'dy'])
    vectorDatosX = datos['x']
    vectorDatosDX = datos['dx']
    vectorDatosY = datos['y']
    vectorDatosDY = datos['dy']
    
    return [vectorDatosX, vectorDatosY, vectorDatosDX, vectorDatosDY]



def hola():
    print('hola')

