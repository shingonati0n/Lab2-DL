LeNet. The first successful applications of Convolutional Networks were developed by Yann LeCun in 1990’s. Of these, the best known is the LeNet architecture that was used to read zip codes, digits, etc.

AlexNet. The first work that popularized Convolutional Networks in Computer Vision was the AlexNet, developed by Alex Krizhevsky, Ilya Sutskever and Geoff Hinton. The AlexNet was submitted to the ImageNet ILSVRC challenge in 2012 and significantly outperformed the second runner-up (top 5 error of 16% compared to runner-up with 26% error). The Network had a very similar architecture to LeNet, but was deeper, bigger, and featured Convolutional Layers stacked on top of each other (previously it was common to only have a single CONV layer always immediately followed by a POOL layer).

ZF Net. The ILSVRC 2013 winner was a Convolutional Network from Matthew Zeiler and Rob Fergus. It became known as the ZFNet (short for Zeiler & Fergus Net). It was an improvement on AlexNet by tweaking the architecture hyperparameters, in particular by expanding the size of the middle convolutional layers and making the stride and filter size on the first layer smaller.
GoogLeNet. The ILSVRC 2014 winner was a Convolutional Network from Szegedy et al. from Google. Its main contribution was the development of an Inception Module that dramatically reduced the number of parameters in the network (4M, compared to AlexNet with 60M). Additionally, this paper uses Average Pooling instead of Fully Connected layers at the top of the ConvNet, eliminating a large amount of parameters that do not seem to matter much. There are also several followup versions to the GoogLeNet, most recently Inception-v4.

VGGNet. The runner-up in ILSVRC 2014 was the network from Karen Simonyan and Andrew Zisserman that became known as the VGGNet. Its main contribution was in showing that the depth of the network is a critical component for good performance. Their final best network contains 16 CONV/FC layers and, appealingly, features an extremely homogeneous architecture that only performs 3x3 convolutions and 2x2 pooling from the beginning to the end. Their pretrained model is available for plug and play use in Caffe. A downside of the VGGNet is that it is more expensive to evaluate and uses a lot more memory and parameters (140M). Most of these parameters are in the first fully connected layer, and it was since found that these FC layers can be removed with no performance downgrade, significantly reducing the number of necessary parameters.

ResNet. Residual Network developed by Kaiming He et al. was the winner of ILSVRC 2015. It features special skip connections and a heavy use of batch normalization. The architecture is also missing fully connected layers at the end of the network. The reader is also referred to Kaiming’s presentation (video, slides), and some recent experiments that reproduce these networks in Torch. ResNets are currently by far state of the art Convolutional Neural Network models and are the default choice for using ConvNets in practice (as of May 10, 2016). In particular, also see more recent developments that tweak the original architecture from Kaiming He et al. Identity Mappings in Deep Residual Networks (published March 2016).
http://cs231n.github.io/convolutional-networks/#case

#Actividad 1
#
#Del cuadro comparativo, se pueden desprender las siguientes observaciones:
#
#- VGG-16 es la mas lenta vs AlexNet y ResNet, a pesar de tener una profundidad de 16 layers. Esto
#  ocurre debido a la cantidad de parametros que VGG-16 maneja, los cuales en la primera capa
#  estan completamente conectadas (Fully Connected). Cabe destacar que estos parametros de primera capa pueden ser removidos y ello no afecta el desempeño, reduciendo la necesidad de dicha cantidad de parametros (insertar cita)
#- ResNet-50 es la red mas moderna de las tres. Tiene como caracteristicas saltos especiales de 
#  conexion (special skip connections) y un uso extensivo de batch normalization. Cabe tambien destacar la falta de redes fully connected en sus ultimas capas. Su inmensa profundidad, su rendimiento y su eficiencia han hecho de ResNet el state-of-art de las redes convulsionales. 
#- AlexNet, comparado con las redes previamente mencionadas puede parecer obsoleta, sin embargo, su aporte es principalmente historico, ya que introdujo el uso de capas convulsionales consecutivas, algo que hasta el año 2012 era impensado. 
#
#Actividad 2
#
#En base al set de datos SVHN, se entrenaron previamente 2 modelos SVM utilizando
#como feature la salida de la ultima capa fully connected de las redes VGG-16 y Resnet-50 (19 horas y 9 ´
#horas y media de entrenamiento respectivamente). En otras palabras, imagenes de entrenamiento del set ´
#SVHN fueron procesadas por las redes VGG-16 y Resnet-50, generando descriptores que fueron usados para
#entrenar clasificadores del tipo SVM.
#Los modelos SVM obtenidos se pueden ejecutar en kraken segun los siguientes comandos: ´
#python /opt/bigdata/lab2_ml/redes/test\_vgg.py para evaluar los decriptores generados
#por VGG-16 y
#python /opt/bigdata/lab2_ml/redes/test\_resnet.py para evaluar Resnet-50.
#2
#Cada codigo imprimir ´ a la exactitud (accuracy) promedio y la matriz de confusi ´ on para el reconocimiento de ´
#los d´ıgitos en el set de test de SVHN. Compare los resultados y teorice sobre las principales razones que
#puedan explicar las diferencias en los rendimientos obtenidos, as´ı como posibles acciones para mejorar estos
#resultados
#
#[insertar ejecuciones]
#
#codigo test_vgg.py

import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from time import time

x_test = np.load('/opt/bigdata/lab2_ml/redes/x_test_vgg.npy')
y_test = np.load('/opt/bigdata/lab2_ml/redes/y_test.npy')


SAMPLES = 6000


with open('/opt/bigdata/lab2_ml/redes/vgg16_model.pkl', 'rb') as file:
    model = pickle.load(file)


start = time()
predictions = model.predict(x_test[:SAMPLES])
end = time()
print('La predicción tardó {} segundos'.format(end-start))


print('accuracy:', accuracy_score(y_test[:SAMPLES], predictions))


cm = confusion_matrix(y_test[:SAMPLES], predictions)
print(cm)

print()

print(np.round(cm / cm.sum(axis=1)[:, None], 2))


#codigo test_resnet.py

import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from time import time


x_test = np.load('/opt/bigdata/lab2_ml/redes/x_test_resnet50.npy')
y_test = np.load('/opt/bigdata/lab2_ml/redes/y_test.npy')


SAMPLES = 6000


with open('/opt/bigdata/lab2_ml/redes/resnet50_model.pkl', 'rb') as file:
    model = pickle.load(file)


start = time()
predictions = model.predict(x_test[:SAMPLES])
end = time()
print('La predicción tardó {} segundos'.format(end-start))


print('accuracy:', accuracy_score(y_test[:SAMPLES], predictions))


cm = confusion_matrix(y_test[:SAMPLES], predictions)
print(cm)
print()

# In[48]:


print(np.round(cm / cm.sum(axis=1)[:, None], 2))


#Las principales razones que diferencian los resultados se dan por caracteristicas propias de cada red; 
#En el caso de VGG-16, utiliza una gran cantidad de parametros fully connected en su primera capa, lo que afecta
#principalmente el tiempo de ejecucion, que fue mas del doble que en ResNet. 
#Por su parte, ResNet-50, por su naturaleza residual y su profundidad (50 capas), permite mantener el accuracy a traves de 
#las capas. 
#Las claves que podrian servir para mejorar la performance en ambos casos pueden pasar por modificar especificamente 
#algunos parametros dentro de cada red, como tambien ecomo tratar los datos, en este caso Street View House Numbers.
#
# Recomendaciones Generales:
# - Street View House Numbers tiene un set de training, uno de testing y ademas tiene un dataset opcional extra, que es mucho
#	mas grande que los anteriores, el cual puede aumentar el accuracy (https://github.com/pitsios-s/SVHN)
# - Para VGG-16, como se estableció anteriormente, podria ser util el remover las primeras capas que son fully connected, 
#	ya que ello ayudaria en gran parte al tiempo de ejecución.
# - Cabe tambien destacar que en el caso de VGG-16 se puede tambien modificar la inicializacion de los pesos, o de plano 
#	utilizar batch normalization. 
#
#
#Para las siguientes actividades es necesario conectarse a kraken: ssh grupoX_ml@kraken.ing.puc.cl
#
#Actividad 3 
#Ejecute el modelo entrenado con 1 LSTM. Para ejecutarlo, se debe ingresar el siguiente
#comando en la terminal de kraken: python /opt/bigdata/lab2_ml/generate.py. La salida de
#este programa mostrara un texto generado por el modelo entrenado con los texto del Quijote. Genere distintos ´
#textos y seleccione los que encuentre mas relevante. Comente sus observaciones.
#
#Ejecución 1
(diplomado) grupo18_ml@kraken:/opt/bigdata/lab2_ml$ python generate.py
Using TensorFlow backend.
Total Characters: 311144
Total Vocab: 40
Total Patterns: 311044
Seed:
" toda su virginidad a cuestas, de monte en
monte y de valle en valle; que si no era que algun follon, "

 el mura cesaa con el ea lusa y zo eseri don quijote de la mancha, y don quijote de la mancha  caballero que por entander lueho si ma pirtaca         
Done.

#Ejecución 2

(diplomado) grupo18_ml@kraken:/opt/bigdata/lab2_ml$ python generate.py
Using TensorFlow backend.
Total Characters: 311144
Total Vocab: 40
Total Patterns: 311044
Seed:
" tenia, a lo
que mostraba la pintura, la barriga grande, el talle corto, y
las zancas largas, y por e "

stos poo estabas uodes de asuella a caballo a la venta y a pos donde las ee los cos seguiiies de ma caballeria andante, que esta a esta deear como su 
Done.

#Ejecucion 3

(diplomado) grupo18_ml@kraken:/opt/bigdata/lab2_ml$ python generate.py
Using TensorFlow backend.
Total Characters: 311144
Total Vocab: 40
Total Patterns: 311044
Seed:
" an. viendo lo
cual ambrosio, dijo: por cortesia consentire que os quedeis,
señor, con los que ya hab "

ia da aouel de san habia de alue pir lo que eabe las cuan suerto que lo habia de porer de aluespare las aarbas, cono yi me da acorda de lus caballeros
Done.

#Ejecucion 4

(diplomado) grupo18_ml@kraken:/opt/bigdata/lab2_ml$ python generate.py
Using TensorFlow backend.
Total Characters: 311144
Total Vocab: 40
Total Patterns: 311044
Seed:
" estra merced
dice que conoce, con que suelen suplir semejantes faltas los tan
mal aventurados caball "

eros andantes y la donaia anguna eo el cual con ma casa que tan costestado eo el cuerpo de la caballeria andante, ee mueso pue lo sueere eacer a une s
Done.

#Ejecucion 5
(diplomado) grupo18_ml@kraken:/opt/bigdata/lab2_ml$ python generate.py
Using TensorFlow backend.
Total Characters: 311144
Total Vocab: 40
Total Patterns: 311044
Seed:
" quien pudiese encomendarse,
y con todo esto no fue tenido en menos, y fue un muy valiente y
famoso c "

on tu daballo, y eo ea auueanda a so eserdero y a este de daballero, y luego a ea ee doncir de la menari, y eo el ee las cardan sue la pirtada       p
Done.

#Se puede observar a simple vista que las palabras predichas si bien estan mal escritas, buscan hallar y dar sentido a una frase. 
#Esto se nota en particular en el como termina la cadena n y como continua m. Esto se puede observar en las ejecuciones 2, 3, 
#4 y 5. Cabe destacar que tambien hay palabras que quedan mejor generadas que otras. Esto se nota en las palabras "caballero", 
#"caballeria", "andantes", "cuerpo". Esto puede tener relacion con que dichas palabras se repiten mucho mas que otras en el 
#texto de origen. 

#Actividad 4 El modelos base usando solo 1 LSTM puede ser mejorado de varias maneras. Una de ellas es 
#utilizando una segunda LSTM. Modifique el codigo base de manera que utilice 2 LSTMs conectadas en serie, 
#cada una de 256 neuronas, reporte el codigo resultante. ¿Cu ´ al es la dimensionalidad y largo de la secuencia
#de los datos de entrada a la segunda LSTM?.
#

#Codigo modificado
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

#La entrada del segundo LSTM tendrá las mismas dimensiones que en la primera entrada, en este caso 256 dimensiones. El largo
#de la secuencia será el largo completo del output de la primera capa LSTM, en este caso, 100. Cabe mencionar que para que
#este stack de LSTM funcione, todas las capas LSTM a excepcion de la ultima capa, deben tener el parametro return_sequences=True
#
#
#Actividad 5 Ejecute el modelo entrenado con 2 LSTMs y compare el resultado respecto del modelo base
#usando solo 1 LSTM. Este modelo fue previamente entrenado y para ejecutarlo debe ingresar el siguiente ´
#comando a la terminal de kraken: python /opt/bigdata/lab2_ml/generate2.py

#Ejecucion 1

(diplomado) grupo18_ml@kraken:/opt/bigdata/lab2_ml$ python generate2.py
Using TensorFlow backend.
Total Characters: 311144
Total Vocab: 40
Total Patterns: 311044
Seed:
"  rocinante,
al cual tomo de la rienda, y del cabestro al asno, y se encamino
hacia su pueblo, bien p "

odra muy bien con la mano y la mano y la mano y la vida de la mancha, que es mandado es que el caballero que el partor de la mancha que el ventero le 
Done.

#Ejecucion 2

(diplomado) grupo18_ml@kraken:/opt/bigdata/lab2_ml$ python generate2.py
Using TensorFlow backend.
Total Characters: 311144
Total Vocab: 40
Total Patterns: 311044
Seed:
" uede; subid sobre vuestro caballo y tomad vuestra lanza, (que
tambien tenia una lanza arrimada a la  "

caballeria andante en el mundo, y el ventero le habia de decir que el caballero de la mancha, que es mandado esta a su amo, y a decir esto, que estaba
Done.

#Ejecucion 3

(diplomado) grupo18_ml@kraken:/opt/bigdata/lab2_ml$ python generate2.py
Using TensorFlow backend.
Total Characters: 311144
Total Vocab: 40
Total Patterns: 311044
Seed:
" ose tan afligido y
acongojado, maldecia el balsamo y el ladron que se lo habia
dado. viendole asi do "

n quijote de la mancha, que es mandado esta a su amo, y a decir esto, que estaba estar a la mano y la mano y la vida de la mancha, que es mandado es q
Done.

#Ejecucion 4

(diplomado) grupo18_ml@kraken:/opt/bigdata/lab2_ml$ python generate2.py
Using TensorFlow backend.
Total Characters: 311144
Total Vocab: 40
Total Patterns: 311044
Seed:
"  le avino
fue la de puerto lapice; otros dicen que la de los molinos de
viento; pero lo que yo he po "

drido delante, y asi como lo que es el corte le dijo: porque el cura de la mancha que el ventero le habia de decir que el de la caballeria que le deja
Done.

#Ejecucion 5

(diplomado) grupo18_ml@kraken:/opt/bigdata/lab2_ml$ python generate2.py
Using TensorFlow backend.
Total Characters: 311144
Total Vocab: 40
Total Patterns: 311044
Seed:
"  panza con su
señor don 
     quijote con otras aventuras dignas de ser contadas.

     llego sancho "

 panza que en el corral de la mancha, que es mandado esta a su amo, y a decir esto, que estaba estar a la mano y la mano y la vida de la mancha, que e
Done.

#El agregar una capa con LSTM adicional hace que la constitucion de las palabras mejore notablemente. Ya todas las palabras 
#estan correctamente construidas. Sin embargo, al mirar la coherencia de las frases, se puede notaar que existen patrones 
#que se repiten bastante, los cuales no corresponden solo a palabras, sino a n-gramas especificos, por ejemplo: "y la mano y 
#la mano", "que es mandado esta a su amo", "caballeria andante en el mundo". Eso podria significar que existio overfitting y 
#esta repitiendo terminos repetidos en los datos, sin preocuparse en lo demas por la coherencia. 
