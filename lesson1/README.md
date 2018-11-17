# Language

[English](#english)

[Русский](#russian)

<a name="english"/>

# Lesson instructions

## Problem statement

In this lesson we are going to make a network that does regression. We will
predict the value of some basic function such as sine or cosine in a given
point <img src="/lesson1/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/>.

Speaking more generally, we will make a network that produces some approximation
of an unknown function <img src="/lesson1/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/>.

Note that this is possible to do with any
precision using Neural Network with sigmoid activation functions when <img src="/lesson1/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/> is bounded and has
finite amount of the discontinuities.

## Instructions
### Create a Dataset

In this lesson we will generate the dataset by ourselves. To do this we will
take some function <img src="/lesson1/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> (bounded and that has less than limited amount of
discontinuities). As a dataset we will use a set of pairs
<img src="/lesson1/tex/20886c95cad07a2e07cc69a3cae30aee.svg?invert_in_darkmode&sanitize=true" align=middle width=91.05878099999998pt height=24.65753399999998pt/>, where <img src="/lesson1/tex/1cd32b0756da515bc59142b9318ff797.svg?invert_in_darkmode&sanitize=true" align=middle width=11.323291649999991pt height=14.15524440000002pt/> is a random noise that we will
generate ourselves. The reason why do we need to add some noise is because
we make measurements with some precision. In our case we will consider the noise that is independent in each point <img src="/lesson1/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/> and equally distributed.

In real life, we will have a list of pairs <img src="/lesson1/tex/9cb3b82be5418fbb7dad8a5c4ae38d9b.svg?invert_in_darkmode&sanitize=true" align=middle width=34.88399804999999pt height=14.15524440000002pt/> as a dataset, using which
you should restore the hidden dependency <img src="/lesson1/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/>:
<p align="center"><img src="/lesson1/tex/2398b6b54c85aadd6104d17fb77bebbf.svg?invert_in_darkmode&sanitize=true" align=middle width=104.3349714pt height=16.438356pt/></p>.

To accomplish this, create the file ```sine_dataset.py``` and:
1) make two classes: ```DataSetIndex``` and ```DataSet```.

2) for class ```DataSetIndex``` make
a method
```__init__(self, seed=0, n_items=1000, low=-10.0, high=10.0, noise=0.1)```.

* Here initialize numpy's random number generator with ```seed``` variable

* create random vector of coordinates (uniformly distributed between ```high```
and ```low```),

* compute the values of function <img src="/lesson1/tex/fb9dd87ee5c46cc6f1bac5ad989387d4.svg?invert_in_darkmode&sanitize=true" align=middle width=45.416004149999985pt height=24.65753399999998pt/> in the generated coordinates

* generate noise of the corresponding size (normal distribution with 0 mean and standard deviation equal to the ```noise_level```)

* add the noise to the computed values

* split the coordinates and the values into three subsets: ```train```,
```valid``` and ```test``` (```valid``` should be 10% of the dataset, ```test```
should be 50% of the dataset)

* create the variable ```order``` for each subset. In this variable the order of the elements of the subset will be specified.
Make ```train``` order random (using ```numpy.random.permutation```), for the ```valid``` and ```test``` make sequential
order from 0 to N (using ```numpy.arange```).

3) for the ```DataSet``` class make a method ```__init__(self, ds_index, mode='train')```. In these method save ```ds_index``` and ```mode``` into the object ```self```.

4) for the ```DataSet``` class make a method ```__len__(self)```, which will return amount of the samples in the ```self``` object. Remember that the amount of samples is different for ```train```, ```valid``` and ```test``` splits.

5) For ```DataSet``` class make a method ```__getitem__(self, index)``` which will return two things: first, input for a network (tensor containing one value for our case) wrapped into a list and target value (again, tensor, containing one value) wrapped into a list.  

6) for the class ```DataSet``` make ```shuffle(self)``` method. This method will be called before each epoch (epoch is a period when Neural Network encounters once all of the training samples). It should shuffle the order of the training set for the epoch. It should leave valid and test subsets as they are.

7) test the dataset.

* Observe what the ```DataSet``` object returns when indexed (for exampled when the object with number 0 is taken).

* Check that ```DataSet``` works in
all the modes (```train```, ```valid```, ```test```) for all the indices. Observe your data: plot <img src="/lesson1/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> and noisy <img src="/lesson1/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/> using Jupyter Notebook.

### Create a Model

We have already completed the dataset script -- this is half of the task. The second half is to make a model, loss function, metrics and optimizer.

1) To start with this, we will make the model's architecture first. For this we will define a class ```Model``` in the file ```model.py```. This class should be inherited from ```torch.nn.Module```.

2) In this class make a function ```__init___(self)```, in which all the elements of the net will be defined. In our case Neural Network will consist out of two Linear Layers (Hidden and Ouput). Amount of the hidden neurons may be defined as a parameter of the network.

3) Now, we should define how the output will be computed. Create method ```forward``` in the ```Model``` class. This method gets list of inputs and returns list of results (containing one tensor in our case).
Carry out the computations in the following way: input -> hidden_layer -> sigmoid -> output_layer. Return the result.

Thus, we have completed our Neural Network.

### Make a Socket (model processor)

This class should contain all the information about how the Model class should be treated.

The following methods should be defined for this class: ```__init__``` -- constructor,
```criterion``` -- loss-function, ```metrics```, ```process_result```. Other methods are optional.

1. Make a method ```__init__(self, model)```:
save a model in a field ```self.model```,
define trainable modules (all of the network's layers), optimzer and parameter for the optimizer using scorch.OptimizerSwitch(trainable_modules, optimizer, optimizer_parameters). In this example use torch.optim.Adam optimizer with ```3.0e-2``` learning rate.

2. Make a method ```criterion(self, outupt, target)```. This method will recieve lists of Neural Network's outputs and lists of target values for those outputs. In our case those lists will contain one tensor each. In this task we will use Mean Squared Error loss function. Using operations from pytorch module, compute MSE between ```ouput[0]``` and ```target[0]``` and return its value.

3. Make a method ```metrics(self, output, target)```. It will receive exactly the same lists as the ```criterion``` method. In this task we will use the metrics "fraction of samples with error lower than threshold":

<p align="center"><img src="/lesson1/tex/864fb8323dbdcc47539e9f077d976e2c.svg?invert_in_darkmode&sanitize=true" align=middle width=448.80627825pt height=62.6919018pt/></p>

Here <img src="/lesson1/tex/8e6f8772884838ad7db8233311f53511.svg?invert_in_darkmode&sanitize=true" align=middle width=37.47063044999999pt height=31.50689519999998pt/> is an output of Neural Network in <img src="/lesson1/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/> point, <img src="/lesson1/tex/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode&sanitize=true" align=middle width=12.710331149999991pt height=14.15524440000002pt/> -- target value. Choose <img src="/lesson1/tex/ccc8c36bd75b0a3fd820174d730dea02.svg?invert_in_darkmode&sanitize=true" align=middle width=150.96466109999997pt height=21.18721440000001pt/>. These are our metrics. Return the dictionary containing metrics values.
Assign one of the metrics to the key ```"main"``` of the dictionary.

# Model training

Launch training process for 10 epochs.
```
scorch-train --model model.py --dataset dataset.py --epochs 10 -cp 10_epochs
```

Make yet another training process for 100 epochs.
```
scorch-train --model model.py --dataset dataset.py --epochs 100 -cp 100_epochs
```

Evaluate results of the training

1) make test pass with the model trained for 10 epochs
```
scorch-test --model model.py --dataset dataset.py -cp 10_epochs --prefix 10_epochs
```

make test pass with the model trained for 100 epochs
```
scorch-test --model model.py --dataset dataset.py -cp 100_epochs --prefix 100_epochs
```

2) read results of the test pass into the Jupyter Notebook and plot the results.

# Help

If you have problems during completing this task, you may find example solution [here](https://github.com/RomanovMikeV/deep-dive-in-deep-leaning/blob/master/lesson1/Solution.ipynb)


<a name="russian"/>

# Инструкции
* [Постановка задачи](#rus_statement)
* [Написание кода](#rus_solution)
    * [Датасет](#rus_dataset)
    * [Модель](#rus_model)
    * [Обработчик модели](#rus_socket)
* [Тренировка модели](#rus_training)
* [Помощь](#rus_help)

## Постановка задачи <a name="rus_statement"/>

В этом уроке мы решим задачу регрессии при помощи нейронной сети. Мы будем
предсказывать значение некоторой простой функции (например, синуса или косинуса)
в некоторой точке <img src="/lesson1/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/>.

Иными словами, мы сделаем сеть, которая делает аппроксимацию некоторой
неизвестной (скрытой) зависимости <img src="/lesson1/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/>.

Стоит отметить, что это можно сделать с любой точностью при помощи сигмоидной
нейронной сети (функции активации -- сигмоиды) для ограниченной функции <img src="/lesson1/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/> с конечным числом разрывов.


## Написание кода <a name="rus_solution"/>

### Create a Dataset / Сделайте датасет <a name="rus_dataset"/>

В этом уроке мы сделаем датасет сами. Для этого мы возьмем некоторую функцию <img src="/lesson1/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/>
(ограниченную, имеющую не больше, чем счетное количество разрывов), и сделаем
набор пар <img src="/lesson1/tex/20886c95cad07a2e07cc69a3cae30aee.svg?invert_in_darkmode&sanitize=true" align=middle width=91.05878099999998pt height=24.65753399999998pt/>, где <img src="/lesson1/tex/1cd32b0756da515bc59142b9318ff797.svg?invert_in_darkmode&sanitize=true" align=middle width=11.323291649999991pt height=14.15524440000002pt/> -- некоторый шум, который
мы будем генерировать сами. Причина, по которой мы добавляем шум в наши данные
заключается в том, что всегда, когда мы производим измерения, мы получаем
значение с некоторой точностью. Мы рассмотрим случай, когда в каждой точке <img src="/lesson1/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/> шум независим от значений в других точках и одинаково распределен.

В реальной жизни у нас будет просто набор пар <img src="/lesson1/tex/9cb3b82be5418fbb7dad8a5c4ae38d9b.svg?invert_in_darkmode&sanitize=true" align=middle width=34.88399804999999pt height=14.15524440000002pt/>, в качестве датасета,
по которым необходимо восстановить скрытую зависимость <img src="/lesson1/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> такую, что
<p align="center"><img src="/lesson1/tex/2398b6b54c85aadd6104d17fb77bebbf.svg?invert_in_darkmode&sanitize=true" align=middle width=104.3349714pt height=16.438356pt/></p>.

Для выполнения этого задания создайте файл dataset.py и
1. сделайте в нем два класса: ```DataSetIndex``` и ```DataSet```

2. для класса ```DataSetIndex``` сделайте
метод
```__init__(self, seed=0, n_items=1000, low=-10.0, high=10.0, noise=0.1)```.

    * Здесь инициализируйте генератор случайных чисел numpy значением ```seed```

    * создайте случайный набор координат, равномерно распределенных от ```low``` до
```high```

    * посчитайте в этих координатах значение функции $sin(x)$

    * сгенерируйте
вектор шума соответствующего размера (нормальный шум со матожиданием 0 и
стандартным отклонением ```noise_level```)

    * добавьте шум к значениям функции

    * разбейте координаты и значения на три куска: ```train```, ```valid``` и ```test``` (```valid``` и ```test```
должны составлять соответственно по 10% и 50% от всей выборки).

    * cделайте переменную ```order``` для каждого куска датасета. В ней будет храниться порядок следования элементов датасета. Для ```train``` сделайте
случайный порядок индексов (при помощи ```numpy.random.permutation```), для ```valid``` и ```test``` сделайте порядок от 0 до N (```numpy.arange```).

3. для класса ```DataSet``` сделайте метод
```__init__(self, ds_index, mode='train')```. В нем сохраните ```ds_index``` и ```mode```
в обьект ```self``` класса ```DataSet```.

4. для класса ```DataSet``` сделайте метод ```__len__(self)```, который будет
возвращать количество примеров в ```self```. Помните, что количество примеров
разное для ```train```, ```valid``` и ```test``` кусков датасета.

5. для класса ```DataSet``` сделайте метод ```__getitem__(self, index)```,
который вернет два списка: входное значение для нашей сети, обернутое в список, и целевое значение для выходов, оберутое в список.

6. для класса ```DataSet``` сделайте метод
```shuffle(self)```. Этот метод будет вызываться в начале каждой эпохи (эпоха -- период, за который нейросеть просматривает по разу все обучающие примеры) и
перемешивать тренировочную выборку для эпохи. Валидационную и тестовую выборку он перемешивать не должен.

7. протестируйте датасет. Посмотрите, что возвращает ```DataSet``` при индексации
(например, при вытаскивании элемента с номером 0). Проверьте, что ```DataSet```
работает во всех режимах (```train```, ```valid```, ```test```) для всех
индексов. Посмотрите, как выглядят наши данные: сделайте график при помощи
Jupyter Notebook значений <img src="/lesson1/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> и <img src="/lesson1/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/>.

### Сделайте модель <a name="rus_model"/>

Мы уже сделали скрипт датасета -- это половина работы. Вторая половина --
определиться с архитектурой модели, лосс-функцией, метриками, оптимизатором.

1. Для начала сделаем архитектуру модели. Для этого определим класс ```Model``` в
```model.py```. Он должен быть унаследован от класса ```torch.nn.Module```.

2. В этом классе сделайте функцию ```__init__(self)```, в которой будут определены все
элементы архитектуры. В нашем случае нейронная сеть будет состоять из двух
линейных слоев (скрытого и выходного). Нужно создать эти слои в __init__.
Количество скрытых нейронов можно сделать параметром сети.

3. Далее, нужно определить, как будет вычисляться результат работы сети.
Создайте метод ```forward``` в классе ```Model```. Этот метод принимает один аргумент
(список входов в сеть) и возвращает список результатов (список из одного тензора в нашем случае).
Производите вычисления следующим образом:
вход -> скрытый слой -> сигмоида -> выходной слой.

Таким образом, мы сделали нашу нейронную сеть.

### Сделайте обработчик модели (Socket) <a name="rus_socket"/>

В этом классе находится вся информация о том, как взаимодействовать с моделью.

В этом классе нужно определить следующие методы: ```__init__``` -- конструктор,
```criterion``` -- лосс-функция, ```metrics``` -- метрики. Остальное определять
в этом задании не обязательно.

1. Сделайте метод ```__init__(self, model)```: сохраните модель в поле self.model,
определите тренируемые слои (поле ```self.train_modules```, в этом задании это вся
модель, поэтому можно в качестве тренируемых слоев передать ```self.model```),
задайте оптимизатор self.optimizer (пусть в нашем случае это будет Adam со
скоростью обучения ```3.0e-4```).

2. Сделайте метод ```criterion(self, output, target)```. Сюда будут приходить
списки результатов работы модели и целевых значений. В нашем случае эти списки
будут содержать по одному тензору. В нашем случае функцией ошибки будет
среднеквадратичное отклонение (Mean Squared Error, MSE). Используя операции
пакета torch расчитайте значение MSE между output[0] и target[0] и верните
результат.

3. Сделайте метод ```metrics(self, output, target)```. Сюда будут приходить точно
такие же списки, как и в метод ```criterion```. В этом задании мы будем
использовать метрику "доля точек с ошибкой меньше заданной":

<p align="center"><img src="/lesson1/tex/864fb8323dbdcc47539e9f077d976e2c.svg?invert_in_darkmode&sanitize=true" align=middle width=448.80627825pt height=62.6919018pt/></p>

Здесь <img src="/lesson1/tex/8e6f8772884838ad7db8233311f53511.svg?invert_in_darkmode&sanitize=true" align=middle width=37.47063044999999pt height=31.50689519999998pt/> -- результат работы нейронной сети в точке <img src="/lesson1/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/>, <img src="/lesson1/tex/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode&sanitize=true" align=middle width=12.710331149999991pt height=14.15524440000002pt/> --
таргетное значение. Возьмите <img src="/lesson1/tex/ccc8c36bd75b0a3fd820174d730dea02.svg?invert_in_darkmode&sanitize=true" align=middle width=150.96466109999997pt height=21.18721440000001pt/>. Это и будут
наши метрики. Верните словарь из метрик.

## Тренировка модели <a name="rus_training"/>

1. Запустите процесс тренировки
```
scorch-train --model model.py --dataset dataset.py --epochs 10 -cp 10_epochs
```

2. Также сделайте обучение на протяжении 100 эпох
```
scorch-train --model model.py --dataset dataset.py --epochs 100 -cp 100_epochs
```

3. Оцените визульно качество обучения:

      * сделайте тестовый прогон модели с 10 эпохами
      ```
      scorch-test --model model.py --dataset dataset.py -cp 10_epochs --prefix 10_epochs
      ```

      * cделайте тестовый прогон модели со 100 эпохами
      ```
      scorch-test --model model.py --dataset dataset.py -cp 100_epochs --prefix 100_epochs
      ```

4. считайте результат из ноутбука и сделайте график получившейся функции

## Помощь <a name="rus_help"/>

Пример решения задачи можно найти в файле [ноутбуке](https://github.com/RomanovMikeV/deep-dive-in-deep-leaning/blob/master/lesson1/Solution.ipynb)


