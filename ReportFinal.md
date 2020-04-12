# 1. Tamanho dos Dados
Uma adversidade bastante notória no dataset apresentado, é o facto deste ser relativamente pequeno, em comparação com por exemplo, os de outros temas.
Ora, como veremos mais adiante, isto teve um impacto significativo nos resultados obtidos, uma vez que cria uma certa tendencia para overfitting. Para dar uma ideia, o dataset é constituido por cerca de
2062 imagens, distruibuidas por 10 classes. Estando equilibrado, temos cerca de 200 imagens por classe. Com um numero tao reduzido de dados, aquando do testing size, optou-se por selecionar 20%, o que se traduz em aproximadamente 160 imagens para treino e 40 para testing - note-se que se garantiu uma divisão equilibrada por classe. Com um número (consideravelmente) maior de dados, eventualmente escolher-se-ia uma percentagem menor, aumentando o volume de treino, o que neste caso não é possível.
[grafico distribuiçao de dados]
(nao sei se falo de CV ser importante pk small ds)

# 2. Metodo Aplicado
Numa primeira fase, e tendo em conta a inexperiencia neste dominio, pensou-se em aplicar todos os metodos e hyperparameters (possíveis) de uma vez, de forma a conseguir filtrar quais os que apresentavam melhores resultados. Contudo não se veio a tornar uma ideia muito brilhante, uma vez que não só era computacionalmente dispendioso, como havia resultados simplesmente inuteis, com accuracies a rondar 0.1. Esta tentativa falhada, não se pode considerar inesperada, contudo foi um primeiro passo interessante, que permitisse uma melhoria do método a aplicar.
Numa visão mais empirica, e tendo em conta que datasets mais pequenos encorrem num maior risco de overfitting, quando expostos a modelos mais complexos, decidiu-se portanto priorizar modelos mais simples.

# 3. Considerações 
## 3.1 Cost Functions
Para os métodos em que se usou a libraria sklearn, não existe um método que retorne o custo por iteração. Embora existam alguns "workarounds" nao internet, estes nao foram considerados práticos nem pythonics, e por isso descartados. Por esse motivo, e tendo em conta que o dataset esta bastante bem equilibrado, decidiu-se usar a accuracy como métrica principal nos resultados obtidos. Note-se que esta medida por ser falaciosa, principalmente em datasets desilibrados, mas como referido, tal não se aplica a este caso.
## 3.2 Gráficos Apresentados
Nos gráficos apresentados, em alguns casos, não se encontram presentes todos os dados estudados durante este projeto. Apenas se incluem os considerados mais relevantes - casos extremos e melhores casos, de modo a minimizar a poluição dos gráficos. Considera-se contudo que esta permissa não afeta negativamente o estudo a ser feito.

# 4. Métodos
## 4.1 Logistic Regression
Com o considerado anteriormente, optou-se então por começar com um dos métodos mais simples de machine learning. Graças ao que foi dito no ponto 2., nesta etapa foi aplicada uma escolha inicial de hyperparameters mais seletiva e promissora.

### 4.1.1 Parametros
A função da biblioteca sklearn utilizada, LogisticRegressionCV, é como o nome indica uma função encarregaa de implementar regressão logistica com validação cruzada.
Tal como outras funções usadas mais a frente, apresenta parametros que desconheciamos à posteori, pelo que passaremos agora a uma justificação dos mesmos:
- na generalidade dos casos, pelo desconhecimento em muitos dos parametros, os não mandatórios foram deixados como default;
- como *solver*, escolheu-se newton-cg e lbfgs - pois eram os metodos mais comuns e são os unicos que suportam "multinomial loss";
- penalty: como consequencia da escolha anterior, apenas podiamos escolher penalti "l2" - uma forma de "ridge regression" - o que vai ao encontro do referido anteiormente sobre o problema de overfit, na medida em que esta tecnica é bastante boa a evita-lo;
- cv, ou seja o número de k-fold, por esta variavel apresentar um elevado custo computacional - no sentido em que variaçoes deste parametro são dispendiosas de trabalhar com - manteve-se o default de 5. Aplicou-se o método Stratified K-fold, que se destaca por tentar equilibrar o numero de exemplos de cada class em cada fold - Stratification is the process of rearranging the data as to ensure each fold is a good representative of the whole;

### 4.1.2 Resultados
(2 graficos here)
Valores de C (pertence) [0.01, 0.03, 0.1, 0.3, 1, 1.3, 3, 6, 10]
Como se pode ver em ambos os graficos, os dois métodos convergiram para o mesmo valor. Isto deve-se ao facto de que LBFGS é "in a nutshell", analoga ao metodo de Newton, daí ser natural esta sobreposição ao longo do gráfico. A diferença está principalmente no facto de que LBFGS usa aproximações e tecnnicas para poupar memória. O que lhe confere potencial em problemas com pequenos datasets.

Quanto ao gráfico 1, observa-se rapidamente que estamos perante um caso de overfitting, à medida que o C (lembrando que é o inverso de lambda) aumenta. Contudo pode-se apontar o facto de que a amplitude de valores de C é bastante notória, pelo que se decidiu fazer tuning aos valores mais promissores, neste caso à volta do valor C=0.1.
Isto leva-nos ao 2o gráfico, que demonstra que de facto era uma suposição correta. Note-se que o ponto C=0.12 apresenta um lijeiro aumento na accuracy, contudo esta é bastante negligenciável para estudo posterior.
Confirmando estas observações, na própria função do sklearn, é possível verificar qual o mehor C para cada classe:
[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
O mapeamento aqui é feito de forma a que cada index corresponda a uma class.

### 4.1.3 Conclusão
Sendo um modelo relativamente simples, e tendo em conta que estamos a trabalhar com imagens, ou seja classes com "elevado" numero de features, um modelo linear deste tipo não apresenta uma performance muito elevada nesta situação

## 4.2 SVM
Avançando para um algoritmo mais complexo, deparamo-nos com SVMs. Embora um bom Kernel para começar seja o RBF, devido a sua qualidade "general purpose", decidiu-se começar pelo kernel linear.
Recorrendo às ferramentas disponibilizadas pela biblioteca em uso, fez-se uso da função GridSearchCV, que permite uma pesquisa exaustiva por parametros passados, escolhendo por fim os mais apropriados. 

### 4.2.1 Linear Kernel
Também conhecido por representar uma SVM sem kernel, parece uma boa alternativa para continuar o processo anterior, tendo em conta que continuamos a trabalhar com modelos lineares.

#### 4.2.1.1 Parametros
Uma vez que não possuí Kernel, não existem muitos Hyperparameters com os quais se possa trabalhar, exceto o valor C, que também aqui representa o inverso da força de regularização (for 1/gamma).
- como cv, usado na GridSearch, mais uma vez optou-se pelo default,  StratifiedKFold 5-fold;

#### 4.2.1.2 Resultados

#### 4.2.1.? Conclusão
Embora inicialmente se pensa-se que este modelo e o anterior fossem apresentar resultados muito identicos, devido à natureza dos mesmos, tal não aconteceu. Neste caso, a base accuracy subiu consideravelmente. Numa tentativa de possível justificação para tal, relembra-se o facto de que o dataset é bastante pequeno. Ora, nesta situação, **Outliers** têm maior influencia nos dados, o que pode afetar os resultados. Estes Outliers terão mais influencia no modelo anterior, uma vez que o mesmo tem com consideração todas as entradas para o calculo da função de regressão logistica. Por outro lado, no modelo atual, há um enfoque maior na "decision boundary", tal que o objetivo passa por colocar esta boundary linear de maneira intelegente. Desta forma, apenas pontos próximos da decision boundary fazem realmente diferença, diminuindo assim a unfluencia destes outliers, e possivelmente aumento a accuracy do modelo.

#### 4.2.2 RBF Kernel
Sigla para (Gaussian) Radial Basis Function, é a escolha mais comum, sendo inclusive a escolha por defeito. Um outro motivo a favor deste e do anterior Kernel encontra-se também no _slide 27 da lecture 5_ (por imagem) da cadeira de TAA: elaborando, considerou-se que o dataset em estudo se podia encontrar de alguma maneira entre os dois primeiros casos apresentados.

#### 4.2.2.1 HiperParametros
Neste contexto, a nivel de parametros quer da função de SVM, quer do "wrapper" GridSearch, não foram alterados, mantendo-se os valores default.
Quandos aos hiperparametros, este kernel apresenta mais um, gamma (por icon), devido à sua natureza Gaussiana.
Numa primeira fase foram escolhidos alguns valores arbitrários, mas pensados, para um primeiro desenho dos resultados, ao qual depois foi feito algum tuning nos parametros de forma a maximizar a accuracy.

#### 4.2.2.2 Resultados
(por 2 graficos)
https://datascience.stackexchange.com/a/42600 -> cuidado ao dizer k LOSS ~= ACC
MELHOR VALOR: lambda:0.01 - C=3
# 4.3 Neural Network
