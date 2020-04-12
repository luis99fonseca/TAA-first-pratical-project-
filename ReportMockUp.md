## 1. Inicio
(falar da data em si... eu devia era dividi-la!! :/) -> que pus 20% pa testing and shit
Also o uso de CV foi importante, pois sendo 1 set pequeno, é util utlizar todos os dados, per see

Tentamos tudo (ser mais explicito neste tudo [ml-one.py]), so para ter uma ideia de quais os melhores modelos e hyperparameters pelos quais deveriamos exprimentar. (Ver se faz sentido falar do método cientifico (ou empirico?) - aka tentar tudo pa ver the best). Contudo não se tornou uma ideia muito acolhedora, uma vez que era computacionalmente dispendioso (de longe), pelo que rapidamente tivemos de dar drop deste método. Acrescentando a isto, havia resultados, que eram simplesmente inuteis, com accuraries abaixo de 0.1, o que further evidenciava o desperdicio de tempo que e estava a cometer. Contudo serviu como um 1o passo interessante, tendo em conta a nossa inexperiencia, e a "vontade" de exprimentar tudo.

Acrescentando a isto de método empirico. vimos em [https://datascience.stackexchange.com/a/26146] e o porque [https://www.quora.com/What-is-an-intuitive-explanation-of-over-fitting-particularly-with-a-small-sample-set-What-are-you-essentially-doing-by-over-fitting-How-does-the-over-promise-of-a-high-R%C2%B2-low-standard-error-occur] que smalldata sets têm um maior risco de overfitting, como sentimos adiante. Além disso outliers têm mais impacto. (pensamos nisto para o testsize).
Para evitar overffing, é necessário evitar complexidade, ou seja priorizar modelos mais simples. (like we did)

Dizer que até o test size variamos [TAA-1st Projeto, LR- 2nd run-], e tendo em conta o size do dataset, acabamos por considerar 20% para testing, uma vez que foi o k consideramos melhor. Isto é bue arbitrario, há algumas thumb rules e tals. Por isso acabou por ficar assim. Se, por exemplo o dataset fosse maior (em alguns milhares), poder-se ia eventualmente reduzir a % de teste, pk em valor absoluto iria ja representar um bom set com shit != e o crl.

## Notas: 
### 1 (ver depois com o pedro)
Para as funçoes onde se usou a lib sklearn, não existe 1 metodo que retorne o custo / itereçao em cada tecnica usada. Embora existam workarounds quanto a isto na internet, estes nao foram considerados muito praticos, let alone pythonic, e por isso descartados. Por esse motivo, nos casos de uso é considerada a accuracy dos resultados obtidos. Note-se que esta medida pode ser falaciosa, especialmente em casos em que o dataset nao esta bem distribuido, mas isso nao é o caso neste dataset em especifico, que é equilibrado. (or is it?? fucking mete aquilo igual)
### 2 
Nos graficos apresentados, em alguns casos, nao se encontram todos os dados estudados aapresentados, apenas os mais relevantes - casos extremos e melhores casos, de modo a minimzar poluiçao dos graficos. Considera-se contudo que isto permite um estudo satizfatorio anyway. TODO: nos graficos em que usamos outros valores, justifica-lo!!!

## 2. LogReg
### NOTA: falar dos Ks de cada CV
(Com esta merda do metodo cientifico em mente e o que foi dito anteriormente), decidiu-se refazer isto de forma mais intelegente, começando então por um dos métodos mais simples (LogReg), mas desta vez escolhendo apenas hyperparameters (incluindo penalty and solvers) mais promissores (baseado tanto no resultado de 1. como em alguma pesquisa na net).

### Parametros
A funçao da biblio sklearn, LogisticRegressionCV, é como o nome indica uma funçao encarregada de implements logistic regression with cross validation.
Tal como outras funçoes que veremos mais a frente, apresenta parametros que desconheciamos à posteri, apresentando agora justificativas para a escolha dos mesmos:
- no geral, por como disse, desconhecer a lot of shit, deixamos a maior partes das coisas como default;
- como *solver*, escolhermos newton-cg e lbfgs - pois eram as mais comuns e sao das unicas que suportam "multinomial loss" (é basicamente a funçao de custo normal anyway, so k pa varias classes) ;
- penalty: como consequencia da escolha anterior, apenas podiamos escolher penalti "l2", o que também vai de acordo com o queriamos anywya: de acordo com [https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c] l2 ajuda muiiitto com overffiting problems. (é 1 caso d Ridge regression)
- cv, ou seja o numero de K-Fold, por esta variavel (https://datascience.stackexchange.com/a/28160) representar um alto custo computacional para valores mais elevados, manteve-se o default, 5, com o "metodo" [https://medium.com/@xzz201920/stratifiedkfold-v-s-kfold-v-s-stratifiedshufflesplit-ffcae5bfdf] Stratified K-fold, que tem o seu uso por tentar equilibrar o numero de exemplos de cada class em cada fold - Stratification is the process of rearranging the data as to ensure each fold is a good representative of the whole;

### Resultados
[por aqui os dois graficos]
(Dizer que os dois metodos convergiram po msm valor)
uma vez que lbfgs In a nutshell, it is analogue of the Newton’s Method, +é natural que estas se sobreponham ao longo do grafico. A diferença acontece no facto de lbfgs usar algumas tecnicas e aproximaçoes para ser menos computacionalmente custosa e poupar memoria. Considara até bastante boa em problemas com pequeno dataset devido a isso. [https://stackoverflow.com/a/52388406]

>> https://stackoverflow.com/questions/42424444/scipy-optimisation-newton-cg-vs-bfgs-vs-l-bfgs [o pk de newton e lfgs serem identicos]

à 1a vista observa-se rapidamente que estamos perante um caso de overfitting. à medidade que o C (que é o inverso de lambda, aumenta). Nao obstante, tbm é notório a amplitude de valores de C, pelo que se decidiu fazer tunning aos valores mais promissoes, neste caso, around 0.1;
Isto leva-nos ao 2o grafico, que demonstra que de facto 0.1 seria o melhor valor possível. Note-se que eventualmente outros valores poderiam potencialmente apresenar resultados lijeiramente melhores, mas isso implicaria fazer 1 pesquisa com casas decimais desnecess+ariamente pequenas, com o objetivo de subir negligenciavelmente isto.


### K-folding
COmo foi dito, ao ser 1 var associada a gds custos computacionais, decidiu-se testá-la posteriormente, após se encontrar os hyperparameters mais promissores e tals.
Nao obstante, talvez pelo facto de os valoes ja terem convergido, nao se encontrou alteraçoes com K entre 2-10; Gostaria de se testar Leave One Out, mas uma vez que os tempos estavam a aumentar quase exponencialmente, muito provavelmente ainda estariamos aqui hoje a trainar.
[https://stats.stackexchange.com/a/52277] -> nao sei se falo deste chapter pk n sei sepercebi o k faz/fiz 

## 2.2 Conclusao: 
Sendo um algoritmo relativamente simples, nao é surpreendente que os resultados nao sejam otimos. (DIZER MAIS, but what??)


## 3. SVM

Dizer que [https://medium.com/axum-labs/logistic-regression-vs-support-vector-machines-svm-c335610a3d16] [https://stats.stackexchange.com/questions/95340/comparing-svm-and-logistic-regression] -> tbm ta de acordo, e como é do professor do coursera, falar disso (OU da stora, que tem no ppt do SVM)
1. If n is large (1–10,000) and m is small (10–1000) : use logistic regression or SVM with a linear kernel.
2. If n is small (1–10 00) and m is intermediate (10–10,000) : use SVM with (Gaussian, polynomial etc) kernel
daí decidiu-se nao usar mais kernels do que estes, pk pronto
> o nosso caso, maybe os dados caiam mais no caso 2, mas decidiu se começar pelo 1, pk tal como foi dito é maiss imples e sendo lineares sao mais rapidos 

## 3.1 
Logistic regression is great in a low number of dimensions and when the predictors don't suffice to give more than a probabilistic estimate of the response. SVMs do better when there's a higher number of dimensions, and especially on problems where the predictors do certainly (or near-certainly) determine the responses. [https://www.quora.com/What-is-the-difference-between-Linear-SVMs-and-Logistic-Regression] o que torna natural que seja melhor o Linear Kernel. nota: este link tem bue respostas interessantes

## 4. Neural NetWork
(do ur shit)



