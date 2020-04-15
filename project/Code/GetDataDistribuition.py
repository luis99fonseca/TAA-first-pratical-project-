import numpy as np
import matplotlib.pyplot as plt

dt01 = {0: 164, 1: 165, 2: 165, 3: 165, 4: 165, 5: 166, 6: 165, 7: 165, 8: 166, 9: 163}
dt02 = {0: 41, 1: 41, 2: 41, 3: 41, 4: 42, 5: 41, 6: 42, 7: 41, 8: 42, 9: 41}
z = {}
for k in dt01:
    z[k] = dt01[k] + dt02[k]
print(z)

plt.bar(z.keys(), z.values())
plt.bar(dt01.keys(), dt01.values())
plt.legend(["Testing Size", "Training Size"])
plt.xlabel("Classes")
plt.ylabel("Quantidade de Exemplos")
plt.title("Distruibuição de Dados por Classe")
plt.show()

total = 0
for i in dt01.values():
    total+=i
for i in dt02.values():
    total+=i
print(total)
