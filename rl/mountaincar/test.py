import numpy as np

a = list()

a.append([(-1, 2, 3), True])
a.append([(2, 2, 3), False])
a.append([(-3, 2, 3), True])
a.append([(4, 2, 3), False])
a.append([(5, 2, 3), True])

print(a)
arr1 = np.array([x[1] for x in a])
arr = np.array([x[1] for x in a])
arr_state = np.array(a)
print(arr)
print(arr[arr == True])
i = np.argmax(arr == True)
print(np.argmax(arr == True))
print(arr1[0: i + 1])
# print(np.any(arr, False))
# np.any(a[:][0])
print("====================")
print(arr_state)
print(arr_state[:, 1] == False)
idx = arr_state[:, 1] == False
print(arr_state[idx])

np.dot([0.2, 0.3, 0.4], [1, 2, 3])
print(np.sum(np.multiply([0.2, 0.3, 0.4], [1, 2, 3])))
print(np.sum(np.dot([0.2, 0.3, 0.4], [1, 2, 3])))
