"""
@author Megan Fass, Kieran Kitchener

T2 Take One: inital equations
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# for P[Aa-aa]


def eq1(w, t, lamed):
    prob = (w * (1 + t)) / lamed
    return prob

# for the case of P[Aa-Aa]


def eq2(K, lamed):
    prob = (K) / lamed
    return prob

# case of of P[AA-AA]


def eq3(Q, lamed):
    prob = (Q) / lamed
    return prob

# case of P[AA-Aa]


def eq4(J, lamed):
    prob = (J) / lamed
    return prob

# case of P[AA-aa]


def eq5(v, z, lamed):
    prob = (v * (1 + z)) / lamed
    return prob

# case of p[aa-aa]


def eq6(u, p, lamed):
    prob = (u * (1 + p)) / lamed
    return prob

# lambda is a reserved word; combines all populations + level of preference


def lamed(u, p, v, t, z, w, K, Q, J, N):
    l = u * (1 + p) + v * (1 + z) + w * (1 + t) + K + Q + J
    return l


def popChange(N, t):
    N_t = N * 2
    return N_t


# initial pop. of each diploid pair; pos integers
# u = ini. pop of [aa-aa]
u = float(sys.argv[1])
# v = ini. pop of [AA-aa]
v = float(sys.argv[2])
# w = [Aa-aa]
w = float(sys.argv[3])
# ini pop. [AA-AA]
K = float(sys.argv[4])
# [ AA-Aa]
Q = float(sys.argv[5])
# [Aa-Aa]
J = float(sys.argv[6])

# preferability; range is from -1 to 1
# pref of u
p = float(sys.argv[7])
# pref of v
z = float(sys.argv[8])
# pref of w
t = float(sys.argv[9])

gen = 10


w_list = list()
w_list.insert(0, w)
w_prob = list()

K_list = list()
K_list.insert(0, K)
K_prob = list()

Q_list = list()
Q_list.insert(0, Q)
Q_prob = list()

J_list = list()
J_list.insert(0, J)
J_prob = list()

v_list = list()
v_list.insert(0, v)
v_prob = list()

u_list = list()
u_list.insert(0, u)
u_prob = list()

N_0 = u + v + w + K + Q + J

N_graph = list()
N_graph.insert(0, N_0)

# h = lamed(u, p, v, z, w, K, Q, J, N_0)

n = 1
N = N_0
t = 1
while n <= gen:
    print ""
    h = lamed(u, p, v, z, t, w, K, Q, J, N_0)

    print "Gen" + str(n)
    prob1 = eq1(w, z, h)
    w_prob.append(prob1)
    prob2 = eq2(K, h)
    K_prob.append(prob2)
    prob3 = eq3(Q, h)
    Q_prob.append(prob3)
    prob4 = eq4(J, h)
    J_prob.append(prob4)
    prob5 = eq5(v, t, h)
    v_prob.append(prob5)
    prob6 = eq6(u, p, h)
    u_prob.append(prob6)

    print "lamed: " + str(h)
    print "eq1 probability " + str(prob1)
    print "eq2 probability " + str(prob2)
    print "eq3 probability " + str(prob3)
    print "eq4 probability " + str(prob4)
    print "eq5 probability " + str(prob5)
    print "eq6 probability " + str(prob6)

    total = prob1 + prob2 + prob3 + prob4 + prob5 + prob6

    print "total prob: " + str(total) + " for population " + str(N)
    N = popChange(N, t)
    N_graph.append(N)
    n += 1
    t += 1

    w = N * prob1
    w_list.append(w)
    K = N * prob2
    K_list.append(K)
    Q = N * prob3
    Q_list.append(Q)
    J = N * prob4
    J_list.append(J)
    v = N * prob5
    v_list.append(v)
    u = N * prob6
    u_list.append(u)


# plt.xlabel('Time(s)')
# plt.ylabel('Conc (M)')
plt.title('shitall')
# plt.axis([0,tmax,0,1.0])
plt.plot(w_prob, 'red')
# plt.plot(K_prob, 'cyan')
# plt.plot(Q_prob, 'green')
plt.plot(J_prob, 'blue')
plt.plot(v_prob, 'black')
plt.plot(u_prob, 'purple')
plt.savefig("shitall")
plt.show()

plt.title('population change of diploid shit')
plt.plot(w_list, 'red')
plt.plot(K_list, 'cyan')
plt.plot(Q_list, 'green')
plt.plot(J_list, 'blue')
plt.plot(v_list, 'black')
plt.plot(u_list, 'purple')
# plt.plot(N_graph, "pink")
plt.savefig("population")
plt.show()
