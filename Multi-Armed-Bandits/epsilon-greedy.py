
#################################################################################################
#																								                                                #
#	MULTI-ARMED BANDITS ---- 10-ARM TESTBED														                            #								                  #
#																								                                                #
#	Author: Kaustubh Joshi | NA16B024															                                #
#																								                                                #
#	References: 																				                                          #
#	1) Sutton, R.S. and Barto, A.G., 2018. Reinforcement learning: An introduction. MIT press	    #
#	2) GitHub: 																					                                          #
#		i)	Sahana Ramnath - https://github.com/SahanaRamnath/MultiArmedBandit_RL				            #
#		ii) Jett D. Lee	   - https://github.com/jettdlee/10_armed_bandit						                #
#																								                                                #
#################################################################################################

import numpy as np 
import matplotlib.pyplot as plt 
import random
import time

#Begin time counter
start_time = time.time()		

n = 2000						# Number of bandit problems
k = 10							# Number of Arms
p = 1000						# Number of plays

E = [0,0.01,0.02,0.1,0.2,1]		# Set of Epsilon values for Epsilon-Greedy Method

#Expected Reward for a selected action
q_t = np.random.normal(0,1,(n,k))		# q(a) = E [ R | A = a ]

#Optimal Action
A_t = np.argmax(q_t,1)					# A_t = argmax [ Q_t (a) ]

#Initialize Plots
f1 = plt.figure().add_subplot(111)
f2 = plt.figure().add_subplot(111)
f1.title.set_text(r'$\epsilon$-greedy : Average Reward Vs Steps for 10 arms')
f1.set_ylabel('Average Reward')
f1.set_xlabel('Steps')
f2.title.set_text(r'$\epsilon$-greedy : $\%$ Optimal Action Vs Steps for 10 arms')
f2.set_ylabel(r'$\%$ Optimal Action')
f2.set_xlabel('Steps')
f2.set_ylim(0,100)

Legend_Entries1 = []
Legend_Text1 = []
Legend_Entries2 = []
Legend_Text2 = []

for e in range(len(E)):

	print('Start trials for epsilon = ', E[e])
	time_e = time.time()

	# Initialize Matrices 
	Q = np.zeros((n,k))			# Estimated Reward	
	N = np.ones((n,k)) 			# Number of Times each Arm was chosen

	# Pull Each Arm atleast once. Therefore, assign random value > 1 for each arm
	#Initial pull for all arms
	Q_i = np.random.normal(q_t,1)	

	R_e = [] 					#Initialize vector for reward values for each epsilon
	R_e.append(0)
	R_e.append(np.mean(Q_i))
	R_e_opt = []				# Optimal Reward for each epsilon

	for pull in range( 2, p + 1 ):

		R_p = []				# Initialize vector for all rewards for the pull
		count_opt_arm_pulls = 0	# Initialize counter for counting number of pulls of the optimal pulls

		for i in range(n):

			if random.random() < E[e]:

				j = np.random.randint(k)	# Choosing an arm at random

			else:

				j = np.argmax(Q[i])			# Randomly breaking ties

			if j == A_t[i]:

				count_opt_arm_pulls = count_opt_arm_pulls + 1

			R_temp = np.random.normal(q_t[i][j],1)
			R_p.append(R_temp)
			N[i][j] = N[i][j] + 1
			Q[i][j] = Q[i][j] + (R_temp - Q[i][j]) / (N[i][j])

		R_p_avg = np.mean(R_p)
		R_e.append(R_p_avg)
		R_e_opt.append(float(count_opt_arm_pulls)*100/n)

	f1.plot(range(0,p+1),R_e)
	f2.plot(range(2,p+1),R_e_opt)

	p1 = f1.plot(range(0,p+1),R_e)
	p2 = f2.plot(range(2,p+1),R_e_opt)

	Legend_Entries1.append(p1)
	Legend_Entries2.append(p2)

	if (E[e] == 0):
		print("epsilon = 0")
		Legend_Text1.append(r"$\epsilon=$"+str(E[e])+" (greedy) ")
	
		Legend_Text2.append(r"$\epsilon=$"+str(E[e])+" (greedy) ")

	else:

		Legend_Text1.append(r"$\epsilon=$"+str(E[e]))
	
		Legend_Text2.append(r"$\epsilon=$"+str(E[e]))
	
	#print(Legend_Text1)
	print('Trials done for epsilon = ', E[e])
	print("Execution Time for epsilon " + str(E[e]) + "  = %s" % (time.time() - time_e) )

print("Total Execution time: %s seconds" % (time.time() - start_time))
f1.legend((Legend_Text1),loc='best')
f2.legend((Legend_Text2),loc='best')
plt.show()
