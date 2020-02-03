
#################################################################################################
#																								#
#	MULTI-ARMED BANDITS ---- 10-ARM TESTBED SOFTMAX METHOD										#
#																								#
#	Author: Kaustubh Joshi      															#
#																								#
#	References: 																				#
#	1) Sutton, R.S. and Barto, A.G., 2018. Reinforcement learning: An introduction. MIT press	#
#	2) GitHub: 																					#
#		i)	Sahana Ramnath - https://github.com/SahanaRamnath/MultiArmedBandit_RL				#
#		ii) Jett D. Lee	   - https://github.com/jettdlee/10_armed_bandit						#
#																								#
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

T = [0.01,0.2,1,10]		# Set of Temperature values

#Expected Reward for a selected action
q_t = np.random.normal(0,1,(n,k))		# q(a) = E [ R | A = a ]

#Optimal Action
A_t = np.argmax(q_t,1)					# A_t = argmax [ Q_t (a) ]

#Initialize Plots
f1 = plt.figure().add_subplot(111)
f2 = plt.figure().add_subplot(111)
f1.title.set_text(r'$Temperature$-greedy : Average Reward Vs Steps for 10 arms')
f1.set_ylabel('Average Reward')
f1.set_xlabel('Steps')
f2.title.set_text(r'$temperature$ : $\%$ Optimal Action Vs Steps for 10 arms')
f2.set_ylabel(r'$\%$ Optimal Action')
f2.set_xlabel('Steps')
f2.set_ylim(0,100)

Legend_Entries1 = []
Legend_Text1 = []
Legend_Entries2 = []
Legend_Text2 = []

for temp in range(len(T)):

	print('Start trials for temperature = ', T[temp])
	time_e = time.time()

	# Initialize Matrices 
	Q = np.zeros((n,k))			# Estimated Reward	
	N = np.ones((n,k)) 			# Number of Times each Arm was chosen

	# Pull Each Arm atleast once. Therefore, assign random value > 1 for each arm
	#Initial pull for all arms
	Q_i = np.random.normal(q_t,1)	

	R_t = [0] 					#Initialize vector for reward values for each epsilon
	R_t.append(np.mean(Q_i))
	R_t_opt = []				# Optimal Reward for each epsilon

	for pull in range(2, p + 1):

		#print(pull)
		R_p = []				# Initialize vector for all rewards for the pull
		count_opt_arm_pulls = 0	# Initialize counter for counting number of pulls of the optimal pulls

		for i in range(n):

			#print(pull)
			#print(i)
			Q_ex = np.exp(Q[i]/T[temp])
			Q_softmax = Q_ex/np.sum(Q_ex)

			j = np.random.choice(range(k),1,p=Q_softmax)

			temp_R = np.random.normal(q_t[i][j],1)
			R_p.append(temp_R)

			if j == A_t[i]:

				count_opt_arm_pulls = count_opt_arm_pulls + 1

			N[i][j] = N[i][j] + 1
			Q[i][j] = Q[i][j] + (temp_R - Q[i][j]) / N[i][j]

		R_p_avg = np.mean(R_p)
		R_t.append(R_p_avg)
		R_t_opt.append(float(count_opt_arm_pulls)*100/n)

	f1.plot(range(0,p+1),R_t)
	f2.plot(range(2,p+1),R_t_opt)

	p1 = f1.plot(range(0,p+1),R_t)
	p2 = f2.plot(range(2,p+1),R_t_opt)

	Legend_Entries1.append(p1)
	Legend_Entries2.append(p2)

	if (T[temp] == 0):

		print("Temperature = 0")
		Legend_Text1.append(r"$T = $"+str(T[temp])+" (greedy) ")
		Legend_Text2.append(r"$T = $"+str(T[temp])+" (greedy) ")

	else:

		Legend_Text1.append(r"$T = $"+str(T[temp]))
	
		Legend_Text2.append(r"$T = $"+str(T[temp]))
	
	#print(Legend_Text1)
	print('Trials done for temperature = ', T[temp])
	print("Execution Time for temperature " + str(T[temp]) + "  = %s" % (time.time() - time_e) )

print("Total Execution time: %s seconds" % (time.time() - start_time))
f1.legend((Legend_Text1),loc='best')
f2.legend((Legend_Text2),loc='best')
plt.show()
