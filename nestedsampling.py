import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Nested Sampling Algorithm for Bayesian Parameter Fitting and Model Selection (With Python)
Philip Mocz (2023), @PMocz

Apply the Nested Sampling Monte Carlo algorithm to fit exoplanet radial 
velocity data and estimate the posterior distribution of the model 
parameters

"""


def rv_model( V, K, w, e, P, chi, t ):
	"""
    Calculate the radial velocity curve of exoplanet system
    V      systemic velocity
    K      velocity semiamplitude
    w      longitude of periastron
    e      eccentricity
    P      orbital period
    chi    fraction of an orbit, prior to t=0, at which periastron occured
    t      array of times to output radial velocities
	"""
	# initialize RV curve
	rv = 0*t + V

		
	# 1. calculate mean anomaly M (in [0,2pi])
	M = ( 2.0*np.pi / P ) * ( t + chi*P )
	M = M % (2.0*np.pi)
	
	# 2. calculate eccentric anomaly E
	#    by solving Kepler's equation with Newton-Raphson iterator
	E = 0*t
	tolerance = 1.0e-8
	max_iter = 100
	for i in range(len(t)):
		Ei = np.pi
		# f = @(E) E - e*sin(E) - M[i]
		# f_prime = @(E) = 1 - e*np.cos(E)
		it = 0
		dE = 1
		while (it <= max_iter) and (np.abs(dE) > tolerance):
			dE = -(Ei - e*np.sin(Ei) - M[i])/(1.0 - e*np.cos(Ei))   # dE = -f(Ei)/f_prime(Ei)
			Ei += dE
			it += 1

		if np.abs(dE) > tolerance:
			print('Error: Newton-Raphson Iterative failed!')

		E[i] = Ei

	# 3. calculate true anomaly f 
	#    (http://en.wikipedia.org/wiki/True_anomaly)
	f = 2.0*np.arctan2( np.sqrt(1.0+e)*np.sin(E/2.0), np.sqrt(1.0-e)*np.cos(E/2.0) )

	# 4. add effect of the planet to the RV curve
	rv -= K * ( np.cos(f+w) + e*np.cos(w) )

	return rv


def log_prior(theta, theta_lo, theta_hi):
	"""
    Calculate the log of the priors for a set of parameters 'theta'
    We assume uniform priors bounded by 'theta_lo' and 'theta_hi'
	"""
	return -np.sum( np.log(theta_hi - theta_lo))
	
	
def propose(theta_prev, sigma_theta, theta_lo, theta_hi):
	"""
    propose a new set of parameters 'theta' given the previous value
    'theta_prev' in the Markov chain. Choose new values by adding a 
    random Gaussian peturbation with standard deviation 'sigma_theta'.
    Make sure the new proposed value is bounded between 'theta_lo'
    and 'theta_hi'
	"""
	# propose a set of parameters
	theta_prop = np.random.normal(theta_prev, sigma_theta)
	
	# reflect proposals outside of bounds
	too_hi = theta_prop > theta_hi
	too_lo = theta_prop < theta_lo

	theta_prop[too_hi] = 2*theta_hi[too_hi] - theta_prop[too_hi]
	theta_prop[too_lo] = 2*theta_lo[too_lo] - theta_prop[too_lo]
	
	return theta_prop


def eval_model(theta, t):
	"""
    Evaluate the RV model given parameters 'theta' at times 't'
	"""
	V = theta[0]
	K = theta[2]
	w = theta[3]
	e = theta[4]
	P = theta[5]
	chi = theta[6]
	return rv_model( V, K, w, e, P, chi, t )


def log_likelihood_eval(rv_pred, s, rv_data, rv_errors):
	"""
    Evaluate the log likelihood of a model 'rv_pred' given the data
	"""
	return np.sum( np.log(1.0/np.sqrt(2.0*np.pi*(rv_errors**2+s**2))) + (-(rv_pred-rv_data)**2 / (2*(rv_errors**2+s**2))) )

def log_likelihood(theta, t, rv_data, rv_errors):
	"""
    Evaluate the log likelihood of a model parameters 'theta' given the data
	"""
	s = theta[1]
	rv_pred = eval_model(theta, t)
	return log_likelihood_eval(rv_pred, s, rv_data, rv_errors)

def log_posterior(theta, t, rv_data, rv_errors):
	"""
    Evaluate the log posterior of a model parameters 'theta' given the data
    Note: since our priors are constant, we ignore adding it
	"""
	return log_likelihood(theta, t, rv_data, rv_errors)


def main():
	""" Fit Radial Velocity curve parameters with Nested Sampling """
	
	# set the random number generator seed
	np.random.seed(917)
	
	# Generate Mock Data
	N_params = 7
	V   = 0.0     # systemic velocity
	s   = 1.0     # stellar jitter (Gaussian error)
	K   = 40.0    # velocity semiamplitude
	w   = 4.2     # longitude of periastron
	e   = 0.3     # eccentricity
	P   = 45.0    # orbital period
	chi = 0.4     # fraction of an orbit, prior to t=0, at which periastron occured
	
	t_max = 100.0
	t  = np.linspace(0, t_max,  30)  # array of times
	tt = np.linspace(0, t_max, 200)  # dense array of times
	
	# array of Gaussian measurement errors at each time
	rv_errors = 2.0 + np.abs(np.random.normal(0.0, 1.0, size=t.size))	
	
	# exact radial velocity curve, for plotting
	rv_exact = rv_model( V, K, w, e, P, chi, tt)
	
	# mock data set (30 points)
	rv_data  = rv_model( V, K, w, e, P, chi, t) 
	for i in range(len(t)):
		rv_data[i] += np.random.normal(0.0,np.sqrt(s**2 + rv_errors[i]**2))
	
	# Bounds on Priors
	V_bounds   = np.array([-4.0,      4.0])
	s_bounds   = np.array([0.5,       1.2])
	K_bounds   = np.array([20.0,     60.0])
	w_bounds   = np.array([0.0, 2.0*np.pi])
	e_bounds   = np.array([0.0,       0.5])
	P_bounds   = np.array([30.0,     50.0])
	chi_bounds = np.array([0.0,       1.0])
	
	theta_bounds = np.array([V_bounds, s_bounds, K_bounds, w_bounds, e_bounds, P_bounds, chi_bounds])
	theta_lo = theta_bounds[:,0]
	theta_hi = theta_bounds[:,1]
	
	sigma_theta_fac = 0.02 
	sigma_theta = sigma_theta_fac * (theta_hi - theta_lo)

	# prep figure
	fig = plt.figure(figsize=(6,6), dpi=80)
	
	# plot exact rv curve and mock data
	ax1 = plt.subplot(2, 1, 1)
	plt.plot(tt, rv_exact, 'k')
	plt.errorbar(t, rv_data, rv_errors, fmt='ko')
	plt.xlabel("time [day]")
	plt.ylabel("radial velocity [m/s]")
	plt.xlim([-10,t_max+10])
	plt.ylim([-60, 60])
	
	# plot Bayesian evidence
	ax2 = plt.subplot(3, 1, 3)
	plt.xlabel("step (#)")
	plt.ylabel("log Z (Bayesian evidence)")
	plt.xlim([0, 600])
	plt.ylim([-900,0])
		
	# Carry out Nested Sampling to get best-fit parameters	
	N_live = 20
	N_mcmc = 10
	N = 600
	
	theta_live = np.zeros((N_live,N_params))
	L_live = np.zeros((N_live))
	
	theta = np.zeros((N,N_params))
	L = np.zeros((N))
	
	delta_logZi = 1
	last_logZ = -np.inf
	
	
	# (1) draw sample of 'live' particles from priors 
	#     and calculate their log-likelihoods
	for i in range(N_live):
		theta_live[i,:] = np.random.uniform(theta_lo, theta_hi)
		L_live[i] = log_likelihood(theta_live[i,:], t, rv_data, rv_errors)
			
	
	# Nested Sampling
	for i in range(N):
		
		# (2) sort the likelihoods and store the smallest one
		sorted_idx = np.argsort(L_live)
		theta_live = theta_live[sorted_idx,:]
		L_live = L_live[sorted_idx]  
		
		# (store the smallest one)
		theta[i,:] = theta_live[0,:]
		L[i] = L_live[0]
		
		
		# (3) replace the point with a higher likelihood sampled point, 
		#     using Metropolis-Hastings MCMC (take N_mcmc steps)
		rnd_index = np.random.randint(N_live)
		theta_new = theta_live[rnd_index,:]
		N_accept = 0
		N_reject = 0
		
		while (N_accept < N_mcmc):
			
			# take random step using the proposal distribution
			theta_prop = propose(theta_new, sigma_theta, theta_lo, theta_hi)

			L_prop = log_likelihood(theta_prop, t, rv_data, rv_errors)
			L_new = log_likelihood(theta_new, t, rv_data, rv_errors)
			
			prior_prop = log_prior(theta_prop, theta_lo, theta_hi)
			prior_new = log_prior(theta_new, theta_lo, theta_hi)
		
			U = np.random.uniform(0.0, 1.0)
			A = prior_prop/prior_new
			
			N_reject += 1
			if( (L_prop > L[i]) and (U < A) ):
				theta_new = theta_prop
				L_new = L_prop
				N_accept += 1
				N_reject -= 1
		
		# replace
		theta_live[0] = theta_new
		L_live[0] = L_new

		# update the proposal Gaussian widths (i.e., adaptive mcmc step sizes)
		if (N_accept > N_reject):
			if (sigma_theta_fac < 0.1):
				sigma_theta_fac *= np.exp(1.0/N_accept)
		else:
			if (sigma_theta_fac > 1.0e-4):
				sigma_theta_fac *= np.exp(-1.0/N_reject)

		sigma_theta = sigma_theta_fac * (theta_hi - theta_lo)

		# calculate evidence weights W and Bayesian evidence Z
		W = np.exp(-np.arange(1,i+2)/N_live)
		# compute Bayesian evidence Z
		maxL = np.max(L[0:i+1])
		logZ = np.log( np.sum( np.exp(L[0:i+1] - maxL) * W) ) + maxL
		delta_logZi = logZ - last_logZ 
		last_logZ = logZ 

		# plot proposed function
		if (i % 10) == 0:
			print("step i=", i)
			print("    logZ=", logZ)
			print("    delta_logZi=", delta_logZi)
			print("    N_accept=", N_accept, " N_reject=", N_reject)
			print("    theta=", theta[i,:])
			rv = eval_model(theta[i,:], tt)
			ax1.plot(tt, rv, linewidth=0.5, color=plt.cm.jet(i/N))
			ax2.plot(i, logZ, 'ko')
			plt.pause(0.0001)
			plt.draw()
			
			
	# Save figure
	plt.savefig('nestedsampling.png',dpi=240)
	plt.show()
	
	
	# Re-sample results with proper weights to obtain posteriors
	N_resample = 100000
	theta_posterior = np.zeros((N_resample,N_params))
	
	W_sample = np.exp(L + np.log(W) - logZ)
	W_sample /= np.sum(W_sample)
	resample_idx = np.random.choice(range(N), N_resample, p=W_sample)
	for i in range(N_resample):
		theta_posterior[i,:] = theta[resample_idx[i],:]

	
	# Plot Posteriors
	
	fig, ((ax0, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(6,4), dpi=80)
	
	n_bins = 20
	ax0.hist(theta_posterior[:,0], n_bins, histtype='step', fill=True)
	ax0.axvline(V, color='r', linewidth=1)
	ax0.set_title('V posterior')
	ax2.hist(theta_posterior[:,2], n_bins, histtype='step', fill=True)
	ax2.axvline(K, color='r', linewidth=1)
	ax2.set_title('K posterior')
	ax3.hist(theta_posterior[:,3], n_bins, histtype='step', fill=True)
	ax3.axvline(w, color='r', linewidth=1)
	ax3.set_title('w posterior')
	ax4.hist(theta_posterior[:,4], n_bins, histtype='step', fill=True)
	ax4.axvline(e, color='r', linewidth=1)
	ax4.set_title('e posterior')
	ax5.hist(theta_posterior[:,5], n_bins, histtype='step', fill=True)
	ax5.axvline(P, color='r', linewidth=1)
	ax5.set_title('P posterior')
	ax6.hist(theta_posterior[:,6], n_bins, histtype='step', fill=True)
	ax6.axvline(chi, color='r', linewidth=1)
	ax6.set_title('chi posterior')
	
	fig.tight_layout()
	
	
	# Save figure
	plt.savefig('nestedsampling2.png',dpi=240)
	plt.show()
	    
	return 0



if __name__== "__main__":
  main()

