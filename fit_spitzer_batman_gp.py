"""
06/27/2019
Input:
argv[1] -- prefix of your output files' names.
argv[2] -- input time&flux data file, which is output from "extract_image_frameAve.py"
argv[3] -- how many seconds do you want to cut from the beginning of observation.

setup:
1.Fixed period, eccentricity and limb-darkening: change these inputs in "transit_params".
2.Rp/Rs, T0, a/Rs and inclination are free parameters: their initial values are set in "param_ini", and you can change their boundaries in the beginning of function "lnlike".
3.Gaussian Process is set up to model uncertainties.
4.Change the number of mcmc steps in "sampler.run_mcmc", and number of burn-in chains in "sampler.chain".
"""

#import matplotlib
#matplotlib.use('Agg')
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
from math import *
import scipy.optimize as op
import emcee
import corner
from lmfit import minimize, Parameters, Minimizer
import batman
import pylab as P
import celerite
from celerite import terms
from celerite.modeling import Model

time = str(sys.argv[1])

data_file = str(sys.argv[2])

cut_initial_secs = float(sys.argv[3])

FLUXCONV = 0.1257    #in header
GAIN = 3.7          #in header
SpitzerStartTime = None # BMJD_OBS days

#transit_duration = 14.737 * 60 * 60 # sec from Holczer

transit_params = batman.TransitParams()
transit_params.per =9.4903*24.0*60*60
transit_params.limb_dark = "quadratic"
transit_params.u = [0.0998, 0.1658]                #limb darkening coefficients
transit_params.ecc = 0.078 #eccentricity
#transit_params.a = 24.4706                #semi-major axis (in units of stellar radii)
#transit_params.inc = 89.45
transit_params.w = 90.0
#
#def inclination_calc(transit_dur, transit_period, a):
#    temp1 = (pi*a*transit_dur/transit_period)**2.0
#    print temp1
#    temp2 = sqrt(1.0866*1.0866-temp1)/a
#    inclin = acos(temp2)*180.0/pi
#    return inclin
#
#transit_params.inc = inclination_calc(transit_duration, transit_params.per, transit_params.a)
#print(transit_params.inc)

flux = []
tot_flux = np.array([])
time_array = np.array([])
for row in open(data_file, "r"):
    ps = row.replace(" \n", "").split(" ")
    ps = [float(ps[i]) for i in range(len(ps))]
    if SpitzerStartTime == None:
        SpitzerStartTime = float(ps[0])
        time_array = np.append(time_array, 0.0)
    else:
        time_array = np.append(time_array, (float(ps[0])-SpitzerStartTime)*24.0*60*60)

    flux_pixels = np.array(ps[1:])
    tot_flux_tmp = np.sum(flux_pixels)
    tot_flux = np.append(tot_flux, tot_flux_tmp)
    flux.append(flux_pixels/tot_flux_tmp)
close(data_file)
flux = np.array(flux)
print(np.shape(tot_flux))
print(np.shape(flux))

N_time, N_pixel = np.shape(flux)

mean = np.mean(tot_flux)
print(mean, len(tot_flux))
tot_flux = tot_flux/mean
tot_flux = np.resize(tot_flux, N_time)

# cut the initial some seconds off the data arrays ###############
for i in range(len(time_array)):
    if time_array[i] > cut_initial_secs:
	cut_index = i
	break
time_array = time_array[cut_index:]
tot_flux = tot_flux[cut_index:]
flux = flux[cut_index:, :]
N_time -= cut_index
###############################################################

###### mask out data points with relative flux > 1.2 or < 0.8 #################
std_tmp = np.std(tot_flux)
clean = False
while clean == False:
    clean = True
    count_delete = 0
    for i in range(N_time):
        if tot_flux[i] > np.mean(tot_flux[:])+5.0*std_tmp or tot_flux[i] < np.mean(tot_flux[:])-5.0*std_tmp:
            count_delete = count_delete+1
            clean = False
            if i == N_time-1:
                break
            for j in range(i, N_time-1):
                tot_flux[j] = tot_flux[j+1]
                flux[j, :] = flux[j+1, :]
                time_array[j] = time_array[j+1]

    N_time -= count_delete
    flux = np.resize(flux, (N_time, N_pixel))
    tot_flux = np.resize(tot_flux, N_time)
    std_tmp = np.std(tot_flux)

time_array = np.resize(time_array, N_time)
tot_flux = tot_flux/np.mean(tot_flux)
time_step = (time_array[-1]-time_array[0])/(N_time-1)
##############################################################

######### calculate the uncertainty on each data point ##########################
mean = np.mean(tot_flux)
err_frac = 1.0/sqrt(mean/FLUXCONV*GAIN*time_step)
#err_frac *= 1.0
print("average uncertainty fraction: ", err_frac)
print("mean: ", mean)
print("total flux array: ", tot_flux)
#############################################################

print "N_pixel: ", N_pixel, N_time

flux = np.concatenate((flux, np.resize(time_array, (N_time, 1))), axis=1)
print(flux, time_array)


param_ini = np.zeros(int(N_pixel+9))+1.0
param_ini[N_pixel] = 1.0e-10     #v in trend (1 + exp(-t/tau) + v*t)
param_ini[N_pixel+1] = 4.84974568441e+14     #tau in trend (1 + exp(-t/tau) + v*t)
param_ini[N_pixel+2] = 0.03
param_ini[N_pixel+3] = time_array[len(time_array)-1]/2.0
#param_ini[N_pixel+4] =             # correlation time scale (rho in the Matern model)
param_ini[N_pixel+5] = -500.0         # in Matern model: log(sigma)  (= 0.5*log(variance))
param_ini[N_pixel+6] = 89.14   # inclination
param_ini[N_pixel+7] = 24.4706   # a/R
param_ini[N_pixel+8] = 0.627   # in GP, the white noise term

weight = np.zeros([N_pixel+1, 1])

class MeanModel(Model):
    parameter_names = ("param0",)
    for i in range(1, N_pixel+6):
        parameter_names += ("param"+str(i),)

    def get_value(self):
        params = self.parameter_vector
        transit_params.rp = params[N_pixel+2]
        transit_params.t0 = params[N_pixel+3]
        transit_params.inc = params[N_pixel+4]
        transit_params.a = params[N_pixel+5]
        m = batman.TransitModel(transit_params, time_array)
        transit_flux = m.light_curve(transit_params)
        
        for i in range(N_pixel+1):
            weight[i, 0] = params[i]

        component1 = np.dot(flux, weight)  # multiply flux matrix by weight.
        out_flux = (np.reshape(component1, N_time)+np.exp(-time_array/params[N_pixel+1]))*transit_flux   # add the exp trend, and multiply with transit model.
        return out_flux


def lnlike(params_in, y_obs):
    if params_in[N_pixel+1]>0.0 and 0.02<params_in[N_pixel+2]<0.04 and 0.0<params_in[N_pixel+3]<time_array[-1] and -10.0<params_in[N_pixel+4]<11.4 and params_in[N_pixel+5]<0.0 and 85.0<params_in[N_pixel+6]<95.0 and 20.0<params_in[N_pixel+7]<30.0 and 0.0<params_in[N_pixel+8]<100.0:
        print(params_in)
        # Set up the GP model
        params_mean_model = np.resize(params_in, int(N_pixel+4))
        params_mean_model = np.append(params_mean_model, params_in[N_pixel+6])
        params_mean_model = np.append(params_mean_model, params_in[N_pixel+7])
        mean_model = MeanModel(*params_mean_model)
        mean_model_out = mean_model.get_value()
        
        kernel = terms.Matern32Term(log_sigma=params_in[N_pixel+5], log_rho=params_in[N_pixel+4], eps = 0.000001)
        gp = celerite.GP(kernel, mean=0.0, fit_mean=False)
        gp.compute(time_array, err_frac*params_in[N_pixel+8], check_sorted=True)

        logprob = -np.sum(((y_obs-mean_model_out)/err_frac)**2.0/2.0) - log(err_frac)*len(time_array) - 0.5*len(time_array)*log(2.0*pi) + gp.log_likelihood(y_obs-mean_model_out)
        #reduced_chisq = np.sum(((y_obs-mu)/err_frac)**2.0)/(len(y_obs)-len(params_in))
        print(logprob)
        return logprob
    print("-inf")
    return -np.inf

#nll = lambda *args: -lnlike(*args)
#result = op.minimize(nll, param, args=(tot_flux,))
#param = result["x"]
#print("finish primary fitting here!!")

ndim, nwalkers = len(param_ini), int(12*len(param_ini))
#print(ndim, nwalkers)
#pos = [result["x"] + 0.0001*np.random.randn(ndim) for i in range(nwalkers)]
pos = np.zeros((nwalkers, ndim))
for i in range(nwalkers):
    step_size = np.zeros(ndim)
    for k in range(ndim):
        step_size[k] =0.001*param_ini[k]*np.random.randn()
    pos[i] = param_ini + step_size

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(tot_flux,))
#pos, _, _, = sampler.run_mcmc(pos, 500)
#sampler.reset()

sampler.run_mcmc(pos, 800)
#fig2 = plt.figure(figsize = (9,5), dpi=120)
#plt.plot(np.arange(shape[0]), samples_check[:, 4], color = 'grey')
#fig2.savefig(time+"_param4_chain.png")
#plt.close(fig2)

samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
f_posterior = open(time+"_posterior.txt", 'w')
for row in samples:
    for ele in row:
        f_posterior.write("%s " % ele)
    f_posterior.write("\n")
f_posterior.close()

label_out=["weight"+str(i) for i in range(N_pixel)]
label_out += ["v", "tau", "Rp/Rs", "T0", "log_rho", "log_sigma", "inc", "a/Rs", "sigma_white"]
fig = corner.corner(samples, labels=label_out, quantiles=[0.16, 0.5, 0.84])
fig.savefig(time+"_mcmc_corner.pdf")
plt.close(fig)

param_out_center = np.zeros(len(param_ini))
f_out = open(str(time+'_param_out.txt'), 'w')

for i in range(len(param_ini)):
    param_out = np.percentile(samples[:, i], [16, 50, 84], axis=0)
    print(param_out)
    param_out_center[i] = param_out[1]
    f_out.write(label_out[i]+": ")
    f_out.write("%s %s %s\n" % (param_out[1], (param_out[1]-param_out[0]), (param_out[2]-param_out[1])))
f_out.close()

transit_params.rp = param_out_center[N_pixel+2]
transit_params.t0 = param_out_center[N_pixel+3]
transit_params.inc = param_out_center[N_pixel+6]
transit_params.a = param_out_center[N_pixel+7]
m = batman.TransitModel(transit_params, time_array)
transit_flux = m.light_curve(transit_params)
print(transit_flux)

out_transit_flux_sum = np.array([0.0]*N_time)
for s in samples[np.random.randint(len(samples), size=100)]:
    for i in range(N_pixel+1):
        weight[i, 0] = s[i]
    out_transit_flux_sum += tot_flux/(np.resize((np.dot(flux, weight)), N_time)+np.exp(-time_array/s[N_pixel+1]))
out_transit_flux = out_transit_flux_sum/100.

fig2 = plt.figure(figsize = (9,5), dpi=120)
plt.scatter(time_array/24./60/60+SpitzerStartTime, out_transit_flux, s=1, zorder=0, c='grey')
num_points, bin_edge = np.histogram(time_array, 100, density=False)
sum_points, bin_edge = np.histogram(time_array, 100, weights=out_transit_flux, density=False)
binned_flux = sum_points/num_points
plt.scatter((bin_edge[:-1]+bin_edge[1:])/2.0/24./60/60+SpitzerStartTime, binned_flux, s=5, zorder=1, c='blue')
plt.plot(time_array/24./60/60+SpitzerStartTime, transit_flux, color = 'r', linewidth=1.5)
#ylim([0.995, 1.005])
xlabel(r'time(BJD)', fontsize=30)
ylabel(r'fraction', fontsize=30)
plt.savefig(time+"_starTransit.pdf", bbox_inches='tight')
plt.show()
plt.close(fig2)

fig = plt.figure(figsize = (9,5), dpi=120)
plt.scatter(time_array, tot_flux, s=3, zorder=0)
# calculate 50 posterior light curves######################################
#samples2 = sampler.chain[:, 2:, :].reshape(-1, ndim)
for s in samples[np.random.randint(len(samples), size=30)]:
    kernel = terms.Matern32Term(log_sigma=s[N_pixel+5], log_rho=s[N_pixel+4], eps = 0.000001)
#    print(s)
    s_in = np.resize(s, N_pixel+4)
    s_in = np.append(s_in, s[N_pixel+6])
    s_in = np.append(s_in, s[N_pixel+7])
    mean_model = MeanModel(*s_in)
    mean_model_out = mean_model.get_value()
    gp = celerite.GP(kernel, mean=0.0, fit_mean=False)
    gp.compute(time_array, err_frac*s[N_pixel+8], check_sorted=True)
#    print(mean_model_out)
    mu = gp.predict(tot_flux-mean_model_out, time_array, return_cov=False) + mean_model_out
    plt.plot(time_array, mu, color="pink", alpha=0.3, linewidth = 1.0)
################################################################
#plt.plot(time_array, out_flux_final, color = 'r')
#ylim([0.98, 1.02])
plt.savefig(time+"_star.pdf", bbox_inches='tight')
plt.show()

