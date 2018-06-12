import numpy as np
import astropy.constants as const
import astropy.units as u

k_B = const.k_B.to("eV / kK").value
c3 = (4 * np.pi * const.sigma_sb.to("erg s-1 Rsun-2 kK-4").value)**-0.5 / 1000. # Rsun --> kiloRsun

class Model():
	def __init__(self, func, input_names, units):
		self.func = func
		self.input_names = input_names
		self.units = units
		self.nparams = len(input_names)
		self.axis_labels = ['${}$ ({:latex_inline})'.format(var, u.Unit(unit)).replace('1 \\times ', '')
				   		    for var, unit in zip(input_names, units)]

	def __call__(self, *args, **kwargs):
		return self.func(*args, **kwargs)

def shock_cooling(t_in, v_s, M_env, f_rho_M, R, t_exp=0., kappa=1., n=1.5, RW=False):
    '''time in days, velocity in 10**8.5 cm/s, masses in M_sun,
       progenitor radius in 10**13 cm, opacity in 0.34 cm**2/g,
       color temperature in kK, blackbody radius in 1000 R_sun'''
    
    if n == 1.5:
        A = 0.94
        a = 1.67
        alpha = 0.8
        epsilon_1 = 0.027
        epsilon_2 = 0.086
        L_0 = 2.0e42
        T_0 = 1.61
        Tph_to_Tcol = 1.1
    elif n == 3.:
        A = 0.79
        a = 4.57
        alpha = 0.73
        epsilon_1 = 0.016
        epsilon_2 = 0.175
        L_0 = 2.1e42
        T_0 = 1.69
        Tph_to_Tcol = 1.0
    else:
        print('not a valid n')
        
    if RW:
        a = 0.
        Tph_to_Tcol = 1.2
    
    t = t_in.reshape(-1, 1) - t_exp
    L_RW = L_0 * (t**2 * v_s / (f_rho_M * kappa))**-epsilon_2 * v_s**2 * R / kappa
    t_tr = 19.5 * (kappa * M_env / v_s)**0.5
    L = L_RW * A * np.exp(-(a * t / t_tr)**alpha)
    T_ph = T_0 * (t**2 * v_s**2 / (f_rho_M * kappa))**epsilon_1 * kappa**-0.25 * t**-0.5 * R**0.25
    T_col = T_ph * Tph_to_Tcol
    T_K = np.squeeze(T_col) / k_B
    R_bb = c3 * np.squeeze(L)**0.5 * T_K**-2
    t_min = 0.2 * R / v_s * np.minimum(0.5, R**0.4 * (f_rho_M * kappa)**-0.2 * v_s**-0.7)
    t_max = 7.4 * (R / kappa)**0.55
    
    return T_K, R_bb, t_min, t_max

def shock_cooling2(t_in, T_1, L_1, t_tr, t_exp=0., n=1.5, RW=False):
    if n == 1.5:
        a = 1.67
        alpha = 0.8
        epsilon_1 = 0.027
        epsilon_2 = 0.086
    elif n == 3.:
        a = 4.57
        alpha = 0.73
        epsilon_1 = 0.016
        epsilon_2 = 0.175
    else:
        print('not a valid n')

    if RW:
        a = 0.
        
    t = t_in.reshape(-1, 1) - t_exp
    
    epsilon_T = 2 * epsilon_1 - 0.5
    epsilon_L = -2 * epsilon_2
    
    T_K = np.squeeze(T_1 * t**epsilon_T)
    L = np.squeeze(L_1 * np.exp(-(a * t / t_tr)**alpha) * t**epsilon_L) * 1e42
    R_bb = c3 * L**0.5 * T_K**-2
    t_max = (8.12 / T_1)**(epsilon_T**-1)

    return T_K, R_bb, 0., t_max

ShockCooling = Model(shock_cooling,
	[
	    'v_\\mathrm{s*}',
	    'M_\\mathrm{env}',
	    'f_\\rho M',
	    'R',
	    '\\Delta t_0'
	],
	[
	    10.**8.5 * u.cm / u.s,
	    u.Msun,
	    u.Msun,
	    1e13 * u.cm,
	    u.d
	]
)

ShockCooling2 = Model(shock_cooling2,
	[
	    'T_1',
	    'L_1',
	    't_\\mathrm{tr}',
	    '\\Delta t_0'
	],
	[
	    u.kK,
	    1e42 * u.erg / u.s,
	    u.d,
	    u.d
	]
)

def log_flat_prior(p):
    return p**-1

def flat_prior(p):
    return np.ones_like(p)
