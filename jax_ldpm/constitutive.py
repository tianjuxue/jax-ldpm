import numpy as onp
import jax
import jax.numpy as np
from jax.experimental import checkify

# https://jax.readthedocs.io/en/latest/debugging/index.html


# def safe_divide(x, y):
#     return np.where(y == 0., 0., x/y)


safe_divide = lambda x, y: np.where(y == 0., 0., x/y)
safe_plus = lambda x: 0.5*(x + np.abs(x))

def calc_st_sKt(params, aLength):
    chi = 0.99
    ft = params['ft'] # Tensile Strength
    E0 = params['E0'] # Young Modulus # GC: This should be EN=E0
    chLen = params['chLen'] # Tensile characteristic length

    def f1():
        st = ft
        # if (aLength.lt.1.0d-10) then
        #    write(*,*)'Warning: aLength in critical range; continuation may produce numerical issues.'     
        # endif
        # if (aLength.eq.chLen) then
        #   write(*,*)'Warning: aLength equals chLen; continuation may produce numerical issues.'
        # endif
        aKt = 2. * E0 / (-1. + chLen / aLength)
        return st, aKt

    def f2():
        aKt = 2. * E0 / (-1. + 1. / chi);
        st = ft * np.sqrt(np.abs(chi * chLen / aLength))
        # Write(*,*)'Warning: Edge length > tensile characteristic length'
        return st, aKt

    st, aKt = jax.lax.cond(aLength < chi * chLen, f1, f2)
    return st, aKt



def stress_fn(eps, epsV, stv, info, params):
    # Variables omitted:
    # nprops, props, rbv, RateEffectFlag

    # LDPM MATERIAL MODEL
    #  
    # LDPM Facet Constitutive Law - Cusatis/Rezakhani Oct 2019
    # Copied from Fortran code and transformed to Python code by Tianju Aug 2023
    #
    # stv state variable
    # stv[1]  Normal N stress
    # stv[2]  Shear M stress
    # stv[3]  Shear L stress
    # stv[4]  Normal N strain
    # stv[5]  Shear M strain
    # stv[6]  Shear L strain
    # stv[7]  Max normal N strain
    # stv[8]  Max shear T strain
    # stv[9]  Tensile strength                             # Useless
    # stv[10] Post-peak slope in tension                   # Useless
    # stv[11] Shear L crack opening
    # stv[12] Minimum Normal Strain                        # Not used?
    # stv[13] Normal N crack opening
    # stv[14] Shear M crack opening
    # stv[15] Total crack opening
    # stv[16] Volumetric Strain
    # stv[17] Dissipated energy density rate
    # stv[18] Dissipated energy density
    # stv[19] Dissipated energy density rate in tension
    # stv[20] Dissipated energy density in tension
    # stv[21] ENc                                          # Created by Tianju, for local use

    dns           = params['rho']           # Density
    E0            = params['E0']            # Young Modulus # GC: This should be EN=E0
    alpha         = params['alpha']         # Poisson ratio # GC: This should be alpha
    ft            = params['ft']            # Tensile Strength
    # chLen         = Props(5)                # Tensile characteristic length
    fr            = params['fr']            # Shear strength ratio
    sen_c         = params['sen_c']         # Softening exponent
    fc            = params['fc']            # Compressive Yield Strength
    RinHardMod    = params['RinHardMod']    # Initial hardening modulus ratio
    tsrn_e        = params['tsrn_e']        # Transitional Strain ratio
    dk1           = params['dk1']           # Deviatoric strain threshold ratio
    dk2           = params['dk2']           # Deviatoric damage parameter
    fmu_0         = params['fmu_0']         # Initial friction
    fmu_inf       = params['fmu_inf']       # Asymptotic friction
    sf0           = params['sf0']           # Transitional stress 
    DensRatio     = params['DensRatio']     # Densification ratio  
    beta          = params['beta']          # Volumetric deviatoric coupling
    unkt          = params['unkt']          # Tensile unloading parameter
    unks          = params['unks']          # Shear unloading parameter
    unkc          = params['unkc']          # Compressive unloading parameter
    Hr            = params['Hr']            # Shear softening modulus ratio
    dk3           = params['dk3']           # Final hardening modulus ratio
    EAF           = params['EAF']           # ElasticAnalysisFlag

    dt = params['dt']
    st = info['st']
    aKt = info['aKt']
    aLength = info['edge_l']

    EN = E0
    ET = alpha*E0
    epsN, epsM, epsL = eps
    DepsN, DepsM, DepsL = epsN - stv[4], epsM - stv[5], epsL - stv[6]

    def facet_failure(stv):
        stv = stv.at[1].set(0.)
        stv = stv.at[2].set(0.)
        stv = stv.at[3].set(0.)
        stv = stv.at[4].add(DepsN) # Why only add DepsN?

        stv_tmp = stv.at[13].add(DepsN*aLength)
        stv_tmp = stv_tmp.at[14].add(DepsM*aLength)
        stv_tmp = stv_tmp.at[11].add(DepsL*aLength)
        stv_tmp = stv_tmp.at[15].set(np.sqrt(np.abs(stv_tmp[13]**2 + stv_tmp[14]**2 + stv_tmp[11]**2)))
        stv = np.where(stv[4] >= 0., stv_tmp, stv)
        
        return stv
    
    def not_facet_failure(stv):
        
        def elastic_response(stv):
            stv = stv.at[1].add(EN*DepsN)
            stv = stv.at[2].add(ET*DepsM)
            stv = stv.at[3].add(ET*DepsL)
            stv = stv.at[4].add(DepsN)
            stv = stv.at[5].add(DepsM)
            stv = stv.at[6].add(DepsL)
            return stv

        def not_elastic_response(stv):

            # Parameters Initializing    
            sqalpha = np.sqrt(alpha)   
            ENc = EN
            ETc = ET  
            stv = stv.at[21].set(ENc)

            sigN0 = stv[1]
            sigM0 = stv[2]
            sigL0 = stv[3]

            fs = fr*ft
            ss = fs
            sc = fc

            # Shear strength and post-peak slope
            Hs = Hr * E0
            ss = fs

            # Old effective Strain
            csiO = np.sqrt(np.abs(stv[4]**2 + alpha * (stv[5]**2 + stv[6]**2)))

            stv = stv.at[4].add(DepsN)
            stv = stv.at[5].add(DepsM)
            stv = stv.at[6].add(DepsL)

            epsN = stv[4]  # Normal Strain
            epsT = np.sqrt(np.abs(stv[5]**2 + stv[6]**2))  # Total Shear Strain
            csi  = np.sqrt(np.abs(epsN**2 + alpha * epsT**2)) # New Effective Strain
            epsD = epsN - epsV
            Dcsi = csi - csiO

            # Coupling Variable
            teta = np.arctan(epsN/(epsT * sqalpha + 1e-10))
            ateta = teta*2./np.pi

            # effective old stress
            Stot0 = np.sqrt(np.abs(stv[1]**2 + (stv[2]**2 + stv[3]**2)/alpha))  

            stv = stv.at[12].set(np.minimum(stv[12], epsN)) # Min Normal Strain
            stv = stv.at[7].set(np.maximum(stv[7], epsN)) # Max Normal Strain
            stv = stv.at[8].set(np.maximum(stv[8], epsT)) # Max Shear Strain

            csiMax = np.sqrt(np.abs(stv[7]**2 + alpha * stv[8]**2))  # Max Effective Strain

            # TODO: rate effect?
            Fdyn = 1.

            def fracture_response(stv):
                steta  = np.sin(teta)
                steta2 = steta * steta
                cteta  = np.cos(teta)
                cteta2 = cteta * cteta

                # checkify.check(st >= 1e-10, f'Warning: st = {st} in critical range; continuation may produce numerical issues.')

                rat = ss / sqalpha / st
                rat2 = rat*rat 
                EPS = 1e-3
                s0 = np.where(cteta > EPS, st*(-steta + np.sqrt(np.abs(steta2 + 4.*cteta2/rat2))) / (2.*cteta2/rat2), st)

                ep0 = s0 / E0
                # checkify.check((ateta >= 0.) | (sen_c >= 1.), 'Warning: ateta and sec_c in ranges to produce exponent failure')

                H = Hs / alpha + (aKt - Hs / alpha) * ateta**sen_c
                tmp = csiMax - Fdyn * ep0       
                bound_f_tmp = np.where(tmp < 0., Fdyn * s0, Fdyn * s0 * np.exp(-H * tmp / s0))
                tmp1 = unkt * (csiMax - bound_f_tmp / E0)
                bound_f = np.where(csi > tmp1, bound_f_tmp, 0.)

                # New effective Stress
                Stot = np.minimum(bound_f , np.maximum(0., Stot0 + E0 * Dcsi))

                sigN = safe_divide(Stot * stv[4], csi)
                sigM = safe_divide(Stot * alpha * stv[5], csi)
                sigL = safe_divide(Stot * alpha * stv[6], csi)

                stv = stv.at[1].set(sigN)
                stv = stv.at[2].set(sigM)
                stv = stv.at[3].set(sigL)

                return stv

            def not_fracture_response(stv):
                ec = sc / E0
                ec0 = tsrn_e * ec     
                ENc_local = np.where(sigN0 < -sc, DensRatio * E0, ENc)
                stv = stv.at[21].set(ENc_local)

                phiD  = 1.
                epsV0 = 0.1*ec
                aKc   = RinHardMod*E0
                aKc1  = E0*dk3
                tmp = np.where(epsV < 0., dk2*(-np.abs(epsD) / (epsV-epsV0) - dk1), dk2*(np.abs(epsD) / (epsV0) - dk1))
                phiD = np.where(tmp >= 0., 1. / (1. + tmp), phiD)
                aKcc =  (aKc - aKc1) * phiD + aKc1

                # aKc = RinHardMod*E0
                # rDV = epsD/epsV
                # aKcc = aKc/(1. + dk2*np.maximum(rDV - dk1, 0.))

                epsEq = epsV + beta * epsD
                tmp = epsEq + ec
                tmp1 = epsEq + ec0

                sc0 = sc + aKcc * (ec0 - ec)
                bound_N_tmp = np.where((tmp < 0.) & (tmp1 > 0.), -sc + aKcc*tmp, np.where(tmp1 < 0.,  -sc0 * np.exp(-aKcc*tmp1/sc0), -sc))
                
                bound_N = bound_N_tmp

                sigN = np.maximum(bound_N, np.minimum(0., sigN0 + ENc_local * DepsN))                

                # Damage of the Cohesive Component 
                ssD  = Fdyn * ss
                tmp2 = csiMax / sqalpha - ssD / ET
            
                ssDdam = np.where(tmp2 < 0., ssD, ssD * np.exp(-Hs*tmp2/ss))

                # Shear Boundary
                dmu = fmu_0 - fmu_inf                   
                bound_T = ssDdam + dmu * sf0 - fmu_inf * sigN - dmu * sf0 * np.exp(sigN / sf0)                

                # New Tangential Stresses
                sigMe = sigM0 + ETc * DepsM
                sigLe = sigL0 + ETc * DepsL
                sigTe = np.sqrt(np.abs(sigMe * sigMe + sigLe * sigLe))
                sigT  = np.minimum(bound_T, np.maximum(0., sigTe))
                sigM = safe_divide(sigT * sigMe, sigTe)
                sigL = safe_divide(sigT * sigLe, sigTe)

                stv = stv.at[1].set(sigN)
                stv = stv.at[2].set(sigM)
                stv = stv.at[3].set(sigL)

                return stv

            fracture_flag = (epsN >= 0.) # >= or >?
            stv = jax.lax.cond(fracture_flag, fracture_response, not_fracture_response, stv)  

            ENc = stv[21]
            DepsIN = DepsN - (stv[1] - sigN0) / ENc
            DepsIM = DepsM - (stv[2] - sigM0) / ETc
            DepsIL = DepsL - (stv[3] - sigL0) / ETc

            # increment of dissipated energy density
            DDE = 0.5 * (stv[1] + sigN0) * DepsIN
            DDE = DDE + 0.5 * (stv[2] + sigM0) * DepsIM
            DDE = DDE + 0.5 * (stv[3] + sigL0) * DepsIL

            stv = stv.at[17].set(DDE)
            stv = stv.at[18].set(stv[18] + stv[17]) # dissipated energy per unit volume
            stv = stv.at[17].set(np.where(dt > 0., stv[17]/dt, stv[17])) # dissipated power per unit volume

            def tmp_fn(stv):
                stv = stv.at[13].add(aLength * DepsIN) # wN
                stv = stv.at[14].add(aLength * DepsIM) # wM
                stv = stv.at[11].add(aLength * DepsIL) # wL
                stv = stv.at[15].set(np.sqrt(np.abs(stv[13]**2 + stv[14]**2 + stv[11]**2))) # wtotal
                stv = stv.at[19].set(DDE)
                stv = stv.at[20].set(stv[20] + stv[19]) # dissipated energy during fracture per unit volume
                stv = stv.at[19].set(np.where(dt > 0., stv[19]/dt, stv[19])) # dissipated power during fracture per unit volume
                return stv

            stv = jax.lax.cond(stv[4] >= 0., tmp_fn, lambda x: x, stv)
            return stv

        elastic_flag = (EAF == 1)  # Elastic Behavior
        return jax.lax.cond(elastic_flag, elastic_response, not_elastic_response, stv)

    stv = stv.at[16].set(epsV)
    facet_fail_flag = stv[16] > 0.2
    stv = jax.lax.cond(facet_fail_flag, facet_failure, not_facet_failure, stv)

    stv = stv.at[0].set(np.where(facet_fail_flag, 1., 0.))

    return stv


def test_cond():
    print(f"start")

    x1 = np.array(onp.random.rand(10000, 10000))
    x2 = np.array(onp.random.rand(10000, 10000))

    def expensive_f():
        x = x1 @ x2
        x = x[0, 0]
        return x

    a = np.where(True, 1.,  expensive_f())
    print(a)

    def f1():
        return 1.

    def f2():
        return expensive_f()

    def single(x):
        b = jax.lax.cond(x > 0, f1, f2)
        return b

    vmap_single = jax.jit(jax.vmap(single))

    x = np.ones(10)
    bs = vmap_single(x)
    print(bs[0])

    bs = vmap_single(x)
    print(bs[0])


if __name__ == '__main__':
    test_cond()
