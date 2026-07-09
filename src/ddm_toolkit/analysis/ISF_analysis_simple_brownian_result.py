#
from matplotlib import pyplot as plt
import numpy as np
#
class ISFanalysis_simple_brownian_result:
    def print_report(self):
        print('D_guess (user):', self.D_guess,'µm2/s')
        print('D_guess (refined):', self.D_guess_refined, 'µm2/s')
        print('q_low: ', self.q_low,'µm-1')
        print('q_opt: ', self.q_opt,'µm-1')
        print('q_high:', self.q_high,'µm-1')
        print('q_max: ', self.q[-1],'µm-1')
        print('D (fit):', self.D_fit,
              'µm2/s (+/- ', self.D_fit_CI95, ', 95% CI)')

    def show_ISF_radavg(self):
        plt.figure("ISF (radially averaged)")
        plt.clf()
        plt.subplot(211)
        plt.title('radially averaged ISF')
        for itau in range(1,11):
            plt.plot(self.q[1:],self.radISF_qtau[itau,1:])
        for itau in range(20,len(self.tau),10):
            plt.plot(self.q[1:],self.radISF_qtau[itau,1:])
        plt.xlabel('q [µm-1]')
        plt.ylabel('ISF')
        plt.subplot(212)
        plt.imshow(self.radISF_qtau[1:,1:].T, origin = 'lower', aspect ='auto',
                   extent =(self.tau[1], self.tau[-1],
                            self.q[1], self.q[-1]))
        plt.colorbar()
        plt.ylabel('q [µm-1]')
        plt.xlabel('tau [s]')

    def show_fits(self):
        plt.figure('Fits at q_low, q_opt, q_high')
        plt.clf()
        plt.subplot(311)
        plt.title('Fits at q_low, q_opt, q_high (refined D_guess)')
        plt.plot(self.tau[1:], self.radISF_qtau[1:,self.iq_low],'o')
        plt.plot(self.tau, self.ISFmodelfit_qlow)
        plt.ylabel('ISF')
        plt.subplot(312)
        plt.plot(self.tau[1:], self.radISF_qtau[1:,self.iq_opt],'o')
        plt.plot(self.tau, self.ISFmodelfit_qopt)
        plt.ylabel('ISF')
        plt.subplot(313)
        plt.plot(self.tau[1:], self.radISF_qtau[1:,self.iq_high],'o')
        plt.plot(self.tau, self.ISFmodelfit_qhigh)
        plt.ylabel('ISF')
        plt.xlabel('tau [s]')

    def show_Aq_Bq_kq(self, plot_B_A_ratio = False):
        plt.figure('Result of simple Brownian analysis of DDM')
        plt.clf()

        plt.subplot(311)
        plt.title('Simple Brownian model analysis of ISF')
        plt.plot(self.q[self.iq_low:self.iq_high],
                 self.k_q[self.iq_low:self.iq_high],'o')
        plt.plot(self.q, self.D_guess*self.q**2, label = 'init. guess')
        plt.plot(self.q, self.D_fit*self.q**2, label = 'fit')
        plt.ylim(0, np.max(self.k_q)*1.3)
        plt.xlim(xmin=0,xmax=self.q[-1])
        plt.ylabel('k(q) [s-1]')
        plt.legend()

        plt.subplot(312)
        plt.plot(self.q[self.iq_low:self.iq_high],
                 self.A_q[self.iq_low:self.iq_high],'o')
        plt.ylim(ymin=0)
        plt.xlim(xmin=0,xmax=self.q[-1])
        plt.ylabel('A(q) [a.u.]')

        plt.subplot(313)
        if plot_B_A_ratio:
            plt.plot(self.q[self.iq_low:self.iq_high],
                     self.B_q[self.iq_low:self.iq_high]/self.A_q[self.iq_low:self.iq_high],'o')
            plt.xlim(xmin=0,xmax=self.q[-1])
            plt.ylabel('B(q)/A(q)')
            plt.xlabel('q [µm-1]')
        else:
            plt.plot(self.q[self.iq_low:self.iq_high],
                     self.B_q[self.iq_low:self.iq_high],'o')
            plt.xlim(xmin=0,xmax=self.q[-1])
            plt.ylabel('B(q) [a.u.]')
            plt.xlabel('q [µm-1]')
