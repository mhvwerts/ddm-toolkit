#%% Insert path to import
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')
#%%
from sys import argv
from configparser import ConfigParser
#%%
from .defaultSimulationParams import default_sim_params

class DDMParams:
    """New version of unified DDM Toolkit parameter class

    Documentation to be updated from ddm_toolkit.parameters etc.

    in its basic version, it generates an empty parameter object
    to be filled with the necessary parameters

    """

    #TODO : Write a proper class init

    #def __init__(self):


        #return params

    @classmethod
    def fromSimulationConfigFile(cls, fpn):
        """
        This returns a DDMParams object with parameters
        populated using a configuration file for simulated data
        """
        CP = ConfigParser()
        CP.read(fpn)
        #Create and POPULATE class
        params = cls()
        params.configfile = fpn
        params.populate_from_configparser_sim(CP)
        return params

    @classmethod
    def defaultSimulationParams(cls):
        """
        Generate a DDMParams object filled with the 'default'
        simulation parameters


        Returns
        -------
        Instance of DDMParams class, populated with default simulation
        parameters

        """
        global default_sim_params

        CP = ConfigParser()
        CP.read_string(default_sim_params)
        #Create and POPULATE class
        params = cls()
        params.populate_from_configparser_sim(CP)
        return params

    @classmethod
    def fromRealVideoConfigFile(cls, fpn):
        """
        Create a DDMParams object from a real video configuration file.

        Parameters
        ----------
        fpn : string
            Pathname of configuration file (for real video).

        Returns
        -------
        Instance of DDMParams class, populated with parameters from
        configuration file.

        """

        params = cls()

        CP = ConfigParser(interpolation=None)
        CP.read(fpn)

        # Temperature required!
        try:
            params.T_K = float(CP['general']['temp_deg_C']) + 273.15
        except KeyError:
            pass

        # fnbase, fnext = os.path.splitext(fpn)

        params.videofpn = CP['videofile']['pathname']
        params.frm_start = int(CP['videofile']['frm_start'])
        params.frm_end = int(CP['videofile']['frm_end'])
        params.segmentation_flag = CP['videofile']['segmentation_flag']

        params.ROI_size = int(CP['videofile']['ROI_size'])
        if not params.ROI_size < 0:
            params.ROI_x1 = int(CP['videofile']['ROI_x'])
            params.ROI_y1 = int(CP['videofile']['ROI_y'])
            params.ROI_x2 = params.ROI_x1 + params.ROI_size
            params.ROI_y2 = params.ROI_y1 + params.ROI_size


        # TO DO: rethink how to fine-tune video preview
        #  (set vmin, vmax)
        #  especially, between darkfield and brightfield videos
        params.frm_Npreview = int(CP['videofile']['frm_Npreview'])
        params.vid_overdrive = float(CP['videofile']['display_overdrive'])
        params.ROIcontrast = float(CP['videofile']['display_ROIcontrast'])
        params.apodization = CP['videofile']['apodization']
        params.img_overdrive = int(CP['videofile']['img_overdrive'])


        # params.ISE_Npx needs to be set once params.ROI_size is known...
        params.ISE_Nbuf = int(CP['ISEngine']['ISE_Nbuf'])

        params.doISFradialaverage = False
        try:
            if CP['ISEngine']['ISF_radialaverage']=='True':
                params.doISFradialaverage = True
        except KeyError:
            pass

        try:
            params.ISE_type = int(CP['ISEngine']['ISE_type'])
        except KeyError:
            params.ISE_type = 0


        params.um_p_pix = float(CP['realworld']['px_size'])
        params.s_p_frame = float(CP['realworld']['frm_period'])

        # in certain cases, the ISF has been radially averaged
        # before saving, in order to save space
        # in that case, ISF_radialaverage is set to True
        params.ISF_radialaverage = False
        try:
            if CP['ISEngine']['ISF_radialaverage']=='True':
                params.ISF_radialaverage = True
        except KeyError:
            pass

        params.ISF_outfpn = CP['ISEngine']['ISF_outfpn']

        #TODO: rethink how to fine-tune ISF preview contrast
        # (set vmix and vmax...)
        imgoverdrive = 2.1 # default value
        try:
            imgoverdrive = float(CP['ISEngine']['ISF_display_overdrive'])
        except KeyError:
            pass
        params.ISF_viewer_overdrive = imgoverdrive

        params.D_guess = float(CP['analysis_brownian']['D_guess'])



        return params



    def populate_from_configparser_sim(self, CP):
        self.sim_D = float(CP['simulation']['D'])
        self.sim_Np = int(CP['simulation']['Np'])

        self.sim_bl = float(CP['simulation']['bl'])

        self.sim_Nt = int(CP['simulation']['Nt'])
        self.sim_T = float(CP['simulation']['T'])

        self.sim_img_border = float(CP['imgsynth']['img_border'])
        self.sim_img_w = float(CP['imgsynth']['img_w'])
        self.sim_img_Npx = int(CP['imgsynth']['img_Npx'])
        self.img_Npx = self.sim_img_Npx

        try :
            self.sim_v = float(CP['simulation']['v'])
        except KeyError:
            self.sim_v = None

        try :
            self.sim_phi_0 = float(CP['simulation']['phi'])
        except KeyError:
            self.sim_phi_0 = None
        try:
            self.sim_img_I_offset = float(CP['imgsynth']['img_I_offset'])
        except KeyError:
            self.sim_img_I_offset = None
        try:
            self.sim_img_I_noise = float(CP['imgsynth']['img_I_noise'])
        except KeyError:
            self.sim_img_I_noise = -1.0
        self.vidfpn = CP['imgsynth']['vidfpn']


        # videoplayer settings (TODO: document)
        video_overdrive = 1.0
        try:
            video_overdrive = float(CP['videoplayer']['video_overdrive'])
        except KeyError:
            pass
        self.video_overdrive = video_overdrive

        self.video_Nview = int(CP['videoplayer']['Nview'])

        try:
            self.ISE_type = int(CP['ISEngine']['ISE_type'])
        except KeyError:
            self.ISE_type = 0
        self.ISE_Nbuf = int(CP['ISEngine']['ISE_Nbuf'])
        self.ISE_Npx = self.img_Npx
        self.ISF_outfpn = CP['ISEngine']['ISF_outfpn']

        # in certain cases, the ISF has been radially averaged
        # before saving, in order to save space
        # in that case, ISF_radialaverage is set to True
        self.ISF_radialaverage = False
        try:
            if CP['ISEngine']['ISF_radialaverage']=='True':
                self.ISF_radialaverage = True
        except KeyError:
            pass

        imgoverdrive = 2.1 # default value
        try:
            imgoverdrive = float(CP['ISEngine']['ISF_display_overdrive'])
        except KeyError:
            pass
        self.ISF_viewer_overdrive = imgoverdrive

        self.D_guess = float(CP['analysis_brownian']['D_guess'])

        # derived parameters
        self.sim_dt = self.sim_T/self.sim_Nt
        self.img_l = (self.sim_bl + 2*self.sim_img_border)
        self.um_p_pix = self.img_l/self.img_Npx
        self.s_p_frame = self.sim_dt
        self.Nframes = self.sim_Nt


    def update_simulation_parameters(self):
        """update parameters derived from the simulation parameters

        for example, the simulation time step sim_dt is calculated from sim_T and sim_Nt

        only use this with simulations, not with real video

        TODO: this method should disappear (become transparent)
        """

        # TODO: the following should be automatic (copy to where sim_dt
        # is being used...)
        self.sim_dt = self.sim_T/self.sim_Nt

        # 'real world' video parameters
        self.s_p_frame = self.sim_dt
        self.Nframes = self.sim_Nt
        self.img_Npx = self.sim_img_Npx
        self.img_l = (self.sim_bl + 2*self.sim_img_border)
        self.um_p_pix = self.img_l/self.img_Npx
