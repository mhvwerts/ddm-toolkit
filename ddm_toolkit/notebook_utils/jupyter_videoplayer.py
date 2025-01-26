import threading
from time import sleep

import ipywidgets as widgets
from IPython.display import display


        
class VideoPlayerUI:
    def __init__(self, framestreamer_instance):
        self.framestrm = framestreamer_instance
        self.videoplaying = False
        self.playvideosleeptime = 0.05 # ~20 image updates per second
        self._init_playvideothread()
        self._setup_UI_box()
        self.frameslide.disabled = not self.framestrm.random_access

    def _setup_UI_box(self):
        self.imgwdg = widgets.Image(
            value = self.framestrm.imgbytes(),
            format = 'png',
            # width = 100 #,
            # height = 100
            )
        self.button1 = widgets.Button(
            description='Start',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='' # (FontAwesome names without the `fa-` prefix)
            )
        self.button1.on_click(self._on_button1_clicked)
        self.button2 = widgets.Button(
            description='Stop',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='' # (FontAwesome names without the `fa-` prefix)
            )
        self.button2.on_click(self._on_button2_clicked)
        # self.rangeslide = widgets.IntRangeSlider(
        #     value=[50, 700],
        #     min=0,
        #     max=1000,
        #     step=1,
        #     description='Loop:',
        #     disabled=False,
        #     continuous_update=False,
        #     orientation='horizontal',
        #     readout=True,
        #    readout_format='d',
        #     )
        self.frameslide = widgets.IntSlider(
            value=self.framestrm.frameix,
            min=0,
            max=self.framestrm.Nframes-1,
            step=1,
            description='Frame#:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        # self.frameslide.observe(self._on_set_frame, names='value') 
        ##  avoid using a callback since changing the slidervalue in the code also triggers it
        self.set_vmin = widgets.FloatText(
            value=self.framestrm.vmin,
            description='Vmin:',
            disabled=False
            )
        self.set_vmax = widgets.FloatText(
            value=self.framestrm.vmax,
            description='Vmax:',
            disabled=False
            )
        self.set_vmin.observe(self._on_set_vmin, names='value')
        self.set_vmax.observe(self._on_set_vmax, names='value')
        
        self.wdgROIcontrol = widgets.Checkbox(
            value=False,
            description='ROI active',
            disabled=False,
            indent=True
        )
        self.wdgROIsize = widgets.Dropdown(
                              options = [('32', 32),
                                         ('64', 64),
                                         ('128', 128),
                                         ('256', 256),
                                         ('512', 512),
                                        ],
                              value = 32,
                              description = 'ROI size:'
                          )
        self.wdgROIx = widgets.BoundedIntText(
            value=self.framestrm.ROI_ix,
            min=0,
            max=self.framestrm.viewport[1],
            step=1,
            description='ROI x:',
            disabled=False
            )
        self.wdgROIy = widgets.BoundedIntText(
            value=self.framestrm.ROI_iy,
            min=0,
            max=self.framestrm.viewport[0],
            step=1,
            description='ROI y:',
            disabled=False
            )
        self.wdgROIcontrol.observe(self._on_wdgROIcontrol, names='value')
        self.wdgROIsize.observe(self._on_wdgROIsize, names='value')
        self.wdgROIx.observe(self._on_wdgROIx, names='value')
        self.wdgROIy.observe(self._on_wdgROIy, names='value')
        
        self.buttonbox = widgets.HBox([self.button1, self.button2])
        rbox = widgets.VBox([self.buttonbox,
                             self.frameslide, 
                             self.set_vmin, self.set_vmax,
                             self.wdgROIcontrol,
                             self.wdgROIsize,
                             self.wdgROIx, self.wdgROIy])
        self.UIbox = widgets.HBox([self.imgwdg, rbox])
         
    def _on_button1_clicked(self, b):
        if self.framestrm.random_access:
            self.framestrm.frameix = self.frameslide.value
        if not self.playvideothread.is_alive():
            self.videoplaying = True
            self.frameslide.disabled = True
            self._init_playvideothread()
            self.playvideothread.start()
            
    def _on_button2_clicked(self, b):
        self.stop_videoplayer()
        
    def _on_set_vmin(self, change):
        self.framestrm.vmin = change['new']
        
    def _on_set_vmax(self, change):
        self.framestrm.vmax = change['new']
        
    def _on_wdgROIx(self, change):
        self.framestrm.ROI_ix = change['new']
        
    def _on_wdgROIy(self, change):
        self.framestrm.ROI_iy = change['new']
        
    def _on_wdgROIsize(self, change):
        self.framestrm.ROI_iw = change['new']
            
    def _on_wdgROIcontrol(self, change):
        self.framestrm.ROI_active = change['new']
        self.framestrm.ROI_iw = self.wdgROIsize.value
        
    # def _on_set_frame(self, change): # also called when value of slider is changed by code
    # #AVOID TO USE THIS
    #     self.framestrm.frameix = change['new']
        
    def _init_playvideothread(self):
        self.playvideothread = threading.Thread(target=self._playvideo, args=()) 
        
    def _playvideo(self):
        while self.videoplaying:
            framedata = self.framestrm.next_frame()
            if framedata is None: 
                # If framestreamser serves 'None'
                # it is at the end of the video or otherwise
                # unable to play back. Stop video player.
                self.stop_videoplayer()
            else:
                self.imgwdg.value = self.framestrm.imgbytes()
                self.frameslide.value = self.framestrm.frameix
                sleep(self.playvideosleeptime)
    
    def updateUIparameters(self):
        self.wdgROIsize.value = self.framestrm.ROI_iw
        self.wdgROIx.value = self.framestrm.ROI_ix
        self.wdgROIy.value = self.framestrm.ROI_iy   
        
    def showUIbox(self):
        self.updateUIparameters()
        display(self.UIbox)
        
    def stop_videoplayer(self):
        self.videoplaying = False
        try:
            self.playvideothread.join()
        except RuntimeError:
            pass # If thread  not running, it cannot be joined. Do nothing.
        self.frameslide.disabled = not self.framestrm.random_access
        
