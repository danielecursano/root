import ROOT
import numpy as np

from hls4ml.backends import FPGABackend
from hls4ml.model.flow import register_flow

class SofieBackend(FPGABackend):
    def __init__(self):
        super().__init__('Sofie')
        initializers = self._get_layer_initializers()
        self._default_flow = register_flow('init_layers', initializers, requires=[], backend=self.name)
        self._writer_flow = register_flow('write', ['make_stamp', 'sofie:write_hls'], requires=[self._default_flow], backend=self.name)
        # TODO add flows for optimization?
        
    def create_initial_config(self, **kwargs):
        return dict()
        
    def get_default_flow(self):
        return self._default_flow
            
    def get_writer_flow(self):
        return self._writer_flow
    
    @staticmethod
    def get_sofie_session(model):
        header_path = model.config.get_output_dir() + "/" + model.config.get_project_name()
        ROOT.gInterpreter.Declare(f'#include "{header_path}.hxx"')
        sofie_project = getattr(ROOT, f"TMVA_SOFIE_{model.config.get_project_name()}", None)
        if not sofie_project:
            raise RuntimeError(f"SOFIE namespace TMVA_SOFIE_{model.config.get_project_name()} not found.")
        session = sofie_project.Session(header_path+".dat")
        return session
        
        
        
